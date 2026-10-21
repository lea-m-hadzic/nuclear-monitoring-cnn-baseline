"""Hard-negative PILOT (80 sites): sourcing, planning, post-extraction QA, review artifacts, diagnostics.

    python hard_negative_pilot.py source     # GPPD (local CSV) + OpenStreetMap/Overpass -> hard_negative_points.csv
    python hard_negative_pilot.py plan       # validation, jitter, split -> hard_negative_tile_plan.csv
    # extraction: extract_external_tiles.py --plan .../hard_negative_tile_plan.csv --preset auto_mosaic_1p9km ...
    python hard_negative_pilot.py post       # QA, review sheet/CSV/HTML, distributions, diagnostics A/B/C

Sourcing (deterministic given the cached source data; all raw responses are cached under
external_imagery/sources/, git-ignored):
  * thermal_power_plant: WRI Global Power Plant Database v1.3.0 (CC BY 4.0), primary_fuel Coal/Gas/Oil,
    capacity_mw >= 500; per broad region, plants are tried in SHA-256(seed | gppd_idnr) order.
  * other categories: OpenStreetMap via Overpass. For each (category, region) slot an "anchor" positive of that
    region is taken in seeded order (only to find similar terrain/region); OSM ways/relations with the category
    tags and a minimum bounding-box area are fetched within 150 km of it, and candidates are tried in seeded order.
  * a candidate is accepted only if: >= 20 km from every external positive, outside every evaluation AOI + buffer,
    no nuclear-like name or tag, no military area / military=* / prison / nuclear plant within 2 km (checked
    against OSM), >= 2 km from every already-accepted hard negative (no co-located duplicates).
Label rule: every tile starts `unreviewed`. A tile showing a recognizable layered fence is marked
`reject_target_signature` (kept for audit), never relabelled. Only `confirmed_hard_negative` may be used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

from dataset_audit import REPORT_DIR
from dataset_plan import broad_region, fence_pixel
from geo import chebyshev_m, destination, haversine_m, unit_floats
from geometry import REPO_ROOT, load_config, tile_geometry
from hard_negatives import NUCLEAR_NAME_RE, assign_hard_negative_split, build_plan, validate_points
from negative_refinement import (CLASSIFIER_FEATURES, NUISANCE, auto_reject, compute_features,
                                 shortcut_diagnostics)

PRESET = "auto_mosaic_1p9km"
HN_STATUSES = ["unreviewed", "confirmed_hard_negative", "reject_target_signature", "reject_artifact", "reject_ambiguous"]
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",          # main instance
                 "https://overpass.kumi.systems/api/interpreter",    # public mirrors (same OSM data, own load)
                 "https://overpass.private.coffee/api/interpreter"]
USER_AGENT = "nuclear-monitoring-benchmark research (hard-negative pilot sampling)"
GPPD_FILE = "external_imagery/sources/gppd/global_power_plant_database.csv"
GPPD_VERSION = "WRI GPPD v1.3.0 (github wri/global-power-plant-database @7a91cfb, 2022-01-26), CC BY 4.0"
SEARCH_RADIUS_KM = 150
SEARCH_RADIUS_KM_BY_CATEGORY: dict[str, int] = {}   # a 300 km refinery search timed out on every Overpass server (2026-10-14)
SAME_CATEGORY_MIN_KM = 50      # spread: sites of one category accepted from the same anchor area are >= 50 km apart
EXCLUDE_KM = 2.0

# OSM category definitions: Overpass filters (ways/relations) + minimum bounding-box area (ha)
OSM_CATEGORIES = {
    "refinery_petrochemical": (['["industrial"~"^(refinery|petrochemical|oil_refinery)$"]',
                                '["man_made"="works"]["product"~"oil|petrol|petrochem|chemical",i]'], 10),
    "hv_substation": (['["power"="substation"]["voltage"~"(^|;)(345|380|400|420|500|525|735|750|765|800|1000|1100|1150)000($|;)"]'], 5),
    "rail_yard_port": (['["landuse"="railway"]["railway"~"yard"]', '["railway"="yard"]', '["landuse"="port"]',
                        '["industrial"="port"]'], 20),
    "logistics_warehouse": (['["industrial"~"^(warehouse|logistics|distribution|depot)$"]',
                             '["building"="warehouse"]', '["landuse"="industrial"]["industrial"="warehouse"]'], 10),
    "water_treatment": (['["man_made"~"^(wastewater_plant|water_works)$"]'], 5),
}
EXCLUSION_FILTERS = ['["landuse"="military"]', '["military"]', '["amenity"="prison"]',
                     '["plant:source"="nuclear"]', '["generator:source"="nuclear"]']


def h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def ddir(cfg) -> Path:
    return REPO_ROOT / cfg["dataset"]["dataset_dir"]


# ------------------------------------------------------------------ Overpass (cached, polite)

def overpass(query: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{h(query)[:24]}.json"
    if f.exists():
        return json.loads(f.read_text())
    for attempt in range(9):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        t0 = time.time()
        try:
            r = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=300)
        except requests.RequestException as e:
            print(f"  overpass error {type(e).__name__}; retry {attempt + 1}", flush=True)
            time.sleep(20 * (attempt + 1))
            continue
        print(f"  overpass {url.split('/')[2]} {r.status_code} in {time.time() - t0:.1f}s ({len(r.content) / 1e6:.2f} MB)", flush=True)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            d = r.json()
            if d.get("remark", "").lower().startswith("runtime error"):
                print(f"  overpass remark: {d['remark'][:120]}", flush=True)
                time.sleep(10)
                continue
            d["endpoint"] = url
            f.write_text(json.dumps({"query": query, **d}))
            time.sleep(2.0)
            return d
        time.sleep(5 if attempt < len(OVERPASS_URLS) else 20 * (attempt // len(OVERPASS_URLS)))   # rotate, then back off
    raise SystemExit("Overpass failed repeatedly; cached results are kept, rerun later")


def bbox_area_ha(b: dict) -> float:
    lat = (b["minlat"] + b["maxlat"]) / 2
    w = haversine_m(lat, b["minlon"], lat, b["maxlon"])
    hgt = haversine_m(b["minlat"], b["minlon"], b["maxlat"], b["minlon"])
    return float(w * hgt / 1e4)


def dist_to_bbox_m(lat, lon, b) -> float:
    """Distance from a point to an OSM bounding box (0 inside); exclusions are areas, not points."""
    clat = min(max(lat, b["minlat"]), b["maxlat"])
    clon = min(max(lon, b["minlon"]), b["maxlon"])
    return float(haversine_m(lat, lon, clat, clon))


def exclusion_elements_multi(points: list[tuple[float, float]], radius_m, cache, fetch=None) -> list[dict]:
    """Military/prison/nuclear features near ANY of the points, in one request (union of around-clauses)."""
    clauses = "".join(f"nwr{f}(around:{int(radius_m)},{la:.5f},{lo:.5f});" for la, lo in points for f in EXCLUSION_FILTERS)
    query = "[out:json][timeout:180];(" + clauses + ");out bb tags;"
    els = (fetch(query) if fetch else overpass(query, cache)).get("elements", [])
    for e in els:
        if "bounds" not in e and "lat" in e:
            e["bounds"] = {"minlat": e["lat"], "maxlat": e["lat"], "minlon": e["lon"], "maxlon": e["lon"]}
    return [e for e in els if "bounds" in e]


def exclusion_elements(lat, lon, radius_m, cache) -> list[dict]:
    return exclusion_elements_multi([(lat, lon)], radius_m, cache)


def excluded_nearby(lat, lon, excl: list[dict]) -> str:
    for e in excl:
        if dist_to_bbox_m(lat, lon, e["bounds"]) < EXCLUDE_KM * 1000:
            t = e.get("tags", {})
            return "excluded_near:" + ",".join(f"{k}={t[k]}" for k in ("landuse", "military", "amenity", "plant:source",
                                                                       "generator:source") if k in t)
    return ""


# ------------------------------------------------------------------ sourcing

def candidate_ok(lat, lon, name, pos, accepted, cfg) -> str:
    """'' if acceptable, else the first failing reason (cheap checks; OSM exclusion check is separate)."""
    hcfg = cfg["hard_negatives"]
    if haversine_m(lat, lon, pos.latitude.to_numpy(), pos.longitude.to_numpy()).min() < hcfg["min_km_from_any_positive"] * 1000:
        return "too_close_to_positive"
    for n, a in cfg["evaluation_aois"].items():
        if haversine_m(lat, lon, a["lat"], a["lon"]) < (a["radius_km"] + cfg["aoi_exclusion_buffer_km"]) * 1000:
            return f"near_{n}_aoi"
    if pd.Series([name or ""]).str.contains(NUCLEAR_NAME_RE, case=False, regex=True).iloc[0]:
        return "nuclear_like_name"
    if accepted and haversine_m(lat, lon, np.array([a["latitude"] for a in accepted]),
                                np.array([a["longitude"] for a in accepted])).min() < max(hcfg["min_km_between_hard_negatives"], 2) * 1000:
        return "co_located_with_accepted"
    return ""


def shortlist_as_points(rows) -> list[dict]:
    return [{"latitude": r.latitude, "longitude": r.longitude} for r in rows]


class OverpassBudgetExceeded(Exception):
    pass


def source(cfg, categories: list[str] | None = None, deadline_s: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (accepted points, missing slots). Stops querying Overpass after `deadline_s` seconds of wall time;
    unfilled (category, region) slots are reported, never silently dropped."""
    t_start = time.time()
    seed = cfg["dataset"]["seed"]
    pos = pd.read_csv(ddir(cfg) / "positive_audit.csv")
    meta = pd.read_csv(ddir(cfg) / "external_tiles_metadata.csv")
    plan = build_plan(pos, meta, cfg)
    if categories:
        plan = plan[plan.category.isin(categories)]
    cache = REPO_ROOT / "external_imagery/sources/overpass"
    accepted, log, missing = [], [], []

    def q(query):          # Overpass with a wall-clock budget (cached responses are always free)
        if deadline_s is not None and time.time() - t_start > deadline_s and not (cache / f"{h(query)[:24]}.json").exists():
            raise OverpassBudgetExceeded()
        return overpass(query, cache)

    # power plants from GPPD
    g = pd.read_csv(REPO_ROOT / GPPD_FILE, low_memory=False)
    g = g[g.primary_fuel.isin(["Coal", "Gas", "Oil"]) & (g.capacity_mw >= 500)
          & ~g.other_fuel1.fillna("").eq("Nuclear")].copy()
    g["broad_region"] = [broad_region(a, b) for a, b in zip(g.latitude, g.longitude)]
    g["order"] = [h(seed, "gppd", i) for i in g.gppd_idnr]
    for _, slot in plan[plan.category == "thermal_power_plant"].iterrows():
        cands = g[g.broad_region == slot.broad_region].sort_values("order")
        need = int(slot.n_facilities)
        shortlist = []
        for r in cands.itertuples():                       # cheap checks first, 3x oversampled shortlist
            why = candidate_ok(r.latitude, r.longitude, r.name, pos, accepted + shortlist_as_points(shortlist), cfg)
            if why:
                log.append({"category": "thermal_power_plant", "source_id": r.gppd_idnr, "result": why})
                continue
            shortlist.append(r)
            if len(shortlist) >= 3 * need + 2:
                break
        try:
            excl = exclusion_elements_multi([(r.latitude, r.longitude) for r in shortlist], EXCLUDE_KM * 1000 + 500, cache,
                                            fetch=q) if shortlist else []
        except (OverpassBudgetExceeded, SystemExit):
            missing.append({"category": "thermal_power_plant", "broad_region": slot.broad_region, "missing": need,
                            "reason": "overpass_unavailable_or_budget"})
            continue
        for r in shortlist:
            if need == 0:
                break
            why = excluded_nearby(r.latitude, r.longitude, excl) or candidate_ok(r.latitude, r.longitude, r.name, pos, accepted, cfg)
            log.append({"category": "thermal_power_plant", "source_id": r.gppd_idnr, "result": why or "accepted"})
            if why:
                continue
            accepted.append({"category": "thermal_power_plant", "broad_region": slot.broad_region, "name": r.name,
                             "latitude": r.latitude, "longitude": r.longitude, "source": GPPD_VERSION,
                             "source_id": f"gppd:{r.gppd_idnr}", "source_detail": f"{r.primary_fuel}, {r.capacity_mw:.0f} MW, {r.country}",
                             "facility_area_ha": np.nan, "anchor_facility_id": ""})
            need -= 1
        if need:
            log.append({"category": "thermal_power_plant", "source_id": "", "result": f"shortfall {need} in {slot.broad_region}"})
            missing.append({"category": "thermal_power_plant", "broad_region": slot.broad_region, "missing": need,
                            "reason": "candidates_exhausted"})

    # OSM categories around seeded anchor positives
    for _, slot in plan[plan.category != "thermal_power_plant"].iterrows():
        filters, min_ha = OSM_CATEGORIES[slot.category]
        anchors = pos[pos.broad_region == slot.broad_region].copy()
        anchors["order"] = [h(seed, "anchor", slot.category, f) for f in anchors.facility_id]
        need = int(slot.n_facilities)
        per_anchor = max(1, math.ceil(need / 2))     # >= 2 anchor areas whenever need >= 2
        overpass_down = False
        print(f"{slot.category} / {slot.broad_region}: need {need}", flush=True)
        for a in anchors.sort_values("order").itertuples():
            if need == 0:
                break
            taken_here = []
            R = SEARCH_RADIUS_KM_BY_CATEGORY.get(slot.category, SEARCH_RADIUS_KM) * 1000
            qry = "[out:json][timeout:240];(" + "".join(
                f"way{f}(around:{R},{a.latitude:.5f},{a.longitude:.5f});relation{f}(around:{R},{a.latitude:.5f},{a.longitude:.5f});"
                for f in filters) + ");out bb tags;"
            try:
                d = q(qry)
            except (OverpassBudgetExceeded, SystemExit):
                overpass_down = True
                break
            osm_base = f'{d.get("osm3s", {}).get("timestamp_osm_base", "")} via {d.get("endpoint", "").split("/")[2] if d.get("endpoint") else "?"}'
            els = [e for e in d.get("elements", []) if "bounds" in e and e["type"] in ("way", "relation")
                   and bbox_area_ha(e["bounds"]) >= min_ha]
            els.sort(key=lambda e: h(seed, "osm", slot.category, e["type"], e["id"]))
            # cheap checks first; shortlist up to 3x what this anchor may contribute
            shortlist = []
            for e in els:
                b = e["bounds"]
                lat, lon = (b["minlat"] + b["maxlat"]) / 2, (b["minlon"] + b["maxlon"]) / 2
                tags = e.get("tags", {})
                name = tags.get("name", "") or tags.get("operator", "")
                why = candidate_ok(lat, lon, name, pos, accepted + [{"latitude": x[0], "longitude": x[1]} for x in shortlist], cfg)
                if not why and any(tags.get(k) == "nuclear" for k in ("plant:source", "generator:source")):
                    why = "nuclear_tag"
                if why:
                    log.append({"category": slot.category, "source_id": f"osm:{e['type']}/{e['id']}", "result": why})
                    continue
                shortlist.append((lat, lon, e, name, tags))
                if len(shortlist) >= 3 * per_anchor + 2:
                    break
            if not shortlist:
                continue
            try:
                excl = exclusion_elements_multi([(x[0], x[1]) for x in shortlist], EXCLUDE_KM * 1000 + 500, cache, fetch=q)
            except (OverpassBudgetExceeded, SystemExit):
                overpass_down = True
                break
            for lat, lon, e, name, tags in shortlist:
                why = candidate_ok(lat, lon, name, pos, accepted, cfg)
                if not why and taken_here and haversine_m(lat, lon, np.array([x[0] for x in taken_here]),
                                                          np.array([x[1] for x in taken_here])).min() < SAME_CATEGORY_MIN_KM * 1000:
                    why = "same_category_within_50km"
                if not why:
                    why = excluded_nearby(lat, lon, excl)
                log.append({"category": slot.category, "source_id": f"osm:{e['type']}/{e['id']}", "result": why or "accepted"})
                if why:
                    continue
                b = e["bounds"]
                accepted.append({"category": slot.category, "broad_region": broad_region(lat, lon), "name": name,
                                 "latitude": lat, "longitude": lon,
                                 "source": f"OpenStreetMap contributors (ODbL), Overpass, osm_base {osm_base}",
                                 "source_id": f"osm:{e['type']}/{e['id']}",
                                 "source_detail": ";".join(f"{k}={v}" for k, v in tags.items()
                                                           if k in ("industrial", "man_made", "product", "power", "voltage",
                                                                    "landuse", "railway", "building", "operator"))[:200],
                                 "facility_area_ha": round(bbox_area_ha(b), 1), "anchor_facility_id": a.facility_id})
                need -= 1
                taken_here.append((lat, lon))
                if need == 0 or len(taken_here) >= per_anchor:
                    break
        if need:
            log.append({"category": slot.category, "source_id": "", "result": f"shortfall {need} in {slot.broad_region}"})
            missing.append({"category": slot.category, "broad_region": slot.broad_region, "missing": need,
                            "reason": "overpass_unavailable_or_budget" if overpass_down else "candidates_exhausted"})
    pts = pd.DataFrame(accepted)
    if len(pts):
        pts.insert(0, "hn_id", [f"hn_{c}_{i + 1:02d}" for c, i in zip(pts.category, pts.groupby("category").cumcount())])
    tag = "_" + "_".join(categories) if categories else ""
    pd.DataFrame(log).to_csv(ddir(cfg) / f"hard_negative_sourcing_log{tag}.csv", index=False)
    return pts, pd.DataFrame(missing, columns=["category", "broad_region", "missing", "reason"])


# ------------------------------------------------------------------ plan

def plan_tiles(pts: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    pos = pd.read_csv(ddir(cfg) / "positive_audit.csv")
    v = validate_points(pts, pos, cfg)
    if not v["valid"].all():
        raise SystemExit(f"invalid hard-negative points: {v.loc[~v['valid'], ['hn_id', 'flags']].to_dict('records')}")
    v["split"] = assign_hard_negative_split(v, cfg)
    j, seed = cfg["dataset"]["jitter"], cfg["dataset"]["seed"]
    rows = []
    for r in v.itertuples():
        u_b, u_d = unit_floats(seed, r.hn_id, f"{r.hn_id}__hn", "jitter")
        bearing = 360.0 * u_b
        dist = math.sqrt(u_d * (j["max_m"] ** 2 - j["min_m"] ** 2) + j["min_m"] ** 2)
        clat, clon = destination(r.latitude, r.longitude, bearing, dist)
        fp = fence_pixel(clat, clon, r.latitude, r.longitude)   # facility point position (same geometry as positives)
        g = tile_geometry(PRESET, clat)
        rows.append({"tile_id": f"{r.hn_id}__hn", "facility_id": r.hn_id, "facility_name_original": r.name,
                     "split_group_id": f"sg_{r.hn_id}", "split": r.split, "broad_region": r.broad_region,
                     "category": r.category, "source": r.source, "source_id": r.source_id, "source_detail": r.source_detail,
                     "latitude": r.latitude, "longitude": r.longitude, "center_lat": clat, "center_lon": clon,
                     "jitter_m": round(dist, 2), "jitter_bearing_deg": round(bearing, 3),
                     "jitter_seed": f"{seed}|{r.hn_id}|{r.hn_id}__hn|jitter",
                     "facility_px_col": fp["fence_px_col"], "facility_px_row": fp["fence_px_row"],
                     "min_km_to_positive": r.min_km_to_positive, "role": "hard_negative",
                     "label_status": "candidate_hard_negative", "status": "planned", "preset": PRESET,
                     "planned_zoom": g.zoom, "planned_grid_n": g.grid_n, "planned_requests": g.requests_per_tile})
    t = pd.DataFrame(rows)
    # overlap with every existing tile (positives + nearby candidates) and among hard negatives
    old = pd.read_csv(ddir(cfg) / "tile_plan.csv")
    old = old[old.status == "planned"]
    allc = pd.concat([old[["tile_id", "center_lat", "center_lon"]], t[["tile_id", "center_lat", "center_lon"]]])
    la, lo = allc.center_lat.to_numpy(), allc.center_lon.to_numpy()
    n_old = len(old)
    cheb = chebyshev_m(la[n_old:, None], lo[n_old:, None], la[None, :], lo[None, :])
    for i in range(len(t)):
        cheb[i, n_old + i] = np.inf
    checks = {"n_points": int(len(t)), "by_category": t.category.value_counts().to_dict(),
              "by_split": t.split.value_counts().to_dict(),
              "category_x_split": pd.crosstab(t.category, t.split).to_dict(),
              "min_km_to_positive": float(t.min_km_to_positive.min()),
              "tiles_overlapping_existing_or_each_other": int((cheb.min(1) < 2000).sum()),
              "split_groups_shared_with_positives": 0,
              "planned_requests": int(t.planned_requests.sum())}
    return t, checks


# ------------------------------------------------------------------ post-extraction

def review_sheet(t: pd.DataFrame, out: Path, per_sheet: int = 16, ncol: int = 4, thumb: int = 460,
                 prefix: str = "hn_review_sheet") -> pd.DataFrame:
    """Hard-negative review: tile + a ~475 m zoom around the facility point side by side; no coordinates shown."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{prefix}_*.png"):
        old.unlink()
    try:
        big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        big = small = ImageFont.load_default()
    t = t.sort_values(["category", "split", "tile_id"]).reset_index(drop=True)
    rows, ncol_pairs = [], ncol // 2 if ncol > 2 else 1
    cell_w, cell_h = 2 * thumb + 6, thumb + 56
    per = ncol_pairs * 4
    for s, start in enumerate(range(0, len(t), per)):
        chunk = t.iloc[start:start + per]
        sheet = Image.new("RGB", (ncol_pairs * (cell_w + 12) + 8, int(np.ceil(len(chunk) / ncol_pairs)) * cell_h + 8), "white")
        dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk.itertuples()):
            x, y = 8 + (k % ncol_pairs) * (cell_w + 12), 8 + (k // ncol_pairs) * cell_h
            im = Image.open(REPO_ROOT / r.png_path).convert("RGB")
            sheet.paste(im.resize((thumb, thumb)), (x, y))
            cx, cy, half = float(r.facility_px_col), float(r.facility_px_row), 320
            z = im.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((thumb, thumb))
            sheet.paste(z, (x + thumb + 6, y))
            kf = thumb / 2560
            dr.ellipse([x + cx * kf - 9, y + cy * kf - 9, x + cx * kf + 9, y + cy * kf + 9], outline=(0, 200, 255), width=3)
            dr.rectangle([x, y, x + 50, y + 32], fill="black")
            dr.text((x + 6, y + 3), f"{k + 1:02d}", fill="yellow", font=big)
            dr.text((x, y + thumb + 4), f"{r.tile_id} | {r.category} | {r.split}"[:90], fill="black", font=small)
            flag = f" | FLAG: {r.screen_reason}" if getattr(r, "pre_extraction_screen", "") == "flag" else ""
            dr.text((x, y + thumb + 22), f"{str(r.facility_name_original)[:50]} | ~475 m zoom right{flag}"[:110],
                    fill="firebrick" if flag else "gray", font=small)
            rows.append({"tile_id": r.tile_id, "review_sheet": f"{prefix}_{s + 1:03d}.png", "sheet_cell": k + 1})
        sheet.save(out / f"{prefix}_{s + 1:03d}.png")
    return pd.DataFrame(rows)


def mixed_set(pos_n, near: pd.DataFrame, hard: pd.DataFrame, seed: str) -> pd.DataFrame:
    """C: equal numbers of nearby and confirmed hard negatives per split (nearby sampled by seeded hash)."""
    parts = []
    for sp in ("train", "val"):
        hh = hard[hard.split == sp]
        nn = near[near.split == sp].assign(o=lambda d: [h(seed, "mix", t) for t in d.tile_id]).sort_values("o")
        parts += [hh, nn.head(len(hh)).drop(columns="o")]
    return pd.concat(parts)


def diag_row(name: str, df: pd.DataFrame, note: str) -> dict:
    if (df.label_role != "positive").sum() < 10 or df[df.split == "val"].label_role.nunique() < 2:
        return {"comparison": name, "status": "not run: too few confirmed negatives", "note": note}
    d = shortcut_diagnostics(df)
    lr = d["classifier"]["logistic_regression"]
    return {"comparison": name, "status": "ok", "n_pos": d["n_pos"], "n_neg": d["n"] - d["n_pos"],
            "lr_val_auc": lr["val_auc"], "lr_grouped_cv_auc": f"{lr['grouped_cv5_auc_mean']} +/- {lr['grouped_cv5_auc_sd']}",
            "lr_val_balanced_accuracy": lr["val_balanced_accuracy"],
            "dominant_features": ", ".join(list(lr["std_coefficients"])[:3]),
            "feature_aucs": {f: v["auc"] for f, v in d["per_feature"].items() if f in CLASSIFIER_FEATURES}, "note": note}


def post(cfg, tag: str = "", interim: bool = False) -> dict:
    """tag selects the plan file (e.g. '_thermal_power_plant'); interim=True holds comparisons B/C."""
    d = ddir(cfg)
    t = pd.read_csv(d / f"hard_negative_tile_plan{tag}.csv")
    oroot = REPO_ROOT / cfg["output_root"] / PRESET
    rec = []
    for r in t.itertuples():
        td = oroot / r.facility_id / r.tile_id
        x = json.loads((td / "extraction_record.json").read_text()) if (td / "extraction_record.json").exists() else {}
        mg = x.get("manifest_geometry") or {}
        pngs = sorted((td / "tiles").glob("*.png"))
        rec.append({"tile_id": r.tile_id, "extraction_status": x.get("status", "missing"), "requests_metered": x.get("requests_metered"),
                    "source_zoom": mg.get("zoom"), "mosaic_grid": mg.get("grid_n"), "crop_px": mg.get("crop_px"),
                    "native_mpp": mg.get("native_mpp"), "footprint_width_km": mg.get("footprint_width_km"),
                    "png_path": str(pngs[0].relative_to(REPO_ROOT)) if pngs else None})
    t = t.merge(pd.DataFrame(rec), on="tile_id")
    t["footprint_area_km2"] = (t.footprint_width_km ** 2).round(4)
    t["effective_mpp"] = (t.footprint_width_km * 1000 / 2560).round(5)
    t["validation_status"] = np.where((t.extraction_status == "complete") & ((t.footprint_width_km - 1.9).abs() < 0.005)
                                      & t.png_path.notna(), "ok", "failed")
    feats = compute_features(t[t.png_path.notna()], 8)
    t = t.merge(feats, on="tile_id", how="left")
    t["decode_error"] = t["decode_error"].fillna("")
    t["artifact_flags"] = [";".join(auto_reject(r)) for _, r in t.iterrows()]
    sp = ((t.mosaic_grid * 1280 - t.crop_px) / 2 * -1 % 1280) * 2560 / t.crop_px
    t["wm_phase_px"] = sp.round(1)

    # review artifacts (statuses start unreviewed; existing decisions preserved)
    where = review_sheet(t, d / "review", prefix=f"hn{tag}_review_sheet")
    t = t.merge(where, on="tile_id", how="left")
    path = d / "hard_negative_review.csv"
    extra = [c for c in ("category_detail", "pre_extraction_screen", "screen_reason") if c in t]
    rv = t[["tile_id", "facility_id", "category", "facility_name_original", "source_id", "split", "artifact_flags",
            "review_sheet", "sheet_cell", "png_path"] + extra].copy()
    rv["manual_review_status"] = "unreviewed"
    rv["reviewer"], rv["review_notes"] = "", ""
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        keep = old[old.manual_review_status.isin(HN_STATUSES[1:])][["tile_id", "manual_review_status", "reviewer", "review_notes"]]
        rv = rv.drop(columns=["manual_review_status", "reviewer", "review_notes"]).merge(keep, on="tile_id", how="left")
        rv = rv.fillna({"manual_review_status": "unreviewed", "reviewer": "", "review_notes": ""})
        others = old[~old.tile_id.isin(rv.tile_id)]          # rows of tiles not in this plan are kept unchanged
        rv = pd.concat([others, rv.astype(str).replace("nan", "")], ignore_index=True)
    bad = set(rv.manual_review_status) - set(HN_STATUSES)
    if bad:
        raise SystemExit(f"unknown manual_review_status values {bad}; allowed {HN_STATUSES}")
    rv.to_csv(path, index=False)
    t = t.merge(rv[["tile_id", "manual_review_status"]], on="tile_id")      # only this plan's tiles
    t["label_role"] = "hard_negative"
    t["training_eligible"] = (t.manual_review_status == "confirmed_hard_negative") & (t.artifact_flags == "") & \
                             (t.validation_status == "ok")
    t.to_csv(d / f"hard_negative_metadata_private{tag}.csv", index=False)
    pub = ["tile_id", "facility_id", "facility_name_original", "category", "source_id", "split_group_id", "split",
           "broad_region", "label_role", "manual_review_status", "training_eligible", "artifact_flags", "validation_status",
           "source_zoom", "mosaic_grid", "footprint_width_km", "footprint_area_km2", "effective_mpp", "wm_phase_px",
           "requests_metered"] + [c for c in NUISANCE if c not in ("wm_phase_px", "mosaic_grid", "source_zoom")]
    t[pub].to_csv(REPORT_DIR / f"external_hard_negatives_pilot{tag}.csv", index=False)

    # diagnostics on CONFIRMED labels only (A uses the auto-cleaned nearby pool, as before)
    m = pd.read_csv(d / "external_tiles_metadata_refined.csv")
    keep_cols = ["tile_id", "split", "split_group_id", "label_role"] + NUISANCE
    A = m[(m.role == "positive") | m.nearby_negative_selected][keep_cols]
    pos = A[A.label_role == "positive"]
    near = A[A.label_role != "positive"]
    hard = t[t.training_eligible][keep_cols]
    seed = cfg["dataset"]["seed"]
    mix = mixed_set(len(pos), near, hard, seed)
    near_matched = pd.concat([near[near.split == s].assign(o=lambda x: [h(seed, "ctl", i) for i in x.tile_id])
                              .sort_values("o").head(int((mix.split == s).sum())).drop(columns="o") for s in ("train", "val")])
    held = "held: no manual review statuses set yet (B/C run only on confirmed labels)"
    diags = [diag_row("A: positives vs cleaned nearby negatives (auto-cleaned, unreviewed)", A, "all 799 selected nearby"),
             diag_row("A': positives vs nearby negatives, size-matched to C", pd.concat([pos, near_matched]), "control for C"),
             diag_row("B: positives vs confirmed hard negatives", pd.concat([pos, hard]), f"{len(hard)} confirmed")
             if not interim else {"comparison": "B", "status": held},
             diag_row("C: positives vs 1:1 nearby + confirmed hard", pd.concat([pos, mix]), "equal nearby and hard counts per split")
             if not interim else {"comparison": "C", "status": held}]
    if interim:
        diags = [x for x in diags if not x["comparison"].startswith("A'")]

    # distribution figure (candidates, descriptive only; not a label-based diagnostic)
    groups = {"positives": pos, "nearby negatives (selected)": near,
              "hard-neg candidates (unreviewed, artifact-free)": t[t.artifact_flags == ""][keep_cols]}
    feats_plot = ["brightness", "contrast", "texture_hp", "edge_density", "water_frac", "wm_phase_px", "mosaic_grid", "source_zoom"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    colors = ["#e45756", "#4c78a8", "#f58518"]
    for ax, f in zip(axes.ravel(), feats_plot):
        data = [g[f].astype(float).dropna() for g in groups.values()]
        if f in ("mosaic_grid", "source_zoom"):
            vals = sorted(set(np.concatenate([x.unique() for x in data])))
            w = 0.25
            for k, (x, c) in enumerate(zip(data, colors)):
                ax.bar(np.arange(len(vals)) + (k - 1) * w, [(x == v).mean() for v in vals], w, color=c)
            ax.set_xticks(range(len(vals)), [str(int(v)) for v in vals])
        else:
            bins = np.histogram_bin_edges(np.concatenate(data), 30)
            for x, c in zip(data, colors):
                ax.hist(x, bins=bins, density=True, histtype="step", lw=2, color=c)
        ax.set_title(f, fontsize=10)
    axes[0, 0].legend(list(groups), fontsize=8, loc="upper right")
    fig.suptitle("Nuisance statistics: positives vs nearby negatives vs hard-negative candidates (pilot, unreviewed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"hard_negative_pilot_distributions{tag}.png", dpi=85)
    plt.close(fig)
    dist = pd.DataFrame({name: g[feats_plot].astype(float).median() for name, g in groups.items()}).round(4)

    summary = {"tiles": int(len(t)), "extraction": t.extraction_status.value_counts().to_dict(),
               "validation": t.validation_status.value_counts().to_dict(),
               "requests_metered": int(t.requests_metered.fillna(0).sum()),
               "artifact_flags": t.artifact_flags.replace("", "none").value_counts().to_dict(),
               "review_status": t.manual_review_status.value_counts().to_dict(),
               "review_by_category": pd.crosstab(t.category, t.manual_review_status).to_dict(),
               "training_eligible": int(t.training_eligible.sum()),
               "diagnostics": diags, "median_by_group": dist.to_dict(),
               "category_x_split": pd.crosstab(t.category, t.split).to_dict()}
    summary["interim"] = interim
    (d / f"hard_negative_pilot_summary{tag}.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["source", "plan", "post"])
    ap.add_argument("--categories", nargs="+", default=None, help="restrict to these categories (files get a suffix)")
    ap.add_argument("--overpass-budget-min", type=float, default=None,
                    help="stop new Overpass requests after this many minutes; missing slots are reported")
    ap.add_argument("--interim", action="store_true", help="post: hold comparisons B/C (partial hard-negative set)")
    ap.add_argument("--tag", default=None, help="file suffix override, e.g. _replacement")
    args = ap.parse_args()
    cfg = load_config()
    d = ddir(cfg)
    tag = args.tag if args.tag is not None else ("_" + "_".join(args.categories) if args.categories else "")
    if args.stage == "source":
        pts, missing = source(cfg, args.categories, None if args.overpass_budget_min is None else args.overpass_budget_min * 60)
        pts.to_csv(d / f"hard_negative_points{tag}.csv", index=False)
        missing.to_csv(d / f"hard_negative_missing_slots{tag}.csv", index=False)
        if len(pts):
            print(pts.groupby(["category", "broad_region"]).size().to_string())
        print(f"{len(pts)} points -> {d / f'hard_negative_points{tag}.csv'}; missing slots: {int(missing.missing.sum()) if len(missing) else 0}")
        if len(missing):
            print(missing.to_string(index=False))
    elif args.stage == "plan":
        t, checks = plan_tiles(pd.read_csv(d / f"hard_negative_points{tag}.csv"), cfg)
        t.to_csv(d / f"hard_negative_tile_plan{tag}.csv", index=False)
        (d / f"hard_negative_plan_checks{tag}.json").write_text(json.dumps(checks, indent=1, default=str))
        print(json.dumps(checks, indent=1, default=str))
    else:
        print(json.dumps(post(cfg, tag, interim=args.interim), indent=1, default=str))


if __name__ == "__main__":
    main()
