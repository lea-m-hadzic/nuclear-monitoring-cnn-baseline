"""Plan the external training set WITHOUT any API call: ingest positives, jitter, candidate negatives, splits, checks.

Input: Yanliang's coordinate file (columns exactly `Name, Latitude, Longitude`; one row per fence point).
Outputs (coordinate-bearing, written to the git-ignored dataset_dir from extraction_config.yaml):
  positive_audit.csv   one row per source row: original strings, IDs, split group, flags, neighbours, AOI distances
  tile_plan.csv        every tile to extract (positives + candidate negatives) with centre, provenance and split
  plan_summary.json    counts and check results used by the audit report

Design (all parameters in extraction_config.yaml -> dataset):
  * facility_id = ext_<slug of the source Name>: exactly one ID per source row/name; names are never merged.
    Separately, rows closer than split_group_link_km are linked (single linkage) into a `split_group_id`, the
    train/val unit, so co-located facilities/sub-sites cannot leak across splits. split_groups.csv lists members
    and the linking distances.
  * positive tile centre = fence point moved by a deterministic jitter: bearing uniform in [0, 360), distance
    area-uniform in [min_m, max_m], both from SHA-256(seed | facility_id | point_id), i.e. per source row/point
    (never per split group). Source coordinates are untouched.
  * candidate negatives: per positive, `per_positive` centres in the annulus [inner, outer] km around its fence point,
    evenly spread in bearing on the first attempt; a candidate is accepted only if it is >= exclusion_km from EVERY
    fence point in the list, does not overlap any other planned tile (axis-aligned separation >= min_center_sep_km),
    and is outside every evaluation AOI (+ buffer). All negatives are `candidate_negative` until reviewed.
  * split: split groups are assigned to train/val per broad region by a seeded hash order (val_fraction).

    python dataset_plan.py --source "../../Reviewed fence list.csv"            # writes the plan
    python dataset_plan.py --source ... --no-jitter                             # ablation plan (separate files)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from geo import chebyshev_m, destination, haversine_m, point_pixel_in_tile, unit_floats
from geometry import REPO_ROOT, load_config, tile_geometry

PRESET = "auto_mosaic_1p9km"
SOURCE_COLUMNS = ["Name", "Latitude", "Longitude"]
LABEL_SOURCE = "Yanliang: 'Reviewed fence list.csv' (received 2026-10-14); points placed on/near layered fence lines"


# ---------------------------------------------------------------- ingest

def slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:60] or "unnamed"


def broad_region(lat: float, lon: float) -> str:
    """Coarse lon/lat box (NOT a country): used for stratification and the region-vs-label check only."""
    if lon < -30:
        return "americas_north" if lat > 15 else "americas_south"
    if lon < 45:
        return "europe" if lat >= 35 else "africa_middle_east"
    if lon < 90:
        return "russia_central_asia" if lat >= 37 else ("africa_middle_east" if lon < 60 else "south_asia")
    return "east_asia" if lat >= 18 else "southeast_asia_oceania"


def load_source(path: Path) -> pd.DataFrame:
    """Read the file exactly as supplied (UTF-8 with or without BOM); keep the original coordinate strings."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if list(df.columns) != SOURCE_COLUMNS:
        raise SystemExit(f"{path}: expected columns {SOURCE_COLUMNS}, got {list(df.columns)}; not guessing a mapping")
    df = df.rename(columns={"Name": "facility_name_original", "Latitude": "latitude_original",
                            "Longitude": "longitude_original"})
    df.insert(0, "source_row", np.arange(2, len(df) + 2))       # spreadsheet row number (header = row 1)
    df["latitude"] = pd.to_numeric(df["latitude_original"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude_original"], errors="coerce")
    return df


def assign_ids(df: pd.DataFrame) -> pd.DataFrame:
    ids, seen = [], {}
    for n in df["facility_name_original"]:
        base = "ext_" + slug(n)
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    df["facility_id"] = ids
    df["id_collision"] = [seen["ext_" + slug(n)] > 1 for n in df["facility_name_original"]]
    df["point_id"] = df["facility_id"] + "__pos"
    return df


def split_groups(df: pd.DataFrame, link_km: float) -> pd.Series:
    """Single-linkage components at link_km; group id = smallest facility_id in the component (stable)."""
    lat, lon = df["latitude"].to_numpy(), df["longitude"].to_numpy()
    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(*np.where(np.triu(d < link_km * 1000, 1))):
        parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(df))]
    first = {}
    for r, fid in zip(roots, df["facility_id"]):
        first[r] = min(first.get(r, fid), fid)
    return pd.Series([first[r].replace("ext_", "sg_", 1) for r in roots], index=df.index)


def split_group_table(df: pd.DataFrame, link_km: float) -> pd.DataFrame:
    """One row per multi-member split group: member facility IDs/names and every linking pair (< link_km) with distance."""
    rows = []
    for g, sub in df.groupby("split_group_id"):
        if len(sub) < 2:
            continue
        lat, lon = sub["latitude"].to_numpy(), sub["longitude"].to_numpy()
        d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) / 1000
        ids, names = sub["facility_id"].tolist(), sub["facility_name_original"].tolist()
        links = [f"{ids[a]}~{ids[b]}:{d[a, b]:.2f}km" for a in range(len(ids)) for b in range(a + 1, len(ids))
                 if d[a, b] < link_km]
        rows.append({"split_group_id": g, "n_facilities": len(sub), "facility_ids": ";".join(ids),
                     "facility_names_original": " | ".join(names), "max_pairwise_km": round(float(d.max()), 2),
                     "linking_pairs_lt_threshold": ";".join(links),
                     "reason": f"single-linkage chain of rows < {link_km} km apart (co-located sub-sites/complex)"})
    return pd.DataFrame(rows)


def audit_positives(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dcfg = cfg["dataset"]
    df["coordinate_type"] = "fence_segment"
    df["label_source"] = LABEL_SOURCE
    df["country"] = ""                                   # not supplied in the source file
    flags = [[] for _ in range(len(df))]
    for i, r in df.iterrows():
        if not (math.isfinite(r.latitude) and math.isfinite(r.longitude)):
            flags[i].append("non_numeric_coordinate")
        elif not (-90 <= r.latitude <= 90 and -180 <= r.longitude <= 180):
            flags[i].append("out_of_range")
        dec = len(r.latitude_original.split(".")[1]) if "." in r.latitude_original else 0
        if dec < 5:
            flags[i].append(f"low_precision_{dec}_decimals")
    ok = df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
    if not ok.all():
        raise SystemExit(f"invalid coordinates in rows {df.loc[~ok, 'source_row'].tolist()}; fix the source first")
    df["broad_region"] = [broad_region(a, b) for a, b in zip(df.latitude, df.longitude)]
    df["split_group_id"] = split_groups(df, dcfg["split_group_link_km"])
    df["split_group_size"] = df.groupby("split_group_id")["facility_id"].transform("size")

    lat, lon = df["latitude"].to_numpy(), df["longitude"].to_numpy()
    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(d, np.inf)
    cheb = chebyshev_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(cheb, np.inf)
    tile_m = cfg["tile_km"] * 1000
    nn = d.argmin(1)
    df["nearest_other_km"] = np.round(d.min(1) / 1000, 3)
    df["nearest_other_facility_id"] = df["facility_id"].to_numpy()[nn]
    df["rows_within_one_tile"] = [";".join(df["facility_id"].to_numpy()[np.where(cheb[i] < tile_m)[0]])
                                  for i in range(len(df))]
    df["exact_duplicate_coordinate"] = df.duplicated(["latitude_original", "longitude_original"], keep=False)
    df["near_duplicate_lt_100m"] = d.min(1) < 100
    for name, a in cfg["evaluation_aois"].items():
        dist = haversine_m(lat, lon, a["lat"], a["lon"]) / 1000
        df[f"dist_{name.lower()}_aoi_centre_km"] = np.round(dist, 1)
        for i in np.where(dist < a["radius_km"] + cfg["aoi_exclusion_buffer_km"])[0]:
            flags[i].append(f"inside_or_near_{name}_aoi")
        for i in np.where((dist >= a["radius_km"] + cfg["aoi_exclusion_buffer_km"]) & (dist < 300))[0]:
            flags[i].append(f"same_region_as_{name}_aoi_{dist[i]:.0f}km_info")
    for i in range(len(df)):
        if df.at[i, "rows_within_one_tile"]:
            flags[i].append("another_row_within_one_tile")
        if df.at[i, "near_duplicate_lt_100m"]:
            flags[i].append("near_duplicate_lt_100m")
        if df.at[i, "id_collision"]:
            flags[i].append("facility_id_collision")
    df["flags"] = [";".join(f) for f in flags]
    return df


# ---------------------------------------------------------------- jitter

def jitter_positive(r: pd.Series, cfg: dict, enabled: bool) -> dict:
    j = cfg["dataset"]["jitter"]
    seed = cfg["dataset"]["seed"]
    if not enabled:
        return dict(center_lat=r.latitude, center_lon=r.longitude, jitter_m=0.0, jitter_bearing_deg=0.0,
                    jitter_seed="none")
    u_b, u_d = unit_floats(seed, r.facility_id, r.point_id, "jitter")
    bearing = 360.0 * u_b
    dist = math.sqrt(u_d * (j["max_m"] ** 2 - j["min_m"] ** 2) + j["min_m"] ** 2)   # area-uniform in the annulus
    lat, lon = destination(r.latitude, r.longitude, bearing, dist)
    return dict(center_lat=lat, center_lon=lon, jitter_m=round(dist, 2), jitter_bearing_deg=round(bearing, 3),
                jitter_seed=f"{seed}|{r.facility_id}|{r.point_id}|jitter")


def fence_pixel(center_lat, center_lon, lat, lon) -> dict:
    """Where the fence point lands in the 2560 px output tile, using the exact auto_mosaic zoom/crop for this centre."""
    cfg = load_config()
    g = tile_geometry(PRESET, center_lat)
    crop_px = int(round(cfg["tile_km"] * 1000.0 / g.native_mpp))
    col, row = point_pixel_in_tile(center_lat, center_lon, lat, lon, g.zoom, 2, crop_px, g.output_px)
    margin_px = min(col, row, g.output_px - col, g.output_px - row)
    return dict(zoom=g.zoom, grid_n=g.grid_n, crop_px=crop_px, fence_px_col=round(col, 1), fence_px_row=round(row, 1),
                fence_offset_px=round(math.hypot(col - g.output_px / 2, row - g.output_px / 2), 1),
                fence_edge_margin_px=round(margin_px, 1), fence_edge_margin_m=round(margin_px * g.effective_mpp, 1))


# ---------------------------------------------------------------- negatives

def plan_negatives(pos: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    n = cfg["dataset"]["negatives"]
    seed = cfg["dataset"]["seed"]
    sep = n["min_center_sep_km"] * 1000
    aois = cfg["evaluation_aois"]
    buf = cfg["aoi_exclusion_buffer_km"]
    fence_lat, fence_lon = pos["latitude"].to_numpy(), pos["longitude"].to_numpy()
    tiles_lat, tiles_lon = list(pos["center_lat"]), list(pos["center_lon"])     # positives first
    rows = []
    for _, p in pos.sort_values("point_id").iterrows():
        u0 = unit_floats(seed, p.facility_id, "neg_rotation", n=1)[0]
        for k in range(n["per_positive"]):
            accepted, reasons = None, {}
            for attempt in range(n["max_attempts"]):
                ub, ur = unit_floats(seed, p.facility_id, "neg", k, attempt)
                if attempt == 0:   # evenly spread bearings, small seeded wobble
                    bearing = (360 * u0 + k * 360 / n["per_positive"] + (ub - 0.5) * 30) % 360
                else:
                    bearing = 360 * ub
                r_m = 1000 * (n["annulus_inner_km"] + ur * (n["annulus_outer_km"] - n["annulus_inner_km"]))
                lat, lon = destination(p.latitude, p.longitude, bearing, r_m)
                d_fence = haversine_m(lat, lon, fence_lat, fence_lon)
                if d_fence.min() < n["exclusion_km"] * 1000:
                    reasons["near_a_fence_point"] = reasons.get("near_a_fence_point", 0) + 1
                    continue
                if chebyshev_m(lat, lon, np.array(tiles_lat), np.array(tiles_lon)).min() < sep:
                    reasons["overlaps_planned_tile"] = reasons.get("overlaps_planned_tile", 0) + 1
                    continue
                if any(haversine_m(lat, lon, a["lat"], a["lon"]) < (a["radius_km"] + buf) * 1000 for a in aois.values()):
                    reasons["evaluation_aoi"] = reasons.get("evaluation_aoi", 0) + 1
                    continue
                accepted = (lat, lon, bearing, r_m, attempt, float(d_fence.min()))
                break
            tile_id = f"{p.facility_id}__neg{k + 1}"
            subset = "primary" if k < n["primary_per_positive"] else "reserve"
            if accepted is None:
                rows.append(dict(tile_id=tile_id, parent_point_id=p.point_id, status="not_placed",
                                 candidate_subset=subset, rejections=json.dumps(reasons)))
                continue
            lat, lon, bearing, r_m, attempt, dmin = accepted
            tiles_lat.append(lat); tiles_lon.append(lon)
            rows.append(dict(tile_id=tile_id, parent_point_id=p.point_id, status="planned", candidate_subset=subset,
                             center_lat=lat, center_lon=lon, neg_bearing_deg=round(bearing, 3),
                             neg_dist_from_parent_fence_km=round(r_m / 1000, 3),
                             neg_min_dist_any_fence_km=round(dmin / 1000, 3), neg_attempt=attempt,
                             neg_seed=f"{seed}|{p.facility_id}|neg|{k}|{attempt}", rejections=json.dumps(reasons)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- split

def assign_split(pos: pd.DataFrame, cfg: dict) -> dict[str, str]:
    frac, seed = cfg["dataset"]["split"]["val_fraction"], cfg["dataset"]["seed"]
    groups = pos.groupby("split_group_id")["broad_region"].agg(lambda s: s.mode().iloc[0])
    split = {}
    for region, gs in groups.groupby(groups):
        order = sorted(gs.index, key=lambda g: hashlib.sha256(f"{seed}|split|{g}".encode()).hexdigest())
        n_val = int(round(frac * len(order))) if len(order) >= 2 else 0
        for i, g in enumerate(order):
            split[g] = "val" if i < n_val else "train"
    return split


# ---------------------------------------------------------------- checks

def pre_extraction_checks(tiles: pd.DataFrame, cfg: dict) -> dict:
    t = tiles[tiles["status"] == "planned"].reset_index(drop=True)
    lat, lon = t["center_lat"].to_numpy(), t["center_lon"].to_numpy()
    cheb = chebyshev_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(cheb, np.inf)
    tile_m = cfg["tile_km"] * 1000
    ov = np.triu(cheb < tile_m, 1)
    i, j = np.where(ov)
    pairs = pd.DataFrame({"a": t.tile_id[i].values, "b": t.tile_id[j].values, "role_a": t.role[i].values,
                          "role_b": t.role[j].values, "split_a": t.split[i].values, "split_b": t.split[j].values,
                          "split_group_a": t.split_group_id[i].values, "split_group_b": t.split_group_id[j].values})
    aoi_min = {name: float(np.round(haversine_m(lat, lon, a["lat"], a["lon"]).min() / 1000 - a["radius_km"], 1))
               for name, a in cfg["evaluation_aois"].items()}
    return {
        "overlapping_tile_pairs": int(len(pairs)),
        "overlapping_pairs_by_roles": pairs.groupby(["role_a", "role_b"]).size().rename("n").reset_index()
                                           .to_dict("records") if len(pairs) else [],
        "overlaps_across_splits": int((pairs.split_a != pairs.split_b).sum()) if len(pairs) else 0,
        "overlaps_across_split_groups": int((pairs.split_group_a != pairs.split_group_b).sum()) if len(pairs) else 0,
        "negative_overlapping_any_tile": int(((pairs.role_a == "candidate_negative") |
                                              (pairs.role_b == "candidate_negative")).sum()) if len(pairs) else 0,
        "min_km_outside_evaluation_aoi_edge": aoi_min,
        "groups_in_both_splits": int((t.groupby("split_group_id")["split"].nunique() > 1).sum()),
        "facilities_in_both_splits": int((t.groupby("facility_id")["split"].nunique() > 1).sum()),
    }, pairs


# ---------------------------------------------------------------- main

def build_plan(source: Path, jitter: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    cfg = load_config()
    pos = audit_positives(assign_ids(load_source(source)), cfg)
    jit = pd.DataFrame([jitter_positive(r, cfg, jitter) for _, r in pos.iterrows()], index=pos.index)
    pos = pd.concat([pos, jit], axis=1)
    fp = pd.DataFrame([fence_pixel(r.center_lat, r.center_lon, r.latitude, r.longitude) for _, r in pos.iterrows()],
                      index=pos.index)
    pos = pd.concat([pos, fp], axis=1)
    pos["jitter_ok"] = pos["fence_edge_margin_m"] >= cfg["dataset"]["jitter"]["min_edge_margin_m"]

    split = assign_split(pos, cfg)
    pos["split"] = pos["split_group_id"].map(split)
    neg = plan_negatives(pos, cfg)
    meta_cols = ["point_id", "facility_id", "facility_name_original", "split_group_id", "broad_region", "split",
                 "latitude_original", "longitude_original", "latitude", "longitude"]
    neg = neg.merge(pos[meta_cols].rename(columns={"point_id": "parent_point_id"}), on="parent_point_id", how="left")
    neg["role"] = "candidate_negative"
    neg["label_status"] = "candidate_negative"

    ptiles = pos[meta_cols + ["center_lat", "center_lon", "jitter_m", "jitter_bearing_deg", "jitter_seed",
                              "fence_px_col", "fence_px_row", "fence_offset_px", "fence_edge_margin_m"]].copy()
    ptiles = ptiles.rename(columns={"point_id": "tile_id"})
    ptiles["parent_point_id"] = ptiles["tile_id"]
    ptiles["role"] = "positive"
    ptiles["label_status"] = "positive_source_label"
    ptiles["candidate_subset"] = "positive"
    ptiles["status"] = "planned"
    tiles = pd.concat([ptiles, neg], ignore_index=True)
    tiles["preset"] = PRESET
    tiles["jitter_enabled"] = jitter
    geo = pd.DataFrame([tile_geometry(PRESET, la).__dict__ if np.isfinite(la) else {}
                        for la in tiles["center_lat"].fillna(np.nan)])
    tiles["planned_zoom"], tiles["planned_grid_n"] = geo["zoom"], geo["grid_n"]
    tiles["planned_requests"] = geo["requests_per_tile"]
    checks, pairs = pre_extraction_checks(tiles, cfg)
    groups = split_group_table(pos, cfg["dataset"]["split_group_link_km"])
    multi = pos[pos["split_group_size"] > 1]
    jit_unique = {"points_in_multi_row_groups": int(len(multi)),
                  "distinct_jitter_vectors_among_them": int(multi[["jitter_m", "jitter_bearing_deg"]]
                                                             .drop_duplicates().shape[0]) if jitter else 0,
                  "distinct_seeds": int(pos["jitter_seed"].nunique()) if jitter else 0}
    summary = {
        "source_rows": int(len(pos)), "facilities": int(pos.facility_id.nunique()),
        "facility_id_one_to_one_with_rows": bool(pos.facility_id.is_unique and len(pos) == pos.facility_name_original.nunique()),
        "split_group_link_km": cfg["dataset"]["split_group_link_km"],
        "jitter_per_point": jit_unique,
        "split_groups": int(pos.split_group_id.nunique()),
        "multi_row_split_groups": int((pos.groupby("split_group_id").size() > 1).sum()),
        "jitter": {"enabled": jitter, **cfg["dataset"]["jitter"],
                   "all_ok": bool(pos.jitter_ok.all()), "n_fail": int((~pos.jitter_ok).sum()),
                   "edge_margin_m_min": float(pos.fence_edge_margin_m.min()),
                   "offset_m_quantiles": pos.jitter_m.quantile([0, .25, .5, .75, 1]).round(1).tolist()},
        "negatives": {**cfg["dataset"]["negatives"], "planned": int((neg.status == "planned").sum()),
                      "not_placed": int((neg.status != "planned").sum()),
                      "primary": int(((neg.status == "planned") & (neg.candidate_subset == "primary")).sum()),
                      "reserve": int(((neg.status == "planned") & (neg.candidate_subset == "reserve")).sum())},
        "checks": checks,
    }
    return pos, tiles, summary, pairs, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--no-jitter", action="store_true", help="ablation: positive tiles centred on the fence point")
    args = ap.parse_args()
    cfg = load_config()
    out = REPO_ROOT / cfg["dataset"]["dataset_dir"]
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_nojitter" if args.no_jitter else ""
    pos, tiles, summary, pairs, groups = build_plan(args.source, jitter=not args.no_jitter)
    groups.to_csv(out / f"split_groups{suffix}.csv", index=False)
    pos.to_csv(out / f"positive_audit{suffix}.csv", index=False)
    tiles.to_csv(out / f"tile_plan{suffix}.csv", index=False)
    pairs.to_csv(out / f"overlapping_tile_pairs{suffix}.csv", index=False)
    (out / f"plan_summary{suffix}.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
