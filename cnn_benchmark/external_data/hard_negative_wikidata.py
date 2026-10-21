"""Replacement hard-negative sourcing WITHOUT Overpass: relabel the 12 OSM sites, add Wikidata sites, build the
final pilot point list and a pre-extraction candidate table. No Static Maps calls.

    python hard_negative_wikidata.py            # writes hard_negative_points_final.csv + candidate table

Sources
  * power plants (25, already extracted): WRI GPPD v1.3.0 (unchanged; same hn_ids)
  * 12 OSM sites (sourced earlier, cached): relabelled from their own OSM tags (`category_detail` keeps the subtype)
  * Wikidata (CC0) via query.wikidata.org SPARQL, one cached query per class: items whose P31 is the class, with a
    P625 coordinate of precision <= 0.001 deg, and without P576 (dissolved) / P3999 (closure date).
Exclusions (in addition to hard_negatives.validate_points): within 2 km of a Wikidata military base, prison or
nuclear power plant, or of a GPPD nuclear plant; nuclear-like names; >= 2 km from every other accepted site.
HIFLD substations were not used: the official dataset is access-restricted on data.gov (only third-party re-hosts).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from dataset_plan import broad_region
from geo import haversine_m
from geometry import REPO_ROOT, load_config
from hard_negative_pilot import GPPD_FILE, candidate_ok
from hard_negatives import region_quotas, validate_points

SPARQL = "https://query.wikidata.org/sparql"
UA = "nuclear-monitoring-benchmark research (hard-negative sampling; one cached query per class)"
CACHE = REPO_ROOT / "external_imagery/sources/wikidata"
MAX_PREC_DEG = 0.001

# category -> Wikidata classes (verified labels 2026-10-15)
CLASSES = {
    "refinery_petrochemical": {"Q12353044": "oil refinery"},
    "industrial_facility": {"Q905286": "chemical plant", "Q2069494": "steel mill", "Q11689547": "cement plant"},
    "rail_yard_port": {"Q44782": "port", "Q519608": "classification yard"},
    "logistics_warehouse": {"Q1868068": "logistics hub", "Q1229659": "distribution center"},
    "water_treatment": {"Q15242449": "wastewater treatment plant", "Q10373565": "waterworks"},
    "hv_substation": {"Q174814": "electrical substation"},
}
EXCLUSION_CLASSES = {"Q245016": "military base", "Q40357": "prison", "Q134447": "nuclear power plant"}
# new-site targets (pilot ~60-80 overall); refinery slots only for regions OSM could not cover
TARGETS = {"rail_yard_port": 10, "logistics_warehouse": 10, "water_treatment": 5, "industrial_facility": 8,
           "hv_substation": 4}
REFINERY_REGIONS = ["russia_central_asia", "south_asia", "africa_middle_east"]
NEW_TOTAL_TARGET = 40
# Pre-extraction quality screen (assistant judgement from source metadata + general knowledge, 2026-10-15).
# drop = not an active built industrial scene, or security fencing plausible by design; flag = keep, inspect closely.
QUALITY_SCREEN = {
    "wikidata:Port Refuge": ("drop", "not a built port (Arctic bay / archaeological site)"),
    "wikidata:Rewas Port": ("drop", "proposed port, largely unbuilt"),
    "wikidata:Geneva Steel": ("drop", "closed 2001, site demolished"),
    "wikidata:Suksun Plant": ("drop", "18th-century ironworks (historic item; country = Russian Empire)"),
    "wikidata:Liquichimica Biosintesi": ("drop", "plant never operated; abandoned"),
    "wikidata:Waltham Water Works Shop": ("drop", "single historic building, not a treatment plant"),
    "wikidata:Worldport": ("drop", "airport cargo hub: airside security perimeter by design"),
    "wikidata:Chonji Lubricant Factory": ("drop", "DPRK site: security/military fencing cannot be ruled out"),
    "wikidata:Chollima Steel Complex": ("drop", "DPRK site: security/military fencing cannot be ruled out"),
    "wikidata:Henry Hub": ("flag", "pipeline interconnect, possibly a small site"),
    "wikidata:Sever Bay port": ("flag", "remote port; existence/extent unverified"),
    "wikidata:Port de Saint-Pierre": ("flag", "small harbour"),
    "wikidata:Port of Chipiona": ("flag", "small fishing/marina harbour"),
    "wikidata:El Port Nou": ("flag", "small harbour"),
    "wikidata:Chūō Shinkansen Meijō electrical substation": ("flag", "possibly under construction"),
}


def h(*p) -> str:
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()


def sparql(qid: str, with_filters: bool = True) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{qid}.json"
    q = f"""SELECT ?item ?itemLabel ?lat ?lon ?prec ?countryLabel WHERE {{
  ?item wdt:P31 wd:{qid} ; p:P625/psv:P625 [ wikibase:geoLatitude ?lat ; wikibase:geoLongitude ?lon ; wikibase:geoPrecision ?prec ] .
  {"FILTER NOT EXISTS { ?item wdt:P576 [] } FILTER NOT EXISTS { ?item wdt:P3999 [] }" if with_filters else ""}
  OPTIONAL {{ ?item wdt:P17 ?country }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,[AUTO_LANGUAGE]". }} }}"""
    if not f.exists():
        for attempt in range(4):
            r = requests.get(SPARQL, params={"query": q}, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
                             timeout=120)
            if r.status_code == 200:
                d = r.json()
                f.write_text(json.dumps({"query": q, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **d}))
                break
            time.sleep(10 * (attempt + 1))
        else:
            raise SystemExit(f"Wikidata query for {qid} failed (HTTP {r.status_code})")
        time.sleep(2)
    d = json.loads(f.read_text())
    rows = [{"qid": b["item"]["value"].rsplit("/", 1)[-1], "name": b.get("itemLabel", {}).get("value", ""),
             "latitude": float(b["lat"]["value"]), "longitude": float(b["lon"]["value"]),
             "prec": float(b["prec"]["value"]) if "prec" in b else np.nan,
             "country": b.get("countryLabel", {}).get("value", "")} for b in d["results"]["bindings"]]
    df = pd.DataFrame(rows).drop_duplicates("qid") if rows else pd.DataFrame(columns=["qid", "name", "latitude", "longitude", "prec", "country"])
    df.attrs["retrieved_utc"] = d.get("retrieved_utc", "")
    return df


def relabel_osm(p: pd.DataFrame) -> pd.DataFrame:
    """Conservative category from OSM tags + name (tags preserved in source_detail)."""
    p = p.copy()
    cat, detail = [], []
    for name, tags in zip(p.name.fillna(""), p.source_detail.fillna("")):
        t = dict(kv.split("=", 1) for kv in tags.split(";") if "=" in kv)
        if t.get("industrial") == "refinery" and "terminal" not in name.lower():
            cat.append("refinery_petrochemical"); detail.append("refinery (OSM industrial=refinery)")
        elif t.get("industrial") == "refinery" or (t.get("industrial") == "oil" and t.get("product") == "petroleum"):
            cat.append("industrial_facility"); detail.append("petroleum / tank terminal")
        elif t.get("industrial") == "chemical":
            cat.append("industrial_facility"); detail.append("chemical plant")
        elif t.get("product") == "oil":
            cat.append("industrial_facility"); detail.append("oil processing (edible oil per name), not a refinery")
        else:
            cat.append("industrial_facility"); detail.append(f"factory ({t.get('industrial', t.get('man_made', 'works'))})")
    p["category_original"] = p.category
    p["category"], p["category_detail"] = cat, detail
    return p


def exclusion_points() -> np.ndarray:
    pts = []
    for qid in EXCLUSION_CLASSES:
        df = sparql(qid, with_filters=False)
        pts += list(zip(df.latitude, df.longitude))
    g = pd.read_csv(REPO_ROOT / GPPD_FILE, low_memory=False)
    g = g[g.primary_fuel == "Nuclear"]
    pts += list(zip(g.latitude, g.longitude))
    return np.array(pts)


def main():
    cfg = load_config()
    d = REPO_ROOT / cfg["dataset"]["dataset_dir"]
    seed = cfg["dataset"]["seed"]
    pos = pd.read_csv(d / "positive_audit.csv")
    power = pd.read_csv(d / "hard_negative_points_thermal_power_plant.csv").assign(
        category_detail=lambda x: "thermal power plant (" + x.source_detail.str.split(",").str[0] + ")",
        category_original="thermal_power_plant")
    osm = relabel_osm(pd.read_csv(
        d / "hard_negative_points_refinery_petrochemical_hv_substation_rail_yard_port_logistics_warehouse_water_treatment.csv"))
    excl = exclusion_points()
    accepted = [{"latitude": a, "longitude": b} for a, b in zip(pd.concat([power, osm]).latitude, pd.concat([power, osm]).longitude)]
    new, log = [], []

    def try_take(cands: pd.DataFrame, category: str, region: str, n: int, class_label: dict) -> int:
        got = 0
        for r in cands.itertuples():
            if got == n:
                break
            why = ""
            if not (r.prec <= MAX_PREC_DEG):
                why = f"coordinate_precision_{r.prec:g}deg"
            elif not r.name or r.name == r.qid:
                why = "no_label"
            else:
                why = candidate_ok(r.latitude, r.longitude, r.name, pos, accepted, cfg)
            if not why and len(excl) and haversine_m(r.latitude, r.longitude, excl[:, 0], excl[:, 1]).min() < 2000:
                why = "within_2km_of_military/prison/nuclear (Wikidata/GPPD)"
            log.append({"category": category, "region": region, "source_id": f"wikidata:{r.qid}", "name": r.name,
                        "result": why or "accepted"})
            if why:
                continue
            accepted.append({"latitude": r.latitude, "longitude": r.longitude})
            new.append({"category": category, "broad_region": region, "name": r.name, "latitude": r.latitude,
                        "longitude": r.longitude, "source": f"Wikidata (CC0), SPARQL retrieved {cands.attrs.get('retrieved_utc', '')}",
                        "source_id": f"wikidata:{r.qid}", "source_detail": f"P31={r.cls} ({class_label[r.cls]}); {r.country}",
                        "category_detail": class_label[r.cls], "facility_area_ha": np.nan, "anchor_facility_id": "",
                        "category_original": category})
            got += 1
        return got

    pools = {}
    for cat, classes in CLASSES.items():
        frames = []
        for qid in classes:
            df = sparql(qid)
            df["cls"] = qid
            frames.append(df)
        pool = pd.concat(frames, ignore_index=True).drop_duplicates("qid")
        pool["broad_region"] = [broad_region(a, b) for a, b in zip(pool.latitude, pool.longitude)]
        pool["order"] = [h(seed, "wikidata", cat, q) for q in pool.qid]
        pool = pool.sort_values("order")
        pool.attrs["retrieved_utc"] = frames[0].attrs.get("retrieved_utc", "")
        pools[cat] = (pool, classes)

    shortfall = []
    slots = [("refinery_petrochemical", r, 1) for r in REFINERY_REGIONS]
    for cat, n in TARGETS.items():
        slots += [(cat, r, int(k)) for r, k in region_quotas(pos, n).items() if k > 0]
    for cat, region, n in slots:
        pool, classes = pools[cat]
        sub = pool[pool.broad_region == region]
        sub.attrs = pool.attrs
        got = try_take(sub, cat, region, n, classes)
        if got < n:
            shortfall.append({"category": cat, "broad_region": region, "missing": n - got})
    # top-up: fill shortfalls with industrial facilities, preferring the regions that were short
    need = NEW_TOTAL_TARGET - len(new)
    pool, classes = pools["industrial_facility"]
    for region in [s["broad_region"] for s in shortfall] + list(pos.broad_region.value_counts().index):
        if need <= 0:
            break
        sub = pool[pool.broad_region == region]
        sub = sub[~sub.qid.isin([x["source_id"].split(":")[1] for x in new])]
        sub.attrs = pool.attrs
        need -= try_take(sub, "industrial_facility", region, min(need, 2), classes)

    new = pd.DataFrame(new)
    allp = pd.concat([power, osm.drop(columns=["hn_id"]), new], ignore_index=True)
    # stable IDs: power plants keep theirs; others numbered within (final) category in a fixed source order
    allp["hn_id"] = list(power.hn_id) + [None] * (len(allp) - len(power))
    rest = allp.hn_id.isna()
    allp.loc[rest, "hn_id"] = [f"hn_{c}_{i + 1:02d}" for c, i in
                               zip(allp.loc[rest, "category"], allp.loc[rest].groupby("category").cumcount())]
    allp.loc[allp.category == "hv_substation", "category"] = "electrical_substation"   # voltage not established
    v = validate_points(allp, pos, cfg)
    key = np.where(v.source_id.str.startswith("wikidata"), "wikidata:" + v.name.fillna(""), "")
    scr = [QUALITY_SCREEN.get(k, ("keep", "")) for k in key]
    v["pre_extraction_screen"] = [a for a, _ in scr]
    v["screen_reason"] = [b for _, b in scr]
    v["exclusion_check"] = np.where(v["valid"], "pass", v["flags"])
    v["exclusion_check_basis"] = np.where(v.source_id.str.startswith("wikidata"), "Wikidata + GPPD lists",
                                          "OpenStreetMap (Overpass, cached)")
    cols = ["hn_id", "category", "category_detail", "broad_region", "name", "source", "source_id", "source_detail",
            "latitude", "longitude", "min_km_to_positive", "exclusion_check", "exclusion_check_basis",
            "pre_extraction_screen", "screen_reason"]
    v[cols].to_csv(d / "hard_negative_candidates_private.csv", index=False)          # all, incl. dropped
    v[(v.pre_extraction_screen != "drop") & v["valid"]][cols].to_csv(d / "hard_negative_points_final.csv", index=False)
    v[[c for c in cols if c not in ("latitude", "longitude")]].to_csv(
        REPO_ROOT / "cnn_benchmark/outputs/external_data/hard_negative_candidates.csv", index=False)   # no coordinates
    pd.DataFrame(log).to_csv(d / "hard_negative_sourcing_log_wikidata.csv", index=False)
    pd.DataFrame(shortfall).to_csv(d / "hard_negative_shortfall_wikidata.csv", index=False)
    lg = pd.DataFrame(log)
    fin = v[(v.pre_extraction_screen != "drop") & v["valid"]]
    print(f"candidates: {len(v)} (valid {int(v['valid'].sum())}); dropped by screen {int((v.pre_extraction_screen == 'drop').sum())}; "
          f"final {len(fin)}; flagged {int((fin.pre_extraction_screen == 'flag').sum())}")
    print(pd.crosstab(fin.category, fin.broad_region, margins=True).to_string())
    print("shortfall:", shortfall)
    print("Wikidata rejections:", lg[lg.result != "accepted"].result.str.replace(r"coordinate_precision.*", "coordinate_precision>0.001deg", regex=True).value_counts().to_dict())


if __name__ == "__main__":
    main()
