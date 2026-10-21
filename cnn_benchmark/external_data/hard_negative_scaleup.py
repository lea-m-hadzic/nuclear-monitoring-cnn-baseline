"""Targeted hard-negative scale-up (~130 new candidates toward ~150 confirmed). No Overpass, no Static Maps calls.

    python hard_negative_scaleup.py source   # points + candidate table + quality proxies (pre-extraction)
    python hard_negative_scaleup.py plan     # tile plan + overlap checks against EVERY existing tile

Targets weight categories by the pilot's confirmation rates (refinery 9/9, industrial 10/11, power 22/25, water 4/4,
logistics 6/7; substations 1/4 and ports 4/8 only sparingly). Region quotas follow the positives' broad regions.
Sources: WRI GPPD v1.3.0 (power), Wikidata SPARQL (others), cached under external_imagery/sources/.
Automatic quality rules (replace the pilot's manual screen): Wikidata state of use (P5817) closed/demolished/
abandoned/proposed/under construction, historical-state countries, North Korea, airport/military/prison names.
Exclusions as in the pilot: >= 20 km from positives, outside AOIs + buffer, nuclear-like names, no Wikidata
military base / prison / nuclear plant or GPPD nuclear plant within 2 km, and >= 3.5 km from every hard-negative
site already planned (pilot incl. rejected/screened ones), so jittered 1.9 km tiles cannot overlap.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
import requests

from dataset_plan import broad_region
from geo import chebyshev_m, haversine_m
from geometry import REPO_ROOT, load_config, tiered_cost_usd
from hard_negative_pilot import GPPD_FILE, GPPD_VERSION, candidate_ok, plan_tiles
from hard_negative_wikidata import CACHE, MAX_PREC_DEG, SPARQL, UA, exclusion_points
from hard_negatives import region_quotas

# Revised 2026-10-15 after the first sourcing pass (paper mills / waterworks were mostly heritage records)
TARGETS = {"refinery_petrochemical": 45, "thermal_power_plant": 40, "industrial_facility": 20,
           "water_treatment": 12, "logistics_warehouse": 10, "electrical_substation": 2, "rail_yard_port": 2}
CLASSES = {
    "refinery_petrochemical": {"Q12353044": "oil refinery"},
    "industrial_facility": {"Q905286": "chemical plant", "Q2069494": "steel mill", "Q11689547": "cement plant",
                            "Q44396585": "aluminium smelter"},            # paper mill (Q918088) dropped
    "water_treatment": {"Q15242449": "wastewater treatment plant"},     # generic waterworks (Q10373565) dropped
    "logistics_warehouse": {"Q1868068": "logistics hub", "Q1229659": "distribution center"},
    "electrical_substation": {"Q174814": "electrical substation"},
    "rail_yard_port": {"Q44782": "port", "Q519608": "classification yard"},
}
MIN_KM_BETWEEN_HN = 3.5
BAD_STATE = re.compile(r"closed|demolish|abandon|disused|ruin|destroy|under construction|planned|proposed|"
                       r"decommission|former|defunct|unbuilt|inactive|out of service", re.I)
HISTORIC_NAME = re.compile(r"\bformer\b|\bold\b|historic|ruin|demolish|museum|heritage|\bsite of\b|mill site|"
                           r"\bhouses?\b|\bweir\b|\bdam\b", re.I)
AIRPORT_CLASSES = ["Q1248784", "Q644371", "Q94993988"]     # airport, international airport, commercial airport
AIRPORT_KM = 3.0
# region roll-over order for unfilled quota: avoid adding to North America unless nothing else is left
ROLLOVER = ["europe", "east_asia", "russia_central_asia", "south_asia", "africa_middle_east", "americas_south",
            "southeast_asia_oceania", "americas_north"]
HISTORIC_COUNTRY = re.compile(r"Empire|Soviet|Yugoslavia|Czechoslovakia|German Democratic|Ottoman|Kingdom of Prussia|"
                              r"Austria-Hungary|Qing|Manchukuo", re.I)
BAD_NAME = re.compile(r"airport|air base|airbase|military|naval|prison|penitentiary|barracks|\bRAF\b|superterminal", re.I)
DPRK = re.compile(r"North Korea|Democratic People's Republic of Korea", re.I)


def h(*p) -> str:
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()


def sparql_v2(qid: str) -> pd.DataFrame:
    """Like the pilot query, plus state of use (P5817), heritage designation (P1435) count and earliest inception (P571)."""
    f = CACHE / f"{qid}_v3.json"
    q = f"""SELECT ?item ?itemLabel ?lat ?lon ?prec ?countryLabel (GROUP_CONCAT(DISTINCT ?useLabel; separator="|") AS ?use)
  (COUNT(DISTINCT ?her) AS ?heritage) (MIN(?inc) AS ?inception) WHERE {{
  ?item wdt:P31 wd:{qid} ; p:P625/psv:P625 [ wikibase:geoLatitude ?lat ; wikibase:geoLongitude ?lon ; wikibase:geoPrecision ?prec ] .
  FILTER NOT EXISTS {{ ?item wdt:P576 [] }} FILTER NOT EXISTS {{ ?item wdt:P3999 [] }}
  OPTIONAL {{ ?item wdt:P17 ?country }}
  OPTIONAL {{ ?item wdt:P5817 ?u . ?u rdfs:label ?useLabel FILTER(lang(?useLabel)="en") }}
  OPTIONAL {{ ?item wdt:P1435 ?her }} OPTIONAL {{ ?item wdt:P571 ?inc }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,[AUTO_LANGUAGE]". ?item rdfs:label ?itemLabel . ?country rdfs:label ?countryLabel }}
}} GROUP BY ?item ?itemLabel ?lat ?lon ?prec ?countryLabel"""
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            r = requests.get(SPARQL, params={"query": q}, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
                             timeout=180)
            if r.status_code == 200:
                f.write_text(json.dumps({"query": q, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **r.json()}))
                break
            time.sleep(15 * (attempt + 1))
        else:
            raise SystemExit(f"Wikidata query for {qid} failed (HTTP {r.status_code})")
        time.sleep(3)
    d = json.loads(f.read_text())
    rows = [{"qid": b["item"]["value"].rsplit("/", 1)[-1], "name": b.get("itemLabel", {}).get("value", ""),
             "latitude": float(b["lat"]["value"]), "longitude": float(b["lon"]["value"]),
             "prec": float(b["prec"]["value"]) if "prec" in b else np.nan,
             "country": b.get("countryLabel", {}).get("value", ""), "state_of_use": b.get("use", {}).get("value", ""),
             "heritage": int(b.get("heritage", {}).get("value", 0)),
             "inception": pd.to_numeric(b.get("inception", {}).get("value", "")[:5].lstrip("+").split("-")[0] or None, errors="coerce")}
            for b in d["results"]["bindings"]]
    df = pd.DataFrame(rows).drop_duplicates("qid")
    df.attrs["retrieved_utc"] = d.get("retrieved_utc", "")
    return df


def airports() -> np.ndarray:
    f = CACHE / "airports.json"
    vals = " ".join(f"wd:{q}" for q in AIRPORT_CLASSES)
    q = f"""SELECT DISTINCT ?item ?lat ?lon WHERE {{ VALUES ?c {{ {vals} }} ?item wdt:P31 ?c ;
      p:P625/psv:P625 [ wikibase:geoLatitude ?lat ; wikibase:geoLongitude ?lon ] . }}"""
    if not f.exists():
        r = requests.get(SPARQL, params={"query": q}, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"}, timeout=240)
        r.raise_for_status()
        f.write_text(json.dumps({"query": q, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **r.json()}))
    b = json.loads(f.read_text())["results"]["bindings"]
    return np.array([(float(x["lat"]["value"]), float(x["lon"]["value"])) for x in b])


def quality_reason(name: str, country: str, use: str, heritage: int = 0, inception: float = np.nan) -> str:
    if heritage and heritage > 0:
        return "heritage_designation"
    if inception == inception and inception < 1900:
        return "founded_before_1900"
    if use and BAD_STATE.search(use):
        return f"state_of_use:{use}"
    if HISTORIC_NAME.search(name or ""):
        return "historic_wording_in_name"
    if HISTORIC_COUNTRY.search(country or ""):
        return f"historic_country:{country}"
    if DPRK.search(country or ""):
        return "dprk_site"
    if BAD_NAME.search(name or ""):
        return "security_by_design_name"
    return ""


def source(cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = REPO_ROOT / cfg["dataset"]["dataset_dir"]
    seed = cfg["dataset"]["seed"]
    c2 = copy.deepcopy(cfg)
    c2["hard_negatives"]["min_km_between_hard_negatives"] = MIN_KM_BETWEEN_HN
    pos = pd.read_csv(d / "positive_audit.csv")
    pilot = pd.read_csv(d / "hard_negative_candidates_private.csv")                 # all 77 incl. screened-out
    pilot_ids = set(pilot.source_id)
    accepted = [{"latitude": a, "longitude": b} for a, b in zip(pilot.latitude, pilot.longitude)]
    excl = exclusion_points()
    apt = airports()
    log, new = [], []

    def accept(cat, region, name, lat, lon, src, sid, detail, cdet, prec=np.nan):
        accepted.append({"latitude": lat, "longitude": lon})
        new.append({"category": cat, "broad_region": region, "name": name, "latitude": lat, "longitude": lon,
                    "source": src, "source_id": sid, "source_detail": detail, "category_detail": cdet,
                    "coord_precision_deg": prec, "facility_area_ha": np.nan, "anchor_facility_id": ""})

    def checks(lat, lon, name, sid):
        if sid in pilot_ids:
            return "duplicate_of_pilot_source_id"
        why = candidate_ok(lat, lon, name, pos, accepted, c2)
        if why == "co_located_with_accepted":
            return "within_3.5km_of_existing_or_new_hard_negative"
        if not why and len(excl) and haversine_m(lat, lon, excl[:, 0], excl[:, 1]).min() < 2000:
            return "within_2km_of_military/prison/nuclear (Wikidata/GPPD)"
        if not why and len(apt) and haversine_m(lat, lon, apt[:, 0], apt[:, 1]).min() < AIRPORT_KM * 1000:
            return "within_3km_of_airport"
        return why

    # --- power plants (GPPD)
    g = pd.read_csv(REPO_ROOT / GPPD_FILE, low_memory=False)
    g = g[g.primary_fuel.isin(["Coal", "Gas", "Oil"]) & (g.capacity_mw >= 500) & ~g.other_fuel1.fillna("").eq("Nuclear")].copy()
    g["broad_region"] = [broad_region(a, b) for a, b in zip(g.latitude, g.longitude)]
    g["order"] = [h(seed, "scaleup_gppd", i) for i in g.gppd_idnr]
    pools = {"thermal_power_plant": g.sort_values("order")}

    # --- Wikidata categories
    stats = {}
    for cat, classes in CLASSES.items():
        frames = []
        for qid, label in classes.items():
            df = sparql_v2(qid)
            df["cls"], df["cls_label"] = qid, label
            frames.append(df)
        pool = pd.concat(frames, ignore_index=True).drop_duplicates("qid")
        pool["broad_region"] = [broad_region(a, b) for a, b in zip(pool.latitude, pool.longitude)]
        pool["order"] = [h(seed, "scaleup_wd", cat, q) for q in pool.qid]
        pool["quality"] = [quality_reason(n, c, u, hh, inc) for n, c, u, hh, inc in
                           zip(pool.name, pool.country, pool.state_of_use, pool.heritage, pool.inception)]
        stats[cat] = {"pool": int(len(pool)), "precise": int((pool.prec <= MAX_PREC_DEG).sum()),
                      "quality_rule_hits": int((pool.quality != "").sum())}
        pools[cat] = pool.sort_values("order")

    def take(cat, region, n) -> int:
        got = 0
        pool = pools[cat]
        sub = pool[pool.broad_region == region] if region else pool
        for r in sub.itertuples():
            if got == n:
                break
            if cat == "thermal_power_plant":
                sid, name = f"gppd:{r.gppd_idnr}", r.name
                why = checks(r.latitude, r.longitude, name, sid)
            else:
                sid, name = f"wikidata:{r.qid}", r.name
                if sid in {x["source_id"] for x in new}:
                    continue
                why = ("coordinate_precision>0.001deg" if not r.prec <= MAX_PREC_DEG else
                       "no_label" if (not name or name == r.qid) else r.quality or checks(r.latitude, r.longitude, name, sid))
            log.append({"category": cat, "region": r.broad_region, "source_id": sid, "name": name, "result": why or "accepted"})
            if why:
                continue
            if cat == "thermal_power_plant":
                accept(cat, r.broad_region, name, r.latitude, r.longitude, GPPD_VERSION, sid,
                       f"{r.primary_fuel}, {r.capacity_mw:.0f} MW, {r.country}; geolocation: {r.geolocation_source}",
                       f"thermal power plant ({r.primary_fuel})")
            else:
                accept(cat, r.broad_region, name, r.latitude, r.longitude,
                       f"Wikidata (CC0), SPARQL retrieved {pool.attrs.get('retrieved_utc', '')}", sid,
                       f"P31={r.cls} ({r.cls_label}); {r.country}; state of use: {r.state_of_use or 'n/a'}",
                       r.cls_label, r.prec)
            got += 1
        return got

    shortfall, regional_short = [], []
    for cat, n in TARGETS.items():
        q = region_quotas(pos, n)
        missing = 0
        for region, k in q.items():
            if k:
                got = take(cat, region, int(k))
                if got < k:
                    regional_short.append({"category": cat, "broad_region": region, "quota": int(k), "filled": got})
                missing += k - got
        # roll over one slot at a time to the region furthest below its positives share (counted over all new sites);
        # North America only when no other region can supply a candidate
        share = pos.broad_region.value_counts(normalize=True)
        exhausted = set()
        while missing > 0:
            counts = pd.Series([x["broad_region"] for x in new]).value_counts()
            total = max(len(new), 1)
            cands = [r for r in ROLLOVER if r not in exhausted and r != "americas_north"] or \
                    [r for r in ["americas_north"] if r not in exhausted]
            if not cands:
                break
            region = max(cands, key=lambda r: share.get(r, 0) * (total + 1) - counts.get(r, 0))
            if take(cat, region, 1):
                missing -= 1
            else:
                exhausted.add(region)
        if missing > 0:
            shortfall.append({"category": cat, "missing": int(missing)})

    pts = pd.DataFrame(new)
    existing = pd.read_csv(d / "hard_negative_tile_plan_final.csv").facility_id
    nxt = {}
    for c in pts.category.unique():
        nums = [int(x.rsplit("_", 1)[1]) for x in existing if x.startswith(f"hn_{c}_")]
        nxt[c] = max(nums, default=0)
    ids = []
    for c in pts.category:
        nxt[c] += 1
        ids.append(f"hn_{c}_{nxt[c]:02d}")
    pts.insert(0, "hn_id", ids)
    lg = pd.DataFrame(log)
    return pts, lg, {"pool_stats": stats, "shortfall": shortfall,
                     "regional_shortfall": regional_short}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["source", "plan"])
    cfg = load_config()
    d = REPO_ROOT / cfg["dataset"]["dataset_dir"]
    stage = ap.parse_args().stage
    if stage == "source":
        pts, lg, info = source(cfg)
        pts.to_csv(d / "hard_negative_points_scaleup.csv", index=False)
        lg.to_csv(d / "hard_negative_sourcing_log_scaleup.csv", index=False)
        pub = pts.drop(columns=["latitude", "longitude"])
        pub.to_csv(REPO_ROOT / "cnn_benchmark/outputs/external_data/hard_negative_candidates_scaleup.csv", index=False)
        (d / "hard_negative_scaleup_info.json").write_text(json.dumps(info, indent=1, default=str))
        print(f"new candidates: {len(pts)}")
        print(pd.crosstab(pts.category, pts.broad_region, margins=True).to_string())
        print("shortfall:", info["shortfall"])
        print("pool stats:", json.dumps(info["pool_stats"]))
        print("rejections:", lg[lg.result != "accepted"].result.str.replace(r"state_of_use:.*", "state_of_use(closed/planned/...)", regex=True)
              .str.replace(r"historic_country:.*", "historic_country", regex=True).value_counts().to_dict())
    else:
        pts = pd.read_csv(d / "hard_negative_points_scaleup.csv")
        t, checks = plan_tiles(pts, cfg)
        # overlap against EVERY existing tile: positives + nearby candidates (checked in plan_tiles) AND pilot hard negatives
        pil = pd.read_csv(d / "hard_negative_tile_plan_final.csv")
        ch = chebyshev_m(t.center_lat.to_numpy()[:, None], t.center_lon.to_numpy()[:, None],
                         pil.center_lat.to_numpy()[None, :], pil.center_lon.to_numpy()[None, :])
        checks["overlap_with_pilot_hard_negative_tiles"] = int((ch < 2000).sum())
        checks["duplicate_source_ids_vs_pilot"] = int(t.source_id.isin(pd.read_csv(d / "hard_negative_candidates_private.csv").source_id).sum())
        t = t.merge(pts[["hn_id", "category_detail", "coord_precision_deg"]], left_on="facility_id", right_on="hn_id").drop(columns="hn_id")
        t.to_csv(d / "hard_negative_tile_plan_scaleup.csv", index=False)
        (d / "hard_negative_plan_checks_scaleup.json").write_text(json.dumps(checks, indent=1, default=str))
        n_req = int(t.planned_requests.sum())
        print(json.dumps({k: checks[k] for k in ("n_points", "by_split", "min_km_to_positive",
                                                 "tiles_overlapping_existing_or_each_other",
                                                 "overlap_with_pilot_hard_negative_tiles", "duplicate_source_ids_vs_pilot")}))
        print(f"planned requests {n_req}, approx ${tiered_cost_usd(n_req, cfg['pricing'], already_used_this_month=22298):.2f}")


if __name__ == "__main__":
    main()
