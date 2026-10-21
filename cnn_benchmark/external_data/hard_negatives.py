"""Hard (industrial) negative pool: STRATEGY + plan + validation of a future coordinate list. No API calls,
no downloads, no extraction (bulk extraction waits for approval).

    python hard_negatives.py                       # writes outputs/external_data/external_hard_negatives_plan.csv
    python hard_negatives.py --validate points.csv # checks a candidate list (columns: hn_id, name, category,
                                                   # latitude, longitude, source, source_id) before any extraction

Design (extraction_config.yaml -> hard_negatives):
  * categories chosen to be visually close to the positives (large built-up, fenced industrial/energy sites) but
    without the layered-fence target signature; prisons/military/nuclear sites are excluded by tag and name;
  * facility counts per category are split across broad regions in proportion to the positives, so region and
    latitude (hence zoom/grid/watermark layout) do not predict "hard negative";
  * each facility gets its own split group (>= 20 km from every positive, > the 15 km link) and is assigned to
    train/val with the same seeded per-region procedure, stratified by category so every category is in both;
  * label = `candidate_hard_negative` until manual review (same statuses as nearby negatives); a tile showing a
    recognizable layered fence is rejected (`reject_visible_layered_fence`), not relabelled positive.
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd

from dataset_audit import REPORT_DIR
from dataset_plan import broad_region
from geo import haversine_m
from geometry import REPO_ROOT, load_config, tiered_cost_usd, tile_geometry

PRESET = "auto_mosaic_1p9km"
NUCLEAR_NAME_RE = r"nuclear|atom|npp|reactor|uranium|enrichment|reprocess|radioactive|spent.fuel"


def region_quotas(pos: pd.DataFrame, n_total: int) -> pd.Series:
    share = pos.broad_region.value_counts(normalize=True)
    q = np.floor(share * n_total).astype(int)
    for r in (share * n_total - q).sort_values(ascending=False).index[: n_total - q.sum()]:   # largest remainders
        q[r] += 1
    return q


def build_plan(pos: pd.DataFrame, meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h = cfg["hard_negatives"]
    rows = []
    req_by_region = meta[meta.role == "positive"].groupby("broad_region")["requests_metered"].mean()
    for cat, spec in h["categories"].items():
        q = region_quotas(pos, spec["n"])
        for region, n in q.items():
            if n == 0:
                continue
            rq = float(req_by_region.get(region, meta.requests_metered.mean()))
            rows.append({"category": cat, "broad_region": region, "n_facilities": int(n),
                         "tiles_per_facility": h["tiles_per_facility"], "n_tiles": int(n * h["tiles_per_facility"]),
                         "expected_requests": round(n * h["tiles_per_facility"] * rq), "source": spec["source"],
                         "selection_rule": "seeded random sample within category x region after exclusions; "
                                           f">= {h['min_km_from_any_positive']} km from every positive; outside "
                                           "evaluation AOIs + buffer; no excluded tag within "
                                           f"{h['exclude_tags_within_km']} km; manual check of the centroid",
                         "label_status_initial": "candidate_hard_negative"})
    return pd.DataFrame(rows)


def validate_points(pts: pd.DataFrame, pos: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Flags for a proposed hard-negative coordinate list (nothing is fetched)."""
    h = cfg["hard_negatives"]
    p = pts.copy()
    flags = [[] for _ in range(len(p))]
    lat, lon = p.latitude.to_numpy(float), p.longitude.to_numpy(float)
    d_pos = haversine_m(lat[:, None], lon[:, None], pos.latitude.to_numpy()[None, :], pos.longitude.to_numpy()[None, :]) / 1000
    p["min_km_to_positive"] = d_pos.min(1).round(2)
    p["nearest_positive_facility_id"] = pos.facility_id.to_numpy()[d_pos.argmin(1)]
    for i in np.where(d_pos.min(1) < h["min_km_from_any_positive"])[0]:
        flags[i].append("too_close_to_positive")
    for name, a in cfg["evaluation_aois"].items():
        d = haversine_m(lat, lon, a["lat"], a["lon"]) / 1000
        for i in np.where(d < a["radius_km"] + cfg["aoi_exclusion_buffer_km"])[0]:
            flags[i].append(f"inside_or_near_{name}_aoi")
    for i in np.where(p.name.fillna("").str.contains(NUCLEAR_NAME_RE, case=False, regex=True))[0]:
        flags[i].append("nuclear_like_name")
    if len(p) > 1:
        dd = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) / 1000
        np.fill_diagonal(dd, np.inf)
        for i in np.where(dd.min(1) < h["min_km_between_hard_negatives"])[0]:
            flags[i].append("overlaps_another_hard_negative")
    for i in np.where(p.duplicated("hn_id", keep=False))[0]:
        flags[i].append("duplicate_id")
    p["broad_region"] = [broad_region(a, b) for a, b in zip(lat, lon)]
    p["flags"] = [";".join(f) for f in flags]
    p["valid"] = p["flags"] == ""
    return p


def assign_hard_negative_split(p: pd.DataFrame, cfg: dict) -> pd.Series:
    """Each valid site is its own split group; val per (category, region) stratum by seeded hash order."""
    frac, seed = cfg["dataset"]["split"]["val_fraction"], cfg["dataset"]["seed"]
    out = pd.Series("train", index=p.index)
    for _, g in p.groupby(["category", "broad_region"]):
        order = sorted(g.index, key=lambda i: hashlib.sha256(f"{seed}|hn_split|{p.at[i, 'hn_id']}".encode()).hexdigest())
        for i in order[: int(round(frac * len(order)))]:
            out[i] = "val"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", default=None, help="proposed hard-negative point list to check (no extraction)")
    args = ap.parse_args()
    cfg = load_config()
    ddir = REPO_ROOT / cfg["dataset"]["dataset_dir"]
    pos = pd.read_csv(ddir / "positive_audit.csv")
    meta = pd.read_csv(ddir / "external_tiles_metadata.csv")
    plan = build_plan(pos, meta, cfg)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plan.to_csv(REPORT_DIR / "external_hard_negatives_plan.csv", index=False)
    n_req = int(plan.expected_requests.sum())
    print(plan.groupby("category")[["n_facilities", "n_tiles", "expected_requests"]].sum().to_string())
    print(f"total: {plan.n_facilities.sum()} facilities, {plan.n_tiles.sum()} tiles, ~{n_req} requests, "
          f"~${tiered_cost_usd(n_req, cfg['pricing'], already_used_this_month=21298):.2f} this month (free cap used)")
    if args.validate:
        v = validate_points(pd.read_csv(args.validate), pos, cfg)
        v["split"] = assign_hard_negative_split(v, cfg)
        v.to_csv(ddir / "hard_negative_points_validated.csv", index=False)
        print(v["flags"].replace("", "ok").value_counts().to_string())


if __name__ == "__main__":
    main()
