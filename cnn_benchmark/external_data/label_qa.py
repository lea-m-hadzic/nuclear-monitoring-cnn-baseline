"""Label QA for the external positives and the provisional benchmark facility map. Reads only; changes no label.

    python label_qa.py positives   # external_imagery/dataset/label_qa/ (private: names, sheets, scores)
    python label_qa.py benchmark   # outputs/qa/benchmark_facility_checks.csv + sheets under external_imagery/.../label_qa/

Positives (all 287, no sampling):
  * co-location: positive pairs closer than one tile width (1.9 km, tiles overlap) and < 0.5 km, overlap share,
    whether the pair has different facility IDs, and whether it crosses splits;
  * out-of-sample model evidence, val positives only (56): score in val_predictions_best.csv of the 9 frozen
    A/B/C runs (shared val, never trained on). Train positives have no out-of-sample score; none is invented;
  * imagery flags from the extraction audit;
  * visual sheets of EVERY positive: 475 m x 475 m crop centred on Yanliang's fence point (from the 1280 px
    decoded cache), 12 per sheet in tile-ID order, for a full manual pass.
Benchmark (every facility in facilities_provisional.csv, no model scores used):
  * footprint tiles whose human_target is 0; human-positive tiles outside every footprint (counts only);
  * z16 / z17 centroid disagreement > 1 km; open author questions; footprints touching tiles missing from the cache;
  * visual sheets: each footprint tile with its 8 neighbours (z16 parent tiles), footprint outlined.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))
from geometry import REPO_ROOT  # noqa: E402

PRIV = REPO_ROOT / "external_imagery" / "dataset" / "label_qa"
PUB = PKG / "outputs" / "qa"
EXT_CACHE = PKG / "cache" / "external_decoded_1280px"
BENCH_CACHE = PKG / "cache" / "decoded_1280px"
RUNS = PKG / "outputs" / "ablation" / "runs"


def haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def positives():
    PRIV.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(REPO_ROOT / "external_imagery" / "dataset" / "external_tiles_metadata_refined.csv")
    p = m[m.role == "positive"].sort_values("tile_id").reset_index(drop=True)
    # co-location (tile centres; tiles are 1.9 km squares)
    la, lo = p.center_lat.to_numpy(), p.center_lon.to_numpy()
    d = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
    pairs = []
    for i, j in zip(*np.where(np.triu(d < 1.9, 1))):
        dy = abs(la[i] - la[j]) * 111.32
        dx = abs(lo[i] - lo[j]) * 111.32 * np.cos(np.radians(la[i]))
        overlap = max(0, 1.9 - dx) * max(0, 1.9 - dy) / 1.9 ** 2
        pairs.append({"tile_a": p.tile_id[i], "tile_b": p.tile_id[j], "dist_km": round(d[i, j], 3),
                      "tile_overlap_share": round(overlap, 3), "same_split_group": p.split_group_id[i] == p.split_group_id[j],
                      "same_split": p.split[i] == p.split[j], "name_a": p.facility_name_original[i], "name_b": p.facility_name_original[j]})
    pairs = pd.DataFrame(pairs).sort_values("dist_km")
    pairs.to_csv(PRIV / "positive_colocated_pairs.csv", index=False)
    # out-of-sample evidence for val positives
    preds = []
    for f in sorted(RUNS.glob("[ABC]/seed*/val_predictions_best.csv")):
        v = pd.read_csv(f)
        preds.append(v[v.label == 1].assign(run=f"{f.parts[-3]}/{f.parts[-2]}"))
    pv = pd.concat(preds).pivot(index="tile_id", columns="run", values="score")
    pv["mean_score"] = pv.mean(axis=1)
    pv["n_runs_below_0.5"] = (pv.drop(columns="mean_score") < 0.5).sum(axis=1)
    pv = pv.sort_values("mean_score")
    pv.to_csv(PRIV / "val_positive_oos_scores.csv")
    # per-positive table (private: names)
    t = p[["tile_id", "facility_id", "facility_name_original", "split_group_id", "split", "fence_offset_px",
           "fence_edge_margin_m", "source_zoom", "mosaic_grid", "qa_flags", "artifact_flags"]].copy()
    t["n_colocated_lt_1p9km"] = [int(((pairs.tile_a == k) | (pairs.tile_b == k)).sum()) for k in t.tile_id]
    t["val_oos_mean_score"] = t.tile_id.map(pv.mean_score)
    t["sheet"] = [f"pos_qa_{i // 12:03d}.png" for i in range(len(t))]
    t["visual_status"] = ""            # filled by the manual pass
    t["visual_notes"] = ""
    t.to_csv(PRIV / "positive_qa_table.csv", index=False)
    # sheets: 475 m crop around the fence point, every positive
    for s in range(0, len(p), 12):
        sub = p.iloc[s:s + 12]
        W, H = 4 * 330, 3 * 352
        sheet = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(sub.itertuples()):
            arr = np.load(EXT_CACHE / f"{r.tile_id}.npy")
            mpp = r.footprint_width_km * 1000 / 1280
            half = int(round(237.5 / mpp))
            cx, cy = int(r.fence_px_col / 2), int(r.fence_px_row / 2)
            x0, y0 = max(cx - half, 0), max(cy - half, 0)
            crop = Image.fromarray(arr[y0:cy + half, x0:cx + half]).resize((320, 320))
            gx, gy = (k % 4) * 330 + 5, (k // 4) * 352 + 26
            sheet.paste(crop, (gx, gy))
            px, py = gx + (cx - x0) * 320 / (2 * half), gy + (cy - y0) * 320 / (2 * half)
            dr.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(255, 0, 255), width=2)
            dr.text((gx, gy - 20), f"{s + k}: {r.tile_id[:44]} [{r.split}]", fill="black")
        sheet.save(PRIV / f"pos_qa_{s // 12:03d}.png")
    print(f"{len(p)} positives; co-located pairs <1.9 km: {len(pairs)} (<0.5 km: {(pairs.dist_km < 0.5).sum()}, "
          f"cross-split: {(~pairs.same_split).sum()}); "
          f"val positives with mean OOS score < 0.5: {(pv.mean_score < 0.5).sum()}/{len(pv)}; sheets: {(len(p) + 11) // 12}")


def benchmark():
    import facility_mapping as fm
    from common import OUTPUT_DIR, load_config
    PRIV.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    fac, assignments = fm.reconstruct(cfg)
    mem = fm.parent_membership(assignments)
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    pt = pt[pt.region.isin(["Saratov", "Natanz", "Fordow"])]
    lab = pt.set_index(["region", "tile_id"]).human_target
    rows = []
    for f in fac.itertuples():
        fp = mem[mem.facility_id == f.facility_id].tile_id.tolist()
        tl = [lab.get((f.region, t), np.nan) for t in fp]
        missing = [t for t in fp if not (BENCH_CACHE / f"{f.region}__{t}.npy").exists()]
        rows.append({"facility_id": f.facility_id, "region": f.region, "facility_type": f.facility_type,
                     "in_primary_denominator": f.in_primary_denominator, "footprint_tiles": ";".join(fp),
                     "n_footprint_tiles": len(fp), "footprint_tiles_human_target_0": sum(x == 0 for x in tl),
                     "z16_z17_centroid_km": f.z16_z17_centroid_km,
                     "centroid_disagreement_gt_1km": bool(f.z16_z17_centroid_km > 1) if pd.notna(f.z16_z17_centroid_km) else None,
                     "needs_author_confirmation": f.needs_author_confirmation, "footprint_tiles_missing_imagery": len(missing),
                     "visual_status": "", "visual_notes": ""})
    out = pd.DataFrame(rows)
    in_fp = set(zip(mem.region, mem.tile_id))
    pos = pt[pt.human_target == 1]
    orphan = pos[[(r, t) not in in_fp for r, t in zip(pos.region, pos.tile_id)]]
    out.to_csv(PUB / "benchmark_facility_checks.csv", index=False)
    pd.DataFrame({"region": orphan.region, "tile_id": orphan.tile_id, "positive_label_details": orphan.positive_label_details}).to_csv(
        PUB / "benchmark_positive_tiles_outside_footprints.csv", index=False)
    # sheets: footprint + 8-neighbourhood
    for f in out.itertuples():
        fp = f.footprint_tiles.split(";")
        rc = [tuple(int(x[1:]) for x in t.split("_")) for t in fp]
        r0, r1 = min(r for r, _ in rc) - 1, max(r for r, _ in rc) + 1
        c0, c1 = min(c for _, c in rc) - 1, max(c for _, c in rc) + 1
        S = 256
        sheet = Image.new("RGB", ((c1 - c0 + 1) * S, (r1 - r0 + 1) * S + 24), "white")
        dr = ImageDraw.Draw(sheet)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                k = BENCH_CACHE / f"{f.region}__r{r}_c{c}.npy"
                x, y = (c - c0) * S, (r - r0) * S + 24
                if k.exists():
                    sheet.paste(Image.fromarray(np.load(k)).resize((S, S)), (x, y))
                if (r, c) in rc:
                    dr.rectangle([x, y, x + S - 1, y + S - 1], outline=(255, 0, 255), width=3)
                else:
                    dr.rectangle([x, y, x + S - 1, y + S - 1], outline=(255, 255, 255), width=1)
                dr.text((x + 3, y + 3), f"r{r}_c{c} h={lab.get((f.region, f'r{r}_c{c}'), '-')}", fill=(255, 255, 0))
        dr.text((4, 4), f"{f.facility_id} {f.facility_type} footprint={f.footprint_tiles}", fill="black")
        sheet.save(PRIV / f"bench_{f.facility_id}.png")
    print(out.drop(columns=["visual_status", "visual_notes"]).to_string(index=False))
    print(f"human-positive tiles outside every facility footprint: {len(orphan)} of {len(pos)} "
          f"({orphan.region.value_counts().to_dict()})")


if __name__ == "__main__" and sys.argv[1] != "centroids":
    {"positives": positives, "benchmark": benchmark}[sys.argv[1]]()


def benchmark_centroids():
    """Where each facility's z16 / z17 label centroid falls relative to its footprint tiles + full-res crops."""
    import facility_mapping as fm
    from common import OUTPUT_DIR, load_config
    fac, assignments = fm.reconstruct(load_config())
    mem = fm.parent_membership(assignments)
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv").set_index(["region", "tile_id"])
    rows, crops = [], []
    for f in fac.itertuples():
        fp = mem[mem.facility_id == f.facility_id].tile_id.tolist()
        for which in ("z16", "z17"):
            la, lo = getattr(f, f"centroid_lat_{which}"), getattr(f, f"centroid_lon_{which}")
            if pd.isna(la):
                continue
            best = None
            for t in pt.loc[f.region].index:
                r = pt.loc[(f.region, t)]
                w = r.tile_km
                x = (lo - r.center_lon) * 111.32 * np.cos(np.radians(r.center_lat)) / w * 1280 + 640
                y = -(la - r.center_lat) * 111.32 / w * 1280 + 640
                if 0 <= x < 1280 and 0 <= y < 1280:
                    best = (t, x, y)
                    break
            t, x, y = best if best else (None, np.nan, np.nan)
            rows.append({"facility_id": f.facility_id, "centroid": which, "tile_containing_centroid": t,
                         "in_footprint": t in fp, "px_x": round(x) if t else None, "px_y": round(y) if t else None,
                         "edge_margin_px": int(min(x, y, 1279 - x, 1279 - y)) if t else None})
            if t and f.region == "Saratov":
                arr = np.load(BENCH_CACHE / f"Saratov__{t}.npy")
                h = 270                                                         # ~400 m half-width at 1.49 m/px
                pad = np.pad(arr, ((h, h), (h, h), (0, 0)), constant_values=255)
                c = Image.fromarray(pad[int(y):int(y) + 2 * h, int(x):int(x) + 2 * h]).resize((360, 360))
                ImageDraw.Draw(c).text((4, 4), f"{f.facility_id} {which} {t}", fill=(255, 255, 0))
                ImageDraw.Draw(c).ellipse([172, 172, 188, 188], outline=(255, 0, 255), width=2)
                crops.append(c)
    out = pd.DataFrame(rows)
    out.to_csv(PUB / "benchmark_facility_centroids.csv", index=False)
    for s in range(0, len(crops), 12):
        sheet = Image.new("RGB", (4 * 365, 3 * 365), "white")
        for k, c in enumerate(crops[s:s + 12]):
            sheet.paste(c, ((k % 4) * 365, (k // 4) * 365))
        sheet.save(PRIV / f"bench_centroids_{s // 12:02d}.png")
    print(out.to_string(index=False))


if __name__ == "__main__" and sys.argv[1] == "centroids":
    benchmark_centroids()
