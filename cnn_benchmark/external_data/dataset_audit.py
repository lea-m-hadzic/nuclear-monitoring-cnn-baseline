"""Post-extraction audit of the external training set. Makes NO API calls.

Reads tile_plan.csv + positive_audit.csv (dataset_plan.py) and the extracted tiles, then writes
  <dataset_dir>/external_tiles_metadata.csv        one row per tile (all fields used later by training)
  <dataset_dir>/candidate_negative_review.csv      manual review sheet (decisions start EMPTY)
  <dataset_dir>/positive_audit.csv                 (rewritten with extraction status)
  <dataset_dir>/near_duplicate_pairs.csv           image-hash near duplicates
  <dataset_dir>/contact_sheets/sheet_NNN.png       one row per facility: positive (+ fence marker) and its candidates
  cnn_benchmark/outputs/external_data/external_dataset_audit.md   aggregate report, NO coordinates
  cnn_benchmark/outputs/external_data/jitter_and_centre_bias.png  aggregate plot, no imagery

Watermark check reuses src/artifact_audit.py (same 1280 px high-pass statistics, frequency-map masks and AUC).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import artifact_audit as aa  # noqa: E402  (shared watermark logic)

from geometry import REPO_ROOT, load_config, tile_geometry  # noqa: E402
from validate_external_tiles import check_scan_dir  # noqa: E402
from secrets_guard import scan_dir_for_secrets  # noqa: E402

PRESET = "auto_mosaic_1p9km"
REPORT_DIR = HERE.parent / "outputs" / "external_data"
DHASH_NEAR = 12            # of 256 bits
PLACEHOLDER_RGB = (229, 227, 223)   # Google "Sorry, we have no imagery here" tile background (measured)
SOURCE_PX = 1280                    # one Static Maps image (size 640, scale 2)
MANUAL_REVIEW_OWNER = "negative_refinement.py"   # writes candidate_negative_review.csv


def tile_dir(output_root: Path, r) -> Path:
    return output_root / PRESET / r["facility_id"] / r["tile_id"]


def image_stats(png: str) -> dict:
    """Content QA + dHash + the shared artifact-audit high-pass map (1280 px)."""
    try:
        hpbin, gx, gy, hp = aa._tile_stats(png)
        g = np.asarray(Image.open(png).convert("L").resize((640, 640)), dtype=np.float32)
    except Exception as e:                                  # corrupt
        return {"png": png, "error": f"{type(e).__name__}: {e}"}
    blocks = g[: 640 // 20 * 20, : 640 // 20 * 20].reshape(32, 20, 32, 20).std(axis=(1, 3))
    rgb = np.asarray(Image.open(png).convert("RGB").reduce(4), dtype=np.int16)        # box filter keeps flat areas exact
    placeholder = float((np.abs(rgb - np.array(PLACEHOLDER_RGB)).max(-1) <= 1).mean())
    black = float((rgb.max(-1) <= 3).mean())                                          # solid no-data blocks
    small = np.asarray(Image.open(png).convert("L").resize((17, 16)), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return {"png": png, "error": "", "gray_mean": float(g.mean()), "gray_std": float(g.std()),
            "featureless_frac": float((blocks < 3.0).mean()), "bright_frac": float((g > 225).mean()),
            "no_imagery_frac": placeholder, "black_frac": black,
            "dhash": np.packbits(bits).tobytes().hex(), "hpbin": hpbin, "hp": hp}


def qa_flags(s: dict) -> str:
    f = []
    if s.get("error"):
        f.append("corrupt")
    else:
        if s["no_imagery_frac"] > 0.5:
            f.append("no_imagery_placeholder")
        elif s["no_imagery_frac"] > 0.02:
            f.append("partial_no_imagery")
        if s["black_frac"] > 0.01:
            f.append("black_nodata")
        if s["gray_std"] < 4:
            f.append("blank")
        if s["featureless_frac"] > 0.6 and s["no_imagery_frac"] <= 0.5:
            f.append("mostly_featureless(water/haze?)")
        if s["bright_frac"] > 0.3 and s["no_imagery_frac"] <= 0.5:
            f.append("bright(cloud/snow?)")
    return ";".join(f)


def hamming_matrix(hexes: list[str]) -> np.ndarray:
    b = np.unpackbits(np.array([np.frombuffer(bytes.fromhex(h), np.uint8) for h in hexes]), axis=1).astype(np.uint8)
    return (b[:, None, :] != b[None, :, :]).sum(-1)


def contact_sheets(meta: pd.DataFrame, out_dir: Path, rows_per_sheet: int = 6, thumb: int = 360) -> pd.DataFrame:
    """One row per facility: [positive with fence marker | neg1..negK]. Returns tile_id -> sheet/position."""
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    fac = list(dict.fromkeys(meta.sort_values(["split", "split_group_id", "facility_id"])["facility_id"]))
    ncol = 1 + int(meta[meta.role == "candidate_negative"].groupby("facility_id").size().max())
    where = []
    for s, start in enumerate(range(0, len(fac), rows_per_sheet)):
        chunk = fac[start:start + rows_per_sheet]
        sheet = Image.new("RGB", (ncol * (thumb + 6) + 6, len(chunk) * (thumb + 34) + 6), "white")
        dr = ImageDraw.Draw(sheet)
        for i, f in enumerate(chunk):
            sub = meta[meta.facility_id == f].sort_values("role", key=lambda c: c != "positive")
            for j, r in enumerate(sub.itertuples()):
                x, y = 6 + j * (thumb + 6), 6 + i * (thumb + 34)
                if isinstance(r.png_path, str) and Path(REPO_ROOT / r.png_path).exists():
                    im = Image.open(REPO_ROOT / r.png_path).convert("RGB").resize((thumb, thumb))
                    if r.role == "positive" and pd.notna(r.fence_px_col):
                        k = thumb / 2560
                        cx, cy = r.fence_px_col * k, r.fence_px_row * k
                        ImageDraw.Draw(im).ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=(255, 0, 0), width=3)
                    sheet.paste(im, (x, y))
                label = ("POS " if r.role == "positive"
                         else f"CAND {r.tile_id.rsplit('__', 1)[-1]} {r.candidate_subset.upper()} ") + \
                        (f"[{r.qa_flags}]" if r.qa_flags else "")
                dr.text((x, y + thumb + 2), label[:60], fill="black", font=font)
                dr.text((x, y + thumb + 16), f"{r.tile_id}"[:60], fill="gray", font=font)
                where.append({"tile_id": r.tile_id, "contact_sheet": f"sheet_{s + 1:03d}.png",
                              "sheet_row": i + 1, "sheet_col": j + 1})
        sheet.save(out_dir / f"sheet_{s + 1:03d}.png")
    return pd.DataFrame(where)


def positive_sheets(meta: pd.DataFrame, out_dir: Path, per_row: int = 3, rows_per_sheet: int = 4,
                    thumb: int = 360, zoom_px: int = 640) -> pd.DataFrame:
    """Positive review: for each positive, [full tile with the original fence point circled | zoom around the point].

    zoom_px source pixels (~475 m at 0.742 m/px) centred on the fence point are shown at `thumb` px. The label is
    Yanliang's; this sheet only lets a reviewer confirm the layered-fence signature is visible in current imagery."""
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    p = meta[meta.role == "positive"].sort_values(["split", "split_group_id", "facility_id"])
    per_sheet = per_row * rows_per_sheet
    cell_w = 2 * thumb + 12
    where = []
    for s_i, start in enumerate(range(0, len(p), per_sheet)):
        chunk = p.iloc[start:start + per_sheet]
        sheet = Image.new("RGB", (per_row * (cell_w + 10) + 6, rows_per_sheet * (thumb + 34) + 6), "white")
        dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk.itertuples()):
            x, y = 6 + (k % per_row) * (cell_w + 10), 6 + (k // per_row) * (thumb + 34)
            if isinstance(r.png_path, str) and (REPO_ROOT / r.png_path).exists():
                im = Image.open(REPO_ROOT / r.png_path).convert("RGB")
                cx, cy = float(r.fence_px_col), float(r.fence_px_row)
                full = im.resize((thumb, thumb))
                kf = thumb / im.size[0]
                ImageDraw.Draw(full).ellipse([cx * kf - 9, cy * kf - 9, cx * kf + 9, cy * kf + 9], outline=(255, 0, 0), width=3)
                h = zoom_px / 2
                box = (int(cx - h), int(cy - h), int(cx + h), int(cy + h))
                z = im.crop(box).resize((thumb, thumb))
                kz = thumb / zoom_px
                zx, zy = (cx - box[0]) * kz, (cy - box[1]) * kz
                dz = ImageDraw.Draw(z)
                dz.ellipse([zx - 14, zy - 14, zx + 14, zy + 14], outline=(255, 0, 0), width=3)
                dz.line([zx - 24, zy, zx - 16, zy], fill=(255, 0, 0), width=2); dz.line([zx + 16, zy, zx + 24, zy], fill=(255, 0, 0), width=2)
                sheet.paste(full, (x, y)); sheet.paste(z, (x + thumb + 6, y))
            dr.text((x, y + thumb + 2), f"{r.facility_name_original}"[:70], fill="black", font=font)
            dr.text((x, y + thumb + 16), f"{r.tile_id} | right: ~{zoom_px * 1900 / 2560:.0f} m zoom", fill="gray", font=font)
            where.append({"tile_id": r.tile_id, "positive_sheet": f"pos_sheet_{s_i + 1:03d}.png",
                          "positive_sheet_index": k + 1})
        sheet.save(out_dir / f"pos_sheet_{s_i + 1:03d}.png")
    return pd.DataFrame(where)


def keep_decisions(new: pd.DataFrame, path: Path, cols: list[str], filled: str) -> pd.DataFrame:
    """Carry over any decisions already entered in an existing review CSV (never overwrite reviews)."""
    if not path.exists():
        return new
    old = pd.read_csv(path, dtype=str, keep_default_na=False)
    keep = old[(old[filled] != "") & (old[filled] != "unreviewed")][["tile_id"] + cols]
    if keep.empty:
        return new
    return new.drop(columns=cols).merge(keep, on="tile_id", how="left").fillna({c: "" for c in cols})


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    t = pd.crosstab(a, b).to_numpy().astype(float)
    if min(t.shape) < 2:
        return 0.0
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    chi2 = ((t - e) ** 2 / e).sum()
    return float(math.sqrt(chi2 / t.sum() / (min(t.shape) - 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-contact-sheets", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    ddir = REPO_ROOT / cfg["dataset"]["dataset_dir"]
    oroot = REPO_ROOT / cfg["output_root"]
    plan = pd.read_csv(ddir / "tile_plan.csv")
    pos_audit = pd.read_csv(ddir / "positive_audit.csv")
    plan = plan[plan.status == "planned"].reset_index(drop=True)

    # ---- extraction records + validation (shared validator)
    rows = []
    for r in plan.to_dict("records"):
        d = tile_dir(oroot, r)
        rec = json.loads((d / "extraction_record.json").read_text()) if (d / "extraction_record.json").exists() else {}
        tiles, problems = check_scan_dir(d, PRESET, False, None, 0.03) if (d / "manifest.csv").exists() else ([], ["not extracted"])
        t = tiles[0] if tiles else {}
        mg = dict(rec.get("manifest_geometry") or {})
        if "footprint_km" in mg:                      # records written before the width/area rename
            mg["footprint_width_km"] = mg.pop("footprint_km")
        pngs = sorted((d / "tiles").glob("*.png")) if (d / "tiles").exists() else []
        rows.append({**r, "extraction_status": rec.get("status", "missing"),
                     "requests_metered": rec.get("requests_metered"),
                     "source_zoom": mg.get("zoom"), "mosaic_grid": mg.get("grid_n"), "crop_px": mg.get("crop_px"),
                     "native_mpp": mg.get("native_mpp"), "footprint_width_km": mg.get("footprint_width_km"),
                     "footprint_area_km2": round(mg["footprint_width_km"] ** 2, 4) if mg else None,
                     "effective_mpp": round(mg["footprint_width_km"] * 1000 / 2560, 5) if mg else None,
                     "validation_errors": "; ".join((t.get("errors") or []) + problems),
                     "png_path": str(pngs[0].relative_to(REPO_ROOT)) if pngs else None})
    meta = pd.DataFrame(rows)
    meta["validation_status"] = np.where((meta.extraction_status == "complete") & (meta.validation_errors == ""),
                                         "ok", "failed")

    # ---- image QA (parallel), dHash, watermark statistics
    ok = meta[meta.png_path.notna()]
    with ProcessPoolExecutor(args.workers) as ex:
        stats = list(ex.map(image_stats, [str(REPO_ROOT / p) for p in ok.png_path], chunksize=8))
    st = {s["png"]: s for s in stats}
    meta["qa_flags"] = [qa_flags(st[str(REPO_ROOT / p)]) if isinstance(p, str) else "not_extracted" for p in meta.png_path]
    for c in ("gray_mean", "gray_std", "featureless_frac", "bright_frac", "no_imagery_frac", "black_frac", "dhash"):
        meta[c] = [st.get(str(REPO_ROOT / p), {}).get(c) if isinstance(p, str) else None for p in meta.png_path]

    good = meta[meta.dhash.notna()].reset_index(drop=True)
    H = hamming_matrix(good.dhash.tolist())
    np.fill_diagonal(H, 999)
    ii, jj = np.where(np.triu(H <= DHASH_NEAR, 1))
    dups = pd.DataFrame({"a": good.tile_id[ii].values, "b": good.tile_id[jj].values, "hamming": H[ii, jj],
                         "role_a": good.role[ii].values, "role_b": good.role[jj].values,
                         "split_group_a": good.split_group_id[ii].values, "split_group_b": good.split_group_id[jj].values,
                         "split_a": good.split[ii].values, "split_b": good.split[jj].values})
    flag_by_id = meta.set_index("tile_id")["qa_flags"].fillna("")
    dups["kind"] = ["placeholder_pair" if ("no_imagery_placeholder" in flag_by_id[a] and "no_imagery_placeholder" in flag_by_id[b])
                    else "genuine_candidate" for a, b in zip(dups.a, dups.b)]
    dups.to_csv(ddir / "near_duplicate_pairs.csv", index=False)
    gdup = dups[dups.kind == "genuine_candidate"]

    wm = {}
    for gn, sub in good.groupby("mosaic_grid"):
        keys = [str(REPO_ROOT / p) for p in sub.png_path]
        freq = np.mean([st[k]["hpbin"] for k in keys], axis=0)
        mask = ndimage.binary_dilation(freq > 0.5, iterations=3)
        contrast = np.array([st[k]["hp"][mask].mean() - st[k]["hp"][~mask].mean() for k in keys])
        is_pos = (sub.role == "positive").to_numpy()
        wm[int(gn)] = {"n_tiles": int(len(sub)), "mask_area_pct": round(100 * float(mask.mean()), 2),
                       "auc_wm_contrast_pos_vs_neg": round(aa.auc(contrast[is_pos], contrast[~is_pos]), 3)
                       if is_pos.any() and (~is_pos).any() else None}
    # Layout check that does not need a fixed-position mask: in auto_mosaic the "Google"/attribution strip sits at
    # the bottom of every 1280 px source image, so its row in the output is set by the centre-crop offset.
    phase = ((good.mosaic_grid * SOURCE_PX - good.crop_px) / 2 * -1 % SOURCE_PX) * 2560 / good.crop_px
    good = good.assign(wm_phase_px=phase.round(1))
    wm_phase = {}
    for gn, sub in good.groupby("mosaic_grid"):
        ip = (sub.role == "positive").to_numpy()
        wm_phase[int(gn)] = {"auc_phase_pos_vs_neg": round(aa.auc(sub.wm_phase_px.to_numpy()[ip], sub.wm_phase_px.to_numpy()[~ip]), 3),
                             "phase_px_range": [float(sub.wm_phase_px.min()), float(sub.wm_phase_px.max())],
                             "fixed_mask_possible": bool(wm[int(gn)]["mask_area_pct"] > 0)}
    meta = meta.merge(good[["tile_id", "wm_phase_px"]], on="tile_id", how="left")
    is_pos = (good.role == "positive").to_numpy()
    simple_auc = {c: round(aa.auc(good[c].to_numpy()[is_pos], good[c].to_numpy()[~is_pos]), 3)
                  for c in ("gray_mean", "gray_std", "featureless_frac", "bright_frac")}
    real = ~good.qa_flags.fillna("").str.contains("no_imagery_placeholder").to_numpy()
    simple_auc_real = {c: round(aa.auc(good[c].to_numpy()[is_pos & real], good[c].to_numpy()[~is_pos & real]), 3)
                       for c in ("gray_mean", "gray_std", "featureless_frac", "bright_frac")}

    # ---- review sheet + contact sheets
    if not args.no_contact_sheets:
        where = contact_sheets(meta, ddir / "contact_sheets" / "negatives")
        meta = meta.merge(where, on="tile_id", how="left")
        meta = meta.merge(positive_sheets(meta, ddir / "contact_sheets" / "positives"), on="tile_id", how="left")

    # positive label QA sheet (Yanliang's labels are not changed; status starts "unreviewed")
    pos_rev = meta[meta.role == "positive"][["tile_id", "facility_id", "facility_name_original", "split_group_id", "split",
                                             "latitude_original", "longitude_original", "center_lat", "center_lon",
                                             "jitter_m", "fence_px_col", "fence_px_row", "qa_flags", "validation_status",
                                             "png_path"] + [c for c in ("positive_sheet", "positive_sheet_index")
                                                            if c in meta]].copy()
    pos_rev["positive_review_status"] = "unreviewed"   # later: fence_visible | fence_not_visible | ambiguous
    pos_rev["positive_review_notes"] = ""
    pos_rev_path = ddir / "positive_review.csv"
    pos_rev = keep_decisions(pos_rev, pos_rev_path, ["positive_review_status", "positive_review_notes"],
                             "positive_review_status")
    pos_rev.to_csv(pos_rev_path, index=False)
    meta = meta.merge(pos_rev[["tile_id", "positive_review_status", "positive_review_notes"]], on="tile_id", how="left")
    # The negative review sheet (candidate_negative_review.csv, `manual_review_status`) is owned by
    # negative_refinement.py, which also preserves entered decisions; this audit never writes it.
    meta.to_csv(ddir / "external_tiles_metadata.csv", index=False)
    split_group_counts = (meta.groupby(["split", "split_group_id"]).candidate_subset.value_counts().unstack(fill_value=0)
                          .reset_index())
    split_group_counts.to_csv(ddir / "split_group_counts.csv", index=False)
    pa = pos_audit.drop(columns=["extraction_status", "validation_status", "qa_flags"], errors="ignore").merge(meta[meta.role == "positive"][["tile_id", "extraction_status", "validation_status", "qa_flags"]]
                         .rename(columns={"tile_id": "point_id"}), on="point_id", how="left")
    pa.to_csv(ddir / "positive_audit.csv", index=False)

    # ---- aggregate figure (no imagery)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = meta[meta.role == "positive"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].hist(p.jitter_m, bins=25, color="#4c78a8"); ax[0].set_xlabel("jitter distance (m)"); ax[0].set_ylabel("positives")
    ax[1].hist(p.jitter_bearing_deg, bins=24, color="#4c78a8"); ax[1].set_xlabel("jitter bearing (deg)")
    ax[2].scatter(p.fence_px_col, p.fence_px_row, s=6, color="#e45756")
    ax[2].add_patch(plt.Rectangle((0, 0), 2560, 2560, fill=False)); ax[2].set_xlim(0, 2560); ax[2].set_ylim(2560, 0)
    ax[2].set_aspect("equal"); ax[2].set_title("fence point position in positive tiles (px)", fontsize=9)
    fig.tight_layout(); fig.savefig(REPORT_DIR / "jitter_and_centre_bias.png", dpi=90); plt.close(fig)

    # ---- secrets / git hygiene
    leak_files = scan_dir_for_secrets(oroot / PRESET)

    summary = {
        "tiles_planned": int(len(meta)), "by_role": meta.role.value_counts().to_dict(),
        "extraction_status": meta.extraction_status.value_counts().to_dict(),
        "validation_status": meta.validation_status.value_counts().to_dict(),
        "requests_metered_total": int(meta.requests_metered.fillna(0).sum()),
        "qa_flags": meta.qa_flags.replace("", "none").str.split(";").explode().value_counts().to_dict(),
        "qa_flags_by_role": pd.crosstab(meta.role, meta.qa_flags.replace("", "none")).to_dict(),
        "near_duplicates": {"pairs": int(len(dups)), "placeholder_pairs": int((dups.kind == "placeholder_pair").sum()),
                            "genuine_pairs": int(len(gdup)),
                            "genuine_cross_split": int((gdup.split_a != gdup.split_b).sum()) if len(gdup) else 0,
                            "cross_split": int((dups.split_a != dups.split_b).sum()) if len(dups) else 0,
                            "cross_split_group": int((dups.split_group_a != dups.split_group_b).sum()) if len(dups) else 0,
                            "by_roles": {f"{a}|{b}": int(v) for (a, b), v in dups.groupby(["role_a", "role_b"]).size().items()} if len(dups) else {}},
        "watermark_by_grid": wm, "watermark_phase_by_grid": wm_phase,
        "no_imagery": {"placeholder_tiles_by_role": meta[meta.qa_flags.fillna("").str.contains("no_imagery_placeholder")]
                                                    .groupby("role").size().to_dict(),
                       "partial_tiles_by_role": meta[meta.qa_flags.fillna("").str.contains("partial_no_imagery")]
                                                .groupby("role").size().to_dict(),
                       "placeholder_by_subset": meta[meta.qa_flags.fillna("").str.contains("no_imagery_placeholder")]
                                                .groupby("candidate_subset").size().to_dict()}, "simple_stat_auc_pos_vs_neg": simple_auc,
        "simple_stat_auc_pos_vs_neg_excl_placeholders": simple_auc_real,
        "grid_by_role": pd.crosstab(meta.role, meta.mosaic_grid).to_dict(),
        "zoom_by_role": pd.crosstab(meta.role, meta.source_zoom).to_dict(),
        "footprint_width_km_range": [float(meta.footprint_width_km.min()), float(meta.footprint_width_km.max())],
        "footprint_area_km2_range": [float(meta.footprint_area_km2.min()), float(meta.footprint_area_km2.max())],
        "effective_mpp_range": [float(meta.effective_mpp.min()), float(meta.effective_mpp.max())],
        "cramers_v_region_vs_role": round(cramers_v(meta.broad_region, meta.role), 4),
        "cramers_v_grid_vs_role": round(cramers_v(meta.mosaic_grid.astype(str), meta.role), 4),
        "split_counts": pd.crosstab(meta.split, meta.role).to_dict(),
        "split_counts_by_subset": pd.crosstab(meta.split, meta.candidate_subset).to_dict(),
        "validation_by_subset": pd.crosstab(meta.candidate_subset, meta.validation_status).to_dict(),
        "split_groups": meta.groupby("split").split_group_id.nunique().to_dict(),
        "split_facilities": meta.groupby("split").facility_id.nunique().to_dict(),
        "facilities_in_both_splits": int((meta.groupby("facility_id").split.nunique() > 1).sum()),
        "fence_px": {"offset_px_quantiles": p.fence_offset_px.quantile([0, .5, 1]).round(1).tolist(),
                     "mean_col": round(float(p.fence_px_col.mean()), 1), "mean_row": round(float(p.fence_px_row.mean()), 1),
                     "min_edge_margin_m": float(p.fence_edge_margin_m.min())},
        "files_containing_key": [str(x) for x in leak_files],
    }
    (ddir / "audit_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
