"""Audit region-identifying imaging artifacts (watermarks, mosaic seams, sharpness) by AOI.

NON-DESTRUCTIVE: nothing in the pipeline is changed; masking is only previewed in memory.

At the model input scale (1280 px), for a random sample of parent tiles per AOI:
  * fixed-pattern map = per-pixel fraction of tiles with strong local contrast (|gray - blur| > T).
    Natural content varies tile to tile; watermark text and mosaic seams sit at fixed positions,
    so they show up as pixels with a high fraction.
  * watermark mask per AOI = pixels whose fraction exceeds `--mask-thresh` (dilated).
  * seam profile = mean |horizontal / vertical gradient| per column / row; peaks = mosaic seams.
  * per-tile artifact features and their AUC for separating AOIs, before and after masking the union of
    the AOI watermark masks (filled by normalized-convolution blur, a preview only).

  python src/artifact_audit.py [--n 120] [--supplementary-pairwise]  -> results/qa/artifact_audit/
The committed audit (2026-10-09) samples Natanz from all 776 canonical tiles; the earlier 2026-10-08 run had only 246.
Add --supplementary-pairwise for all AOI pairs (incl. Natanz) before/after the union-mask fill.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

import image_index
from common import OUTPUT_DIR, REPO_ROOT

SIZE = 1280
HP_SIGMA = 3.0
HP_T = 18.0           # gray levels (0-255)


def _gray(locator: str) -> np.ndarray:
    im = image_index.open_image(locator, REPO_ROOT).convert("RGB")
    im = im.reduce(im.size[0] // SIZE) if im.size[0] % SIZE == 0 else im.resize((SIZE, SIZE))
    return np.asarray(im.convert("L"), dtype=np.float32)


def _tile_stats(locator: str):
    g = _gray(locator)
    hp = np.abs(g - ndimage.gaussian_filter(g, HP_SIGMA))
    gx = np.abs(np.diff(g, axis=1)).mean(0)      # per column boundary
    gy = np.abs(np.diff(g, axis=0)).mean(1)      # per row boundary
    return (hp > HP_T).astype(np.uint8), gx, gy, hp


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) via ranks (0.5 = no separation)."""
    s = np.r_[pos, neg]
    r = pd.Series(s).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fill_masked(g: np.ndarray, mask: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    valid = (~mask).astype(np.float32)
    num = ndimage.gaussian_filter(g * valid, sigma)
    den = ndimage.gaussian_filter(valid, sigma)
    out = g.copy()
    out[mask] = (num / np.maximum(den, 1e-3))[mask]
    return out


def peaks(profile: np.ndarray, k: int = 6) -> list[tuple[int, float]]:
    base = ndimage.median_filter(profile, 31)
    ratio = profile / np.maximum(base, 1e-6)
    idx = [i for i in np.argsort(-ratio) if 8 < i < len(profile) - 8]
    chosen = []
    for i in idx:
        if all(abs(i - j) > 20 for j, _ in chosen):
            chosen.append((int(i), float(ratio[i])))
        if len(chosen) == k:
            break
    return sorted(chosen)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--mask-thresh", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--supplementary-pairwise", action="store_true",
                    help="also write supplementary_pairwise_separation.csv: watermark-contrast separation for every AOI "
                         "pair (incl. Natanz), before/after the union-mask fill. Default outputs are unchanged.")
    args = ap.parse_args(argv)
    out = OUTPUT_DIR.parent.parent / "results" / "qa" / "artifact_audit"
    out.mkdir(parents=True, exist_ok=True)

    root = image_index.image_root_from(None)
    idx = pd.read_csv(OUTPUT_DIR / "image_index.csv")
    idx = idx[idx.status == "ok"]
    regions = ["Saratov", "Fordow", "Natanz"]
    sample = pd.concat([g.sample(min(args.n, len(g)), random_state=args.seed) for _, g in idx.groupby("region")])
    sample = sample[sample.region.isin(regions)]
    locs = [str(root / l) for l in sample.locator]

    freq, colp, rowp, hps = {}, {}, {}, {}
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_tile_stats, locs, chunksize=4))
    for region in regions:
        sel = [r for r, reg in zip(res, sample.region) if reg == region]
        freq[region] = np.mean([r[0] for r in sel], axis=0)
        colp[region] = np.mean([r[1] for r in sel], axis=0)
        rowp[region] = np.mean([r[2] for r in sel], axis=0)
    masks = {r: ndimage.binary_dilation(freq[r] > args.mask_thresh, iterations=3) for r in regions}
    union = masks["Saratov"] | masks["Fordow"] | masks["Natanz"]

    # Per-tile artifact features, before/after masking the union of watermark masks.
    rows = []
    for (region, tile, loc), r in zip(sample[["region", "tile_id", "locator"]].itertuples(index=False), res):
        hp = r[3]
        feat = {"region": region, "tile_id": tile}
        for mreg in ("Saratov", "Fordow"):
            m = masks[mreg]
            feat[f"wm_contrast_{mreg}_mask"] = float(hp[m].mean() - hp[~union].mean())
        for name, prof in (("col", r[1]), ("row", r[2])):
            for preg in ("Saratov", "Fordow"):
                pk = [i for i, _ in peaks(colp[preg] if name == "col" else rowp[preg], 3)]
                base = ndimage.median_filter(prof, 31)
                feat[f"seam_{name}_{preg}_grid"] = float(np.mean(prof[pk] / np.maximum(base[pk], 1e-6)))
        rows.append(feat)
    F = pd.DataFrame(rows)

    # Masked preview: recompute watermark-contrast features after filling the union mask.
    masked = []
    for (region, tile, loc) in sample[["region", "tile_id", "locator"]].itertuples(index=False):
        if region == "Natanz":
            continue
        g = fill_masked(_gray(str(root / loc)), union)
        hp = np.abs(g - ndimage.gaussian_filter(g, HP_SIGMA))
        masked.append({"region": region, "tile_id": tile,
                       **{f"wm_contrast_{m}_mask_after": float(hp[masks[m]].mean() - hp[~union].mean())
                          for m in ("Saratov", "Fordow")}})
    M = pd.DataFrame(masked)
    F = F.merge(M, on=["region", "tile_id"], how="left")
    F.to_csv(out / "tile_artifact_features.csv", index=False)

    sf = F[F.region.isin(["Saratov", "Fordow"])]
    auc_rows = []
    for c in [c for c in F.columns if c.startswith(("wm_", "seam_"))]:
        a = auc(sf[sf.region == "Saratov"][c].dropna().to_numpy(), sf[sf.region == "Fordow"][c].dropna().to_numpy())
        auc_rows.append(dict(feature=c, auc_saratov_vs_fordow=round(a, 3), separation=round(abs(a - 0.5) * 2, 3)))
    A = pd.DataFrame(auc_rows)
    A.to_csv(out / "artifact_feature_auc.csv", index=False)

    summary = {"n_tiles": sample.region.value_counts().to_dict(), "input_px": SIZE,
               "watermark_mask_area_pct": {r: round(100 * masks[r].mean(), 2) for r in regions},
               "union_mask_area_pct": round(100 * union.mean(), 2),
               "fixed_pattern_pixels_pct(freq>thresh)": {r: round(100 * (freq[r] > args.mask_thresh).mean(), 3) for r in regions},
               "column_seam_peaks(px@1280, ratio)": {r: peaks(colp[r]) for r in regions},
               "row_seam_peaks(px@1280, ratio)": {r: peaks(rowp[r]) for r in regions},
               "mask_components": {r: int(ndimage.label(masks[r])[1]) for r in regions}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Figures (statistics only; no raw imagery).
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, r in zip(axs, regions):
        im = ax.imshow(freq[r], vmin=0, vmax=1, cmap="magma")
        ax.set_title(f"{r}: fraction of tiles with strong local contrast\n(n={int((sample.region == r).sum())}; "
                     f"bright = fixed-position artifact)", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axs, shrink=0.7)
    fig.savefig(out / "fixed_pattern_maps.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for r in regions:
        axs[0].plot(colp[r] / ndimage.median_filter(colp[r], 31), label=r, lw=0.8)
        axs[1].plot(rowp[r] / ndimage.median_filter(rowp[r], 31), label=r, lw=0.8)
    axs[0].set_ylabel("column |dI/dx| / local median"); axs[1].set_ylabel("row |dI/dy| / local median")
    axs[1].set_xlabel("pixel position at 1280 px input"); axs[0].legend(); fig.suptitle("Mosaic seam profiles (peaks = seams)")
    fig.tight_layout(); fig.savefig(out / "seam_profiles.png", dpi=90); plt.close(fig)
    print(json.dumps(summary, indent=1)); print(A.to_string(index=False))

    if args.supplementary_pairwise:
        # Same sample, masks, features and fill as above, extended to all three AOI masks and AOI pairs.
        sup = []
        for (region, tile, loc), r in zip(sample[["region", "tile_id", "locator"]].itertuples(index=False), res):
            hp_b = r[3]
            g = fill_masked(_gray(str(root / loc)), union)
            hp_a = np.abs(g - ndimage.gaussian_filter(g, HP_SIGMA))
            rec = {"region": region, "tile_id": tile}
            for m in regions:
                rec[f"{m}_before"] = float(hp_b[masks[m]].mean() - hp_b[~union].mean())
                rec[f"{m}_after"] = float(hp_a[masks[m]].mean() - hp_a[~union].mean())
            sup.append(rec)
        S = pd.DataFrame(sup)
        rows = []
        for a, b in (("Saratov", "Fordow"), ("Saratov", "Natanz"), ("Fordow", "Natanz")):
            for m in regions:
                sep = {st: abs(auc(S[S.region == a][f"{m}_{st}"].to_numpy(), S[S.region == b][f"{m}_{st}"].to_numpy()) - 0.5) * 2
                       for st in ("before", "after")}
                rows.append(dict(pair=f"{a} vs {b}", feature=f"wm_contrast_{m}_mask",
                                 separation_before=round(sep["before"], 3), separation_after=round(sep["after"], 3)))
        P = pd.DataFrame(rows)
        P.to_csv(out / "supplementary_pairwise_separation.csv", index=False)
        print("\nSupplementary pairwise separation (2*|AUC-0.5|):"); print(P.to_string(index=False))


if __name__ == "__main__":
    main()
