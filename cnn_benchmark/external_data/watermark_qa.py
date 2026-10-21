"""QA of the per-tile geometric watermark mask (watermark_geometry.py). No training, no API calls.

    python watermark_qa.py      # -> outputs/ablation/mask_qa/summary.json (no imagery, no coordinates)
                                #    external_imagery/dataset/mask_qa/*.png (visual panels; private, git-ignored)

1. Placeholder ground truth: the 70 grey "no imagery" tiles show the overlay on a flat background. Known overlay
   pixels = cache pixels differing from the placeholder grey by > 25 levels, in source images that are otherwise
   blank (the "Sorry, we have no imagery here" line at source v 560..720 excluded). Coverage = share inside the mask.
2. Real imagery, every layout: per tile, high-pass map (artifact_audit: |g - gauss(g, 3)| > 18) before and after the
   fill, accumulated in the SOURCE-image frame (u, v) per domain. Known watermark source pixels = before-fill
   frequency > 0.5 (the fixed overlay; scene content is not aligned across tiles). Reported: coverage of those
   pixels by the template, residual frequency after the fill, and the background frequency for reference.
   Domains: Saratov / Natanz / Fordow (120 seeded tiles each) and all 669 external ablation tiles by layout.
3. Over-masking: per source image, the attribution box's actual left edge is estimated on real imagery (darkened
   bottom band vs the band above it); mask area beyond the estimated actual overlay is reported as % of tile area.
4. Visual panels (original | mask | filled) for Saratov, Natanz, Fordow and external z16/3x3, z17/3x3, z17/4x4.
"""
from __future__ import annotations

import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(HERE))
import artifact_audit as aa  # noqa: E402
import decoded_cache  # noqa: E402
from geometry import REPO_ROOT, load_config  # noqa: E402
from watermark_geometry import CACHE_PX, RECTS, SOURCE_PX, MaskBank, geometry_mask  # noqa: E402
from watermark_mask import fill_rgb  # noqa: E402

OUT = PKG / "outputs" / "ablation" / "mask_qa"
PANELS = REPO_ROOT / "external_imagery" / "dataset" / "mask_qa"
GREY = np.array([229, 227, 223])
N_AOI, SEED = 120, 20261008
EXT_CACHE = PKG / "cache" / "external_decoded_1280px"
BENCH_CACHE = PKG / "cache" / "decoded_1280px"


def src_uv(grid: int, crop: float, n: int = CACHE_PX):
    """Per cache index (one axis): source-image index and in-image coordinate of the pixel centre."""
    off = (grid * SOURCE_PX - crop) / 2
    s = off + (np.arange(n) + 0.5) * crop / n
    return (s // SOURCE_PX).astype(int), s % SOURCE_PX


def gray(arr):
    return arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def hp(arr):
    g = gray(arr)
    return np.abs(g - ndimage.gaussian_filter(g, aa.HP_SIGMA)) > aa.HP_T


def one_real(job):
    """Accumulate before/after high-pass frequency in the source frame (u, v binned to 4 px)."""
    path, grid, crop = job
    arr = np.load(path)
    m = geometry_mask(grid, crop)
    before, after = hp(arr), hp(fill_rgb(arr, m))
    _, u = src_uv(grid, crop)
    ub = (u // 4).astype(int)
    U, V = np.meshgrid(ub, ub)
    acc_b, acc_a, cnt = (np.zeros((320, 320)) for _ in range(3))
    np.add.at(acc_b, (V.ravel(), U.ravel()), before.ravel())
    np.add.at(acc_a, (V.ravel(), U.ravel()), after.ravel())
    np.add.at(cnt, (V.ravel(), U.ravel()), 1)
    # attribution left edge per visible source image: darkened bottom band (v 1252..1278) vs band above (1205..1231)
    ci, uu = src_uv(grid, crop)
    g = gray(arr)
    lefts = []
    for r in np.unique(ci):
        rb, rc = (ci == r) & (uu > 1252) & (uu < 1278), (ci == r) & (uu > 1205) & (uu < 1231)
        if not rb.any() or not rc.any():
            continue
        diff = g[rb].mean(0) - g[rc].mean(0)
        for c in np.unique(ci):
            sel = ci == c
            if uu[sel].max() < 1270 or uu[sel].min() > 200:
                continue
            bins = (uu[sel] // 20).astype(int)
            prof = np.array([diff[sel][bins == b].mean() if (bins == b).any() else 0.0 for b in range(64)])
            k = 63
            while k > 0 and prof[k - 1] < -12:
                k -= 1
            if k < 63:
                lefts.append(k * 20)
    return acc_b, acc_a, cnt, lefts


def template_src(margin: bool = True) -> np.ndarray:
    t = np.zeros((SOURCE_PX, SOURCE_PX), bool)
    mg = 3 if margin else 0
    for (u0, u1), (v0, v1) in RECTS.values():
        t[max(v0 - mg, 0):v1 + mg, max(u0 - mg, 0):u1 + mg] = True
    return t


def real_domain(jobs) -> dict:
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(one_real, jobs, chunksize=4))
    cnt = sum(r[2] for r in res)
    fb, fa = sum(r[0] for r in res) / np.maximum(cnt, 1), sum(r[1] for r in res) / np.maximum(cnt, 1)
    tmpl = template_src()[::4, ::4] | template_src()[3::4, 3::4]
    known = fb > 0.5
    bg = np.median(fb[~ndimage.binary_dilation(tmpl, iterations=4)])
    lefts = np.concatenate([r[3] for r in res]) if any(len(r[3]) for r in res) else np.array([])
    return {"n_tiles": len(jobs), "known_watermark_bins": int(known.sum()),
            "coverage_of_known": round(float((known & tmpl).sum() / max(known.sum(), 1)), 4),
            "known_bins_outside_template": int((known & ~tmpl).sum()),
            "before_max_freq_in_template": round(float(fb[tmpl].max()), 3),
            "after_max_freq_in_template": round(float(fa[tmpl].max()), 3),
            "after_max_freq_outside_template": round(float(fa[~tmpl].max()), 3),
            "after_bins_gt_0p5": int((fa > 0.5).sum()), "background_median_freq": round(float(bg), 3),
            "attribution_left_edge_est": {q: float(np.percentile(lefts, p)) for q, p in
                                          (("p1", 1), ("p5", 5), ("median", 50))} if len(lefts) else None,
            "_fb": fb, "_fa": fa}


def placeholder_qa() -> dict:
    d = REPO_ROOT / load_config()["dataset"]["dataset_dir"]
    m = pd.concat([pd.read_csv(f) for f in [d / "external_tiles_metadata_refined.csv"] +
                   sorted(glob.glob(str(d / "hard_negative_metadata_private_*.csv")))]).drop_duplicates("tile_id")
    ph = m[m.artifact_flags.fillna("").str.contains("full_placeholder")]
    rows = []
    for r in ph.itertuples():
        arr = decoded_cache.decode_reduce((REPO_ROOT / r.png_path).read_bytes())
        grid, crop = int(r.mosaic_grid), float(r.crop_px)
        mask = geometry_mask(grid, crop)
        non = np.abs(arr.astype(int) - GREY).max(-1) > 25
        ci, v = src_uv(grid, crop)
        sorry = (v > 560) & (v < 720)
        band = v > 1150
        ok = np.zeros_like(non)
        for a in np.unique(ci):                      # keep only source images that are blank apart from the overlay
            for b in np.unique(ci):
                blk = np.ix_(ci == a, ci == b)
                body = non[blk][~band[ci == a] & ~sorry[ci == a]]
                if body.mean() < 0.005:
                    ok[blk] = True
        known = non & ok & ~sorry[:, None]
        if known.sum():
            rows.append({"tile_id": r.tile_id, "layout": f"z{int(r.source_zoom)}/{grid}x{grid}", "known_px": int(known.sum()),
                         "covered_px": int((known & mask).sum()), "mask_px_in_blank_images": int((mask & ok).sum()),
                         "mask_px_on_overlay": int((mask & ok & non).sum())})
    df = pd.DataFrame(rows)
    by = df.groupby("layout")[["known_px", "covered_px", "mask_px_in_blank_images", "mask_px_on_overlay"]].sum()
    by["coverage"] = (by.covered_px / by.known_px).round(4)
    by["share_of_mask_on_overlay"] = (by.mask_px_on_overlay / by.mask_px_in_blank_images).round(3)
    df["cov"] = df.covered_px / df.known_px
    return {"n_tiles": len(df), "coverage_all": round(float(df.covered_px.sum() / df.known_px.sum()), 4),
            "min_tile_coverage": round(float(df["cov"].min()), 4),
            "uncovered_px_total": int((df.known_px - df.covered_px).sum()),
            "by_layout": by.reset_index().to_dict("records")}


def panel(arr, mask, title, path):
    lab, n = ndimage.label(mask)
    objs = ndimage.find_objects(lab)
    interior = [(s, (s[0].stop - s[0].start) * (s[1].stop - s[1].start)) for s in objs
                if s[0].start > 40 and s[0].stop < CACHE_PX - 40 and s[1].start > 60 and s[1].stop < CACHE_PX - 60]
    s = max(interior or [(objs[0], 0)], key=lambda x: x[1])[0]
    y0, y1 = max(s[0].start - 60, 0), min(s[0].stop + 40, CACHE_PX)
    x0, x1 = max(s[1].stop - 360, 0), min(s[1].stop + 60, CACHE_PX)
    filled = fill_rgb(arr, mask)
    ov = arr.copy()
    edge = mask & ~ndimage.binary_erosion(mask)
    ov[edge] = [255, 0, 255]
    crops = [Image.fromarray(a[y0:y1, x0:x1]).resize(((x1 - x0) * 2, (y1 - y0) * 2), Image.NEAREST) for a in (arr, ov, filled)]
    w, h = crops[0].size
    im = Image.new("RGB", (w, 3 * h + 60), "white")
    for i, c in enumerate(crops):
        im.paste(c, (0, 20 + i * (h + 20)))
    dr = ImageDraw.Draw(im)
    dr.text((4, 4), f"{title}: original | mask outline | filled (cache px {x0}:{x1}, {y0}:{y1}; 2x)", fill="black")
    im.save(path)


def main():
    bank = MaskBank()
    g = pd.read_csv(PKG / "cache" / "watermark_geometry" / "geometry.csv").set_index("key")
    OUT.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)
    summary = {"template_sha256": bank.template_sha256, "geometry_table_sha256": bank.table_sha256}
    summary["placeholder_ground_truth"] = placeholder_qa()

    ext = pd.read_csv(EXT_CACHE / "cache_manifest.csv")
    ext = ext.join(g, on="tile_id")
    ext["layout"] = "z" + ext.zoom.astype(str) + "/" + ext.grid.astype(str) + "x" + ext.grid.astype(str)
    domains = {}
    for lay, grp in ext.groupby("layout"):
        domains[f"external {lay}"] = [(str(EXT_CACHE / f"{t}.npy"), int(gr), float(cr)) for t, gr, cr in zip(grp.tile_id, grp.grid, grp.crop_px)]
    rng = np.random.default_rng(SEED)
    for reg in ("Saratov", "Natanz", "Fordow"):
        keys = g[g.domain == reg].index.to_numpy()
        keys = rng.choice(keys, min(N_AOI, len(keys)), replace=False)
        domains[reg] = [(str(BENCH_CACHE / f"{k}.npy"), int(g.grid[k]), float(g.crop_px[k])) for k in keys]
    res = {k: real_domain(v) for k, v in domains.items() if len(v) >= 3}
    summary["real_imagery"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}

    # over-masking: mask area beyond the actual overlay, using the pooled real-imagery attribution left-edge estimate
    lefts = [v["attribution_left_edge_est"]["median"] for v in res.values() if v["attribution_left_edge_est"]]
    left_med = float(np.median(lefts))
    tm = template_src()
    (_, _), (v0, v1) = RECTS["attribution"]
    actual = tm.copy()
    actual[v0 - 3:, 131:int(left_med) - 3] = False                                   # box narrower than the template
    actual[(RECTS["google_logo"][1][0] - 3):, :131] = tm[(RECTS["google_logo"][1][0] - 3):, :131]   # logo always needed
    summary["over_masking"] = {
        "template_area_pct_of_source_image": round(100 * tm.mean(), 3),
        "estimated_actual_overlay_area_pct_median_text": round(100 * actual.mean(), 3),
        "excess_area_pct_of_tile_median_text": round(100 * (tm.mean() - actual.mean()), 3),
        "attribution_left_edge_median_real": left_med,
        "attribution_left_edge_min_placeholder": 309,
        "per_tile_mask_area_pct": {k: [float(g[g.domain == k].mask_area_pct.min()), float(g[g.domain == k].mask_area_pct.max())]
                                   for k in g.domain.unique()},
        "old_fixed_mask_area_pct": 5.82}

    # visual panels
    picks = {"Saratov": None, "Natanz": None, "Fordow": None, "external z16/3x3": None, "external z17/3x3": None, "external z17/4x4": None}
    for name in picks:
        jobs = domains[name]
        best = max(jobs[:40], key=lambda j: gray(np.load(j[0])).mean())       # bright tile: overlay most visible
        arr = np.load(best[0])
        panel(arr, geometry_mask(best[1], best[2]), f"{name} grid {best[1]} crop {best[2]:.0f}",
              PANELS / f"{name.replace(' ', '_').replace('/', '_')}.png")
        f = res[name]["_fb"], res[name]["_fa"]
        Image.fromarray(np.hstack([(np.clip(x, 0, 1) * 255).astype(np.uint8) for x in f])).save(
            PANELS / f"freq_{name.replace(' ', '_').replace('/', '_')}.png")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
