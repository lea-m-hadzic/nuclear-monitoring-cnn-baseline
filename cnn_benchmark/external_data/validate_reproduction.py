"""Re-extract ONE known benchmark tile with the wrapper and compare it with the original benchmark PNG.

Default: Saratov parent r3_c20 with the legacy 2x2 z16 preset (Qiqi's stated command; 5 Static Maps requests).
Google imagery may have changed since the original scan, so pixel equality is NOT required; the goal is to confirm
geometry and format. Compared: size/mode, filename structure, manifest columns, footprint, alignment (phase
correlation), sharpness, watermark/attribution corners, pixel similarity.

Outputs go to cnn_benchmark/external_data/_validation/<label>/ (git-ignored): the vendor output, a JSON + markdown
report and a side-by-side figure.

    python validate_reproduction.py                                   # fetch (needs the API key) + compare
    python validate_reproduction.py --compare-only path/to/new.png    # compare an existing PNG; no API call
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from extract_external_tiles import api_key_or_exit, check_vendor, extract_point
from geometry import REPO_ROOT, load_config, tile_geometry
from validate_external_tiles import NAME_RE

HERE = Path(__file__).resolve().parent
VALIDATION_DIR = HERE / "_validation"
TILES = {   # benchmark tiles we hold at full resolution, with the preset whose manifest mode they were made with
    "saratov_r003_c020": ("region_r003_c020_51.844655_45.945079_tile1.90km.png", "legacy_mosaic2x2_z16"),
    "fordow_r008_c012": ("region_r008_c012_35.016697_50.916684_tile1.90km.png", "auto_mosaic_1p9km"),
}


def find_benchmark(filename: str) -> tuple[Path, dict]:
    pngs = sorted(glob.glob(str(REPO_ROOT / "data" / "*" / "tiles" / filename)))
    if not pngs:
        raise SystemExit(f"benchmark PNG {filename} not found under data/*/tiles/")
    for man in sorted(glob.glob(str(REPO_ROOT / "data" / "*" / "manifest.csv"))):
        m = pd.read_csv(man, dtype=str).drop(columns=["url"], errors="ignore")   # url holds a key: never loaded
        hit = m[m["tile_path"].map(lambda p: Path(p).name) == filename]
        if len(hit):
            return Path(pngs[0]), hit.iloc[0].to_dict() | {"_manifest_columns": list(m.columns)}
    raise SystemExit(f"no manifest row for {filename}")


def gray(a: np.ndarray) -> np.ndarray:
    return a[..., :3].astype(np.float64) @ [0.299, 0.587, 0.114]


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(dy, dx, peak) such that b ~ a shifted by (dy, dx), via normalised cross-power spectrum."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2((a - a.mean()) * win), np.fft.fft2((b - b.mean()) * win)
    r = fb * np.conj(fa)
    corr = np.fft.ifft2(r / (np.abs(r) + 1e-12)).real
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    dy = iy - corr.shape[0] if iy > corr.shape[0] // 2 else iy
    dx = ix - corr.shape[1] if ix > corr.shape[1] // 2 else ix
    return float(dy), float(dx), float(corr.max())


def laplacian_var(g: np.ndarray) -> float:
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def corners(a: np.ndarray, sub: int) -> dict[str, np.ndarray]:
    """Bottom-left (logo) and bottom-right (attribution) strips of every Static Maps sub-image in the mosaic."""
    out = {}
    n = a.shape[0] // sub
    for r in range(n):
        for c in range(n):
            y1, x0 = (r + 1) * sub, c * sub
            out[f"q{r}{c}_logo"] = a[y1 - 70:y1, x0:x0 + 200]
            out[f"q{r}{c}_attrib"] = a[y1 - 40:y1, x0 + sub - 520:x0 + sub]
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = gray(a).ravel(), gray(b).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare(old_png: Path, new_png: Path, old_row: dict, new_manifest: Path | None, preset: str,
            out_dir: Path) -> dict:
    old = np.asarray(Image.open(old_png).convert("RGB"))
    new_im = Image.open(new_png)
    res = {"old": str(old_png.relative_to(REPO_ROOT)), "new": str(new_png), "preset": preset,
           "old_size": list(Image.open(old_png).size), "new_size": list(new_im.size),
           "old_mode": Image.open(old_png).mode, "new_mode": new_im.mode}
    new = np.asarray(new_im.convert("RGB"))
    lat = float(old_png.name.split("_")[3])
    g = tile_geometry(preset, lat)
    res["expected"] = {"zoom": g.zoom, "grid_n": g.grid_n, "footprint_width_km": round(g.footprint_width_km, 4),
                       "effective_mpp": round(g.effective_mpp, 5), "requests_per_tile": g.requests_per_tile}
    mo, mn = NAME_RE.match(old_png.name), NAME_RE.match(new_png.name)
    res["filename"] = {"old": old_png.name, "new": new_png.name, "new_matches_pattern": bool(mn),
                       # row/col index the scan grid (r001_c001 for a single centred tile); centre + size must agree
                       "same_after_row_col": bool(mo and mn and mo.groups()[2:5] == mn.groups()[2:5])}
    if new_manifest is not None and new_manifest.exists():
        nm = pd.read_csv(new_manifest, dtype=str).drop(columns=["url"], errors="ignore").iloc[0].to_dict()
        old_cols = set(old_row["_manifest_columns"]) | {"url"}
        new_cols = set(pd.read_csv(new_manifest, nrows=0).columns)
        res["manifest"] = {"new_row": nm, "columns_only_in_new": sorted(new_cols - old_cols),
                           "columns_only_in_old": sorted(old_cols - new_cols)}
        for k in ("zoom", "returned_m_per_px", "actual_returned_width_km", "mosaic_width_km", "crop_px",
                  "tile_image_px", "mosaic_grid_n", "native_mpp"):
            res["manifest"].setdefault("old_vs_new", {})[k] = (old_row.get(k), nm.get(k))
    if old.shape != new.shape:
        res["verdict"] = "FAIL: shapes differ"
        return res

    go, gn = gray(old), gray(new)
    # alignment: whole tile at 1/4 resolution (robust to imagery change), then the central 1024 px at full resolution
    s = 4
    dy4, dx4, pk4 = phase_shift(go[::s, ::s], gn[::s, ::s])
    c0 = (old.shape[0] - 1024) // 2
    dy, dx, pk = phase_shift(go[c0:c0 + 1024, c0:c0 + 1024], gn[c0:c0 + 1024, c0:c0 + 1024])
    res["alignment"] = {"coarse_shift_px": [dy4 * s, dx4 * s], "coarse_peak": round(pk4, 4),
                        "centre_shift_px": [dy, dx], "centre_peak": round(pk, 4),
                        "centre_shift_m": round(math.hypot(dy, dx) * g.effective_mpp, 2)}
    res["similarity"] = {"pearson_gray": round(corr(old, new), 4),
                         "mean_abs_diff_0_255": round(float(np.abs(old.astype(int) - new.astype(int)).mean()), 2),
                         "identical_pixels_fraction": round(float((old == new).all(-1).mean()), 4)}
    res["sharpness"] = {"laplacian_var_old": round(laplacian_var(go), 1), "laplacian_var_new": round(laplacian_var(gn), 1),
                        "ratio_new_over_old": round(laplacian_var(gn) / laplacian_var(go), 3)}
    sub = load_config()["single_image_px"] if preset.startswith("legacy") else None
    if sub:
        co, cn = corners(old, sub), corners(new, sub)
        res["watermark_corners_pearson"] = {k: round(corr(co[k], cn[k]), 3) for k in co}

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(old); ax[0, 0].set_title(f"benchmark {old_png.name[:22]}…", fontsize=9)
    ax[0, 1].imshow(new); ax[0, 1].set_title("re-extracted (wrapper)", fontsize=9)
    ax[0, 2].imshow(np.abs(go - gn), cmap="magma"); ax[0, 2].set_title("|gray diff|", fontsize=9)
    ax[1, 0].imshow(old[c0:c0 + 1024, c0:c0 + 1024]); ax[1, 0].set_title("benchmark centre 1024 px", fontsize=9)
    ax[1, 1].imshow(new[c0:c0 + 1024, c0:c0 + 1024]); ax[1, 1].set_title("re-extracted centre 1024 px", fontsize=9)
    h = old.shape[0]
    ax[1, 2].imshow(np.vstack([old[h - 70:h, :640], new[h - 70:h, :640]]))
    ax[1, 2].set_title("bottom-left strip: benchmark (top) / new (bottom)", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{preset}: shift {res['alignment']['centre_shift_px']} px, r={res['similarity']['pearson_gray']}, "
                 f"sharpness ratio {res['sharpness']['ratio_new_over_old']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "side_by_side.png", dpi=80)
    plt.close(fig)
    geometry_ok = (res["new_size"] == res["old_size"] == [2560, 2560] and res["new_mode"] == "RGB"
                   and res["filename"]["same_after_row_col"] and math.hypot(dy4 * s, dx4 * s) <= 8)
    res["verdict"] = "PASS (geometry/format)" if geometry_ok else "CHECK: see alignment/size fields"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", choices=sorted(TILES), default="saratov_r003_c020")
    ap.add_argument("--preset", default=None, help="override the tile's benchmark preset")
    ap.add_argument("--compare-only", type=Path, default=None, help="compare this PNG; no extraction")
    args = ap.parse_args()

    filename, preset = TILES[args.tile]
    preset = args.preset or preset
    old_png, old_row = find_benchmark(filename)
    out_dir = VALIDATION_DIR / f"{args.tile}__{preset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_only:
        new_png, new_manifest = args.compare_only, None
    else:
        cfg = load_config()
        key = api_key_or_exit(cfg)
        vendor = check_vendor(cfg, allow_mismatch=False)
        point = {"point_id": args.tile, "facility_id": "validation", "facility_name": f"benchmark tile {filename}",
                 "country": "", "latitude": float(old_row["center_lat"]), "longitude": float(old_row["center_lon"]),
                 "coordinate_type": "other", "label_source": "benchmark manifest centre", "notes": "reproduction check"}
        rec = extract_point(point, preset, 0.0, out_dir / "extract", key, vendor)
        print(f"extraction: {rec['status']} ({rec['png_tiles']} tile, manifest {rec['manifest_status_counts']})")
        pngs = sorted((out_dir / "extract" / "tiles").glob("*.png"))
        if not pngs:
            raise SystemExit(f"no tile produced; see {out_dir / 'extract' / 'run.log'}")
        new_png, new_manifest = pngs[0], out_dir / "extract" / "manifest.csv"

    res = compare(old_png, new_png, old_row, new_manifest, preset, out_dir)
    (out_dir / "comparison.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "manifest"}, indent=1, default=str))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
