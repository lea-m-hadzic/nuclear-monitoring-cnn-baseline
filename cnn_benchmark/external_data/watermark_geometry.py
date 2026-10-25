"""Per-tile geometric watermark mask: the Google overlay template placed on every SOURCE image of each tile's mosaic.

Every Static Maps request (size 640, scale 2 -> 1280 x 1280 px) carries the same overlay at fixed positions in its
own pixel frame: the "Google" logo bottom-left and a right-aligned attribution box along the bottom edge (its left
edge depends on the provider text, so the template uses the widest one observed). Source-frame template measured on
the 70 grey "no imagery" placeholder tiles (overlay = non-grey pixels, all four layouts):
  google_logo   u 0..127,    v 1235..1279
  attribution   u 300..1279, v 1249..1279     (widest box observed: left edge u = 309; real-imagery median ~850)
plus MARGIN_PX source pixels on every side (resampling / JPEG halo, may cross into neighbouring source images).

A tile's mosaic is grid x grid source images (1280 px each), centre-cropped to crop_px and resized to 2560 px;
the decoded cache is a 2x box reduction to 1280 px. Cache pixel x covers mosaic interval
[off + x*crop/1280, off + (x+1)*crop/1280), off = (grid*1280 - crop)/2; it is masked if that interval (on both axes)
overlaps a template rectangle of any source image. Geometry per tile:
  external tiles            auto_mosaic, grid / crop_px from the extraction metadata
  Saratov (benchmark)       legacy --mosaic2x2 z16: grid 2, crop 2560, off 0
  Natanz / Fordow           auto_mosaic: geometry.tile_geometry('auto_mosaic_1p9km', centre lat) (vendor code);
                            identical to the extraction manifests' zoom / grid / crop_px on all rows that carry them
The same template, rule and fill (watermark_mask.fill_rgb) apply to external and benchmark tiles.

    python watermark_geometry.py build    # cache/watermark_geometry/{geometry.csv, template.json}
"""
from __future__ import annotations

import functools
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "cache" / "watermark_geometry"                 # git-ignored (cache/)
SOURCE_PX, CACHE_PX, MARGIN_PX = 1280, 1280, 3
RECTS = {"google_logo": ((0, 128), (1235, 1280)), "attribution": ((300, 1280), (1249, 1280))}   # (u0,u1),(v0,v1) half-open
LEGACY = {"layout": "legacy_mosaic2x2", "zoom": 16, "grid": 2, "crop_px": 2560}
BENCHMARK_REGIONS = ["Saratov", "Natanz", "Fordow"]


def template() -> dict:
    t = {"source_px": SOURCE_PX, "cache_px": CACHE_PX, "margin_px": MARGIN_PX,
         "rects_uv_half_open": {k: {"u": list(u), "v": list(v)} for k, (u, v) in RECTS.items()}}
    t["sha256"] = hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()
    return t


def _hits(grid: int, crop: float, interval: tuple[int, int], n: int) -> np.ndarray:
    """Cache pixels (one axis) whose mosaic interval overlaps [a, b) (+margin) in any source image."""
    off = (grid * SOURCE_PX - crop) / 2
    lo = off + np.arange(n) * crop / n
    hi = lo + crop / n
    a, b = interval[0] - MARGIN_PX, interval[1] + MARGIN_PX
    hit = np.zeros(n, bool)
    for k in range(grid):
        hit |= (lo < k * SOURCE_PX + b) & (hi > k * SOURCE_PX + a)
    return hit


@functools.lru_cache(maxsize=64)
def geometry_mask(grid: int, crop_px: float, n: int = CACHE_PX) -> np.ndarray:
    m = np.zeros((n, n), bool)
    for u, v in RECTS.values():
        m |= _hits(grid, crop_px, v, n)[:, None] & _hits(grid, crop_px, u, n)[None, :]
    m.flags.writeable = False
    return m


def mask_sha(m: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(m).tobytes()).hexdigest()


def external_geometry() -> pd.DataFrame:
    from geometry import REPO_ROOT, load_config
    d = REPO_ROOT / load_config()["dataset"]["dataset_dir"]
    files = [d / "external_tiles_metadata_refined.csv"] + sorted(Path(p) for p in glob.glob(str(d / "hard_negative_metadata_private_*.csv")))
    m = pd.concat([pd.read_csv(f, usecols=["tile_id", "source_zoom", "mosaic_grid", "crop_px"]) for f in files]).drop_duplicates("tile_id")
    return pd.DataFrame({"key": m.tile_id, "domain": "external", "layout": "auto_mosaic", "zoom": m.source_zoom.astype(int),
                         "grid": m.mosaic_grid.astype(int), "crop_px": m.crop_px.astype(float)})


def benchmark_geometry() -> pd.DataFrame:
    from geometry import REPO_ROOT, tile_geometry
    pt = pd.read_csv(PKG / "outputs" / "parent_tiles.csv")
    pt = pt[pt.region.isin(BENCHMARK_REGIONS)]
    rows = []
    for r in pt.itertuples():
        if r.region == "Saratov":
            g = dict(LEGACY)
        else:
            t = tile_geometry("auto_mosaic_1p9km", r.center_lat)
            g = {"layout": "auto_mosaic", "zoom": t.zoom, "grid": t.grid_n, "crop_px": float(round(1900 / t.native_mpp))}
        rows.append({"key": f"{r.region}__{r.tile_id}", "domain": r.region, **g, "center_lat": r.center_lat})
    b = pd.DataFrame(rows)
    # cross-check against every extraction manifest row that records the geometry (the url column is never read)
    man = []
    for f in glob.glob(str(REPO_ROOT / "data" / "*" / "manifest.csv")):
        x = pd.read_csv(f, usecols=lambda c: c != "url")
        if "mosaic_grid_n" in x:
            man.append(x[["center_lat", "zoom", "mosaic_grid_n", "crop_px"]].dropna())
    man = pd.concat(man)
    pred = [tile_geometry("auto_mosaic_1p9km", la) for la in man.center_lat]
    ok = (np.array([p.grid_n for p in pred]) == man.mosaic_grid_n.to_numpy()).all() and \
         (np.array([p.zoom for p in pred]) == man.zoom.to_numpy()).all() and \
         (np.array([round(1900 / p.native_mpp) for p in pred]) == man.crop_px.to_numpy()).all()
    if not ok:
        raise SystemExit("latitude-derived auto_mosaic geometry disagrees with the extraction manifests")
    b.attrs["manifest_rows_checked"] = len(man)
    return b.drop(columns="center_lat")


def build() -> dict:
    ext, bench = external_geometry(), benchmark_geometry()
    g = pd.concat([ext, bench], ignore_index=True)
    assert not g.key.duplicated().any()
    g["off_px"] = (g.grid * SOURCE_PX - g.crop_px) / 2
    masks = {k: geometry_mask(int(k[0]), float(k[1])) for k in set(zip(g.grid, g.crop_px))}
    g["mask_area_pct"] = [round(100 * float(masks[k].mean()), 3) for k in zip(g.grid, g.crop_px)]
    g["mask_sha256"] = [mask_sha(masks[k]) for k in zip(g.grid, g.crop_px)]
    OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT / "geometry.csv", index=False)
    t = template()
    info = {"template": t, "geometry_csv_sha256": hashlib.sha256((OUT / "geometry.csv").read_bytes()).hexdigest(),
            "n_tiles": len(g), "manifest_rows_checked": bench.attrs["manifest_rows_checked"],
            "by_domain": g.groupby(["domain", "layout", "zoom", "grid"]).agg(
                n=("key", "size"), crop_min=("crop_px", "min"), crop_max=("crop_px", "max"),
                area_pct_min=("mask_area_pct", "min"), area_pct_max=("mask_area_pct", "max")).reset_index().to_dict("records")}
    (OUT / "template.json").write_text(json.dumps(info, indent=1, default=float))
    return info


class MaskBank:
    """tile key -> geometric mask (read from the built geometry table; masks rendered lazily and memoised)."""

    def __init__(self, path: Path = OUT / "geometry.csv", n: int = CACHE_PX):
        """`n` = rendered mask size; CACHE_PX (1280) for every frozen run. n=2560 renders the same rule (same physical
        area) for the full-resolution sensitivity only."""
        self.n = n
        g = pd.read_csv(path)
        self.geo = dict(zip(g.key, zip(g.grid.astype(int), g.crop_px.astype(float))))
        self.sha = dict(zip(g.key, g.mask_sha256))
        self.table_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.template_sha256 = template()["sha256"]

    def get(self, key: str) -> np.ndarray:
        if key not in self.geo:
            raise KeyError(f"no mask geometry for tile {key!r}")
        return geometry_mask(*self.geo[key], getattr(self, "n", CACHE_PX))   # subclasses without __init__ -> 1280


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    if sys.argv[1:] == ["build"]:
        print(json.dumps(build(), indent=1, default=float))
    else:
        raise SystemExit(__doc__)
