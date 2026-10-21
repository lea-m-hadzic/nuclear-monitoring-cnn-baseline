"""Tile geometry and Static Maps request counts for Qiqi's extraction script, computed WITHOUT any API call.

Zoom / grid / tile-centre logic is imported from the pinned vendor script itself, so these numbers follow the
exact code that will run. The request counts below come from reading that code (commit in extraction_config.yaml):

  legacy --mosaic2x2:  fetch_mosaic_grid_tile() first fetches a PROBE image at the tile centre (to learn the
                       returned pixel size, then discards it), then 2x2 = 4 sub-images  -> 5 requests per tile.
  --auto_mosaic:       build_auto_mosaic_plan() fetches one probe, then fetch_mosaic_grid_tile() fetches another
                       probe plus grid_n x grid_n sub-images                              -> 2 + grid_n**2.
  cached tile (PNG already on disk, no --overwrite): 0 requests.
  NOTE: the vendor --dry_run flag still performs every fetch (it only skips saving). The wrapper never uses it.
"""
from __future__ import annotations

import functools
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "extraction_config.yaml"


@functools.lru_cache(maxsize=None)
def load_config(path: str | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def vendor_script_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return REPO_ROOT / cfg["vendor"]["path"] / cfg["vendor"]["script"]


@functools.lru_cache(maxsize=None)
def vendor():
    """Import the vendor script as a module (its top level only defines functions; main() is not run)."""
    path = vendor_script_path()
    if not path.exists():
        cfg = load_config()
        raise SystemExit(f"Vendor script not found: {path}\nClone it with:\n  git clone {cfg['vendor']['repo_url']} "
                         f"{cfg['vendor']['path']} && git -C {cfg['vendor']['path']} checkout {cfg['vendor']['commit']}")
    spec = importlib.util.spec_from_file_location("qiqi_scan_tiles", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qiqi_scan_tiles"] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class TileGeometry:
    preset: str
    lat: float
    zoom: int
    grid_n: int
    requests_per_tile: int
    native_mpp: float          # metres per returned Static Maps pixel
    footprint_width_km: float        # ground width of the saved PNG
    output_px: int
    effective_mpp: float       # footprint / output_px


def preset_flags(preset: str) -> list[str]:
    presets = load_config()["presets"]
    if preset not in presets:
        raise SystemExit(f"unknown preset {preset!r}; choose one of {sorted(presets)}")
    return [str(x) for x in presets[preset]["flags"]]


def _flag_value(flags: list[str], name: str, default=None):
    return flags[flags.index(name) + 1] if name in flags else default


def tile_geometry(preset: str, lat: float) -> TileGeometry:
    v, cfg = vendor(), load_config()
    flags = preset_flags(preset)
    scale = int(_flag_value(flags, "--scale", 2))
    tile_km = float(_flag_value(flags, "--tile_km", cfg["tile_km"]))
    overlap_m = float(_flag_value(flags, "--mosaic_overlap_m", 0.0))
    single_px = cfg["single_image_px"]

    if "--mosaic2x2" in flags:
        zoom = int(_flag_value(flags, "--mosaic_zoom", 16))
        mpp = v.returned_m_per_px(lat, zoom, scale=scale)
        spacing = max(1.0, single_px * mpp - overlap_m)
        footprint_width_km = (single_px * mpp + spacing) / 1000.0
        out_px = 2 * single_px
        return TileGeometry(preset, lat, zoom, 2, 1 + 4, mpp, footprint_width_km, out_px, footprint_width_km * 1000 / out_px)

    if "--auto_mosaic" in flags:
        out_px = int(_flag_value(flags, "--output_px", 2560))
        target_mpp = tile_km * 1000.0 / out_px
        zoom = v.choose_zoom_for_target_mpp(lat_deg=lat, target_mpp=target_mpp, scale=scale)
        mpp = v.returned_m_per_px(lat, zoom, scale=scale)
        grid_n, _ = v.choose_grid_n_for_crop(target_width_m=tile_km * 1000.0, single_width_m=single_px * mpp,
                                             overlap_m=overlap_m)
        while int(round(tile_km * 1000.0 / mpp)) > single_px * grid_n:   # same pixel-coverage guard as the vendor
            grid_n += 1
        return TileGeometry(preset, lat, zoom, grid_n, 2 + grid_n ** 2, mpp, tile_km, out_px, target_mpp)

    raise SystemExit(f"preset {preset!r} is neither --mosaic2x2 nor --auto_mosaic")


def n_tiles(radius_km: float, tile_km: float) -> int:
    """Tiles the vendor scan produces for one coordinate (its own centre generator; centre-only if radius < tile)."""
    return len(vendor().generate_tile_centers(lat0=0.0, lon0=0.0, radius_km=radius_km, tile_km=tile_km))


def tiered_cost_usd(n_requests: int, pricing: dict, already_used_this_month: int = 0) -> float:
    """Cost of `n_requests` given the monthly free cap and volume tiers (cumulative within the month)."""
    start, end = already_used_this_month, already_used_this_month + n_requests
    bounds = [(0, pricing["free_requests_per_month"], 0.0)]
    lo = pricing["free_requests_per_month"]
    for hi, price in pricing["tiers_usd_per_1000"]:
        hi = math.inf if hi is None else hi
        bounds.append((lo, hi, price))
        lo = hi
    return sum(max(0, min(end, hi) - max(start, lo)) * price / 1000.0 for lo, hi, price in bounds)
