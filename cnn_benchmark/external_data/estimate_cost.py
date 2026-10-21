"""Estimate output tiles, Static Maps requests and cost for an external extraction. Makes NO API calls.

Requests per tile depend on the preset and, for auto_mosaic, on latitude (see geometry.py). Pricing comes from
extraction_config.yaml (`pricing`, with the date it was checked); override with --usd-per-1000 for a flat rate.

    python estimate_cost.py --n-coords 50 --radius-km 0 --lat 35
    python estimate_cost.py --csv external_facilities.csv --radius-km 2 --preset auto_mosaic_1p9km
"""
from __future__ import annotations

import argparse

import pandas as pd

from geometry import load_config, n_tiles, tile_geometry, tiered_cost_usd

RADIUS_CENTRE_ONLY = 0.001     # the vendor requires radius > 0; anything below tile_km yields the centre tile only


def estimate(lats: list[float], radius_km: float, preset: str, tile_km: float = 1.9,
             already_used: int = 0, usd_per_1000: float | None = None) -> dict:
    tiles_per_coord = n_tiles(max(radius_km, RADIUS_CENTRE_ONLY), tile_km)
    geoms = [tile_geometry(preset, lat) for lat in lats]
    n_req = sum(g.requests_per_tile for g in geoms) * tiles_per_coord
    pricing = load_config()["pricing"]
    cost = n_req * usd_per_1000 / 1000 if usd_per_1000 is not None else tiered_cost_usd(n_req, pricing, already_used)
    return {
        "preset": preset, "n_coords": len(lats), "radius_km": radius_km, "tiles_per_coord": tiles_per_coord,
        "output_tiles": tiles_per_coord * len(lats), "requests": n_req,
        "requests_per_tile": sorted({g.requests_per_tile for g in geoms}),
        "footprint_width_km": sorted({round(g.footprint_width_km, 3) for g in geoms}),
        "approx_cost_usd": round(cost, 2),
        "pricing_basis": f"flat ${usd_per_1000}/1000" if usd_per_1000 is not None else
                         f"tiered, free cap {pricing['free_requests_per_month']}/month, "
                         f"{already_used} already used; checked {pricing['checked_on']}",
        "approx_png_gb": round(tiles_per_coord * len(lats) * 13e6 / 1e9, 2),   # benchmark PNGs: ~9 (Saratov)-13 (Fordow) MB
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--n-coords", type=int, help="number of coordinates (use with --lat)")
    src.add_argument("--csv", help="coordinate CSV with a latitude column (per-row latitudes are used)")
    ap.add_argument("--lat", type=float, default=35.0, help="representative latitude for --n-coords (default 35)")
    ap.add_argument("--radius-km", type=float, default=0.0, help="0 = one tile centred on each coordinate")
    ap.add_argument("--tile-km", type=float, default=1.9)
    ap.add_argument("--preset", choices=sorted(load_config()["presets"]), help="default: show all presets")
    ap.add_argument("--already-used", type=int, default=0, help="Static Maps requests already billed this month")
    ap.add_argument("--usd-per-1000", type=float, default=None, help="flat price override")
    args = ap.parse_args()

    lats = pd.read_csv(args.csv)["latitude"].astype(float).tolist() if args.csv else [args.lat] * args.n_coords
    presets = [args.preset] if args.preset else sorted(load_config()["presets"])
    rows = [estimate(lats, args.radius_km, p, args.tile_km, args.already_used, args.usd_per_1000) for p in presets]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
