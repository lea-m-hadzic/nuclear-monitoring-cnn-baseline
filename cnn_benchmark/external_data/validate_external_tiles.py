"""Check extracted tiles against the benchmark image requirements. Makes NO API calls.

A "scan dir" is any vendor output directory (tiles/*.png + manifest.csv). By default every scan dir under
<output_root>/<preset>/ is checked; --scan-dirs checks explicit directories (e.g. the existing benchmark folders,
with --benchmark to skip our own metadata files).

Per tile: PNG decodes fully, 2560x2560, RGB, benchmark filename pattern, manifest row present with an ok status,
footprint ~1.9 km (from the manifest geometry, else recomputed from latitude), zoom/grid as the preset predicts,
and for external points the tile centre equals the intake coordinate. Across tiles: unique tile ids and no
coordinate extracted twice. Per dir: point.json + extraction_record.json present, and no API key in any text file.

    python validate_external_tiles.py --preset auto_mosaic_1p9km
    python validate_external_tiles.py --benchmark --scan-dirs ../../data/fordow_30km_tiles_auto_mosaic --sample 20
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from geometry import REPO_ROOT, load_config, tile_geometry
from secrets_guard import scan_dir_for_secrets

NAME_RE = re.compile(r"^region_r(\d+)_c(\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_tile([\d.]+)km(\(\d+\))?\.png$")  # = image_index.py
OK_STATUS = {"downloaded", "cached"}
METRES_PER_DEG = 111_195.0


def manifest_footprint_km(m: pd.Series) -> float | None:
    """Ground width of the saved PNG according to the vendor manifest row, if it recorded the geometry."""
    def f(c):
        try:
            v = float(m.get(c, ""))
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None
    mode, msg = str(m.get("mode", "")), str(m.get("message", ""))
    if mode == "auto_mosaic" and f("crop_px") and f("native_mpp"):
        return f("crop_px") * f("native_mpp") / 1000.0          # centre crop, then resized
    if mode == "legacy_mosaic2x2" or "mosaic2x2" in msg:
        return f("mosaic_width_km") or f("actual_returned_width_km")
    return None


def infer_preset(manifest: pd.DataFrame) -> str | None:
    modes = set(manifest.get("mode", pd.Series(dtype=str)).dropna())
    if modes == {"auto_mosaic"}:
        return "auto_mosaic_1p9km"
    if modes == {"legacy_mosaic2x2"} or manifest.get("message", pd.Series(dtype=str)).str.contains("mosaic2x2").all():
        return "legacy_mosaic2x2_z16"
    return None


def check_scan_dir(d: Path, preset: str | None, benchmark: bool, sample: int | None, tol: float) -> tuple[list[dict], list[str]]:
    cfg = load_config()
    tiles, problems = [], []
    pngs = sorted((d / "tiles").glob("*.png"))
    if not pngs:
        problems.append("no tiles/*.png")
    manifest = pd.read_csv(d / "manifest.csv", dtype=str) if (d / "manifest.csv").exists() else None
    if manifest is None:
        problems.append("manifest.csv missing")
    by_name = {} if manifest is None else {Path(p).name: r for p, r in zip(manifest["tile_path"], manifest.to_dict("records"))}
    preset = preset or (infer_preset(manifest) if manifest is not None else None)
    point = None
    if not benchmark:
        for f in ("point.json", "extraction_record.json"):
            if not (d / f).exists():
                problems.append(f"{f} missing")
        if (d / "point.json").exists():
            point = json.loads((d / "point.json").read_text())
    leaks = scan_dir_for_secrets(d)
    if leaks:
        problems.append(f"API key present in {len(leaks)} file(s): {[p.name for p in leaks]}")

    for png in (pngs[:sample] if sample else pngs):
        t = {"scan_dir": str(d), "filename": png.name, "errors": []}
        m = NAME_RE.match(png.name)
        if not m:
            t["errors"].append("filename does not match benchmark pattern")
        else:
            t["lat"], t["lon"] = float(m.group(3)), float(m.group(4))
        try:
            with Image.open(png) as im:
                im.verify()
            with Image.open(png) as im:
                im.load()
                t["size"], t["mode"] = im.size, im.mode
            if t["size"] != (cfg["output_px"], cfg["output_px"]):
                t["errors"].append(f"size {t['size']} != {cfg['output_px']}x{cfg['output_px']}")
            if t["mode"] != "RGB":
                t["errors"].append(f"mode {t['mode']} != RGB")
        except Exception as e:
            t["errors"].append(f"corrupt PNG: {type(e).__name__}: {e}")
        row = by_name.get(png.name)
        if manifest is not None and row is None:
            t["errors"].append("no manifest row")
        elif row is not None and row.get("status") not in OK_STATUS:
            t["errors"].append(f"manifest status {row.get('status')}")
        if "lat" in t and preset:
            g = tile_geometry(preset, t["lat"])
            fp = manifest_footprint_km(pd.Series(row)) if row is not None else None
            t["footprint_width_km"] = fp if fp is not None else g.footprint_width_km
            t["footprint_source"] = "manifest" if fp is not None else "computed"
            if abs(t["footprint_width_km"] - cfg["tile_km"]) / cfg["tile_km"] > tol:
                t["errors"].append(f"footprint {t['footprint_width_km']:.3f} km != {cfg['tile_km']} km +/-{100 * tol:.0f}%")
            if row is not None and str(row.get("zoom", "")).strip():
                if int(float(row["zoom"])) != g.zoom:
                    t["errors"].append(f"zoom {row['zoom']} != expected {g.zoom}")
                gn = str(row.get("mosaic_grid_n", "")).strip()
                if gn and int(float(gn)) != g.grid_n:
                    t["errors"].append(f"grid {gn} != expected {g.grid_n}")
        if point is not None and "lat" in t and len(pngs) == 1:
            off_m = math.hypot((t["lat"] - point["latitude"]) * METRES_PER_DEG,
                               (t["lon"] - point["longitude"]) * METRES_PER_DEG * math.cos(math.radians(t["lat"])))
            t["centre_offset_m"] = round(off_m, 2)
            if off_m > 1.0:
                t["errors"].append(f"tile centre {off_m:.1f} m from the intake coordinate")
        tiles.append(t)
    return tiles, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=sorted(load_config()["presets"]), help="expected preset (else inferred)")
    ap.add_argument("--scan-dirs", nargs="+", type=Path, help="explicit vendor output directories")
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--benchmark", action="store_true", help="existing benchmark folders: no point.json/record")
    ap.add_argument("--sample", type=int, default=None, help="check only the first N tiles per dir (speed)")
    ap.add_argument("--tolerance", type=float, default=0.03, help="relative footprint tolerance (default 3%%)")
    ap.add_argument("--report", type=Path, default=None, help="per-tile CSV (holds coordinates: keep it git-ignored)")
    args = ap.parse_args()

    cfg = load_config()
    if args.scan_dirs:
        dirs = args.scan_dirs
    else:
        if not args.preset:
            raise SystemExit("--preset is required unless --scan-dirs is given")
        base = (args.output_root or REPO_ROOT / cfg["output_root"]) / args.preset
        dirs = sorted(p.parent for p in base.rglob("manifest.csv"))
        if not dirs:
            raise SystemExit(f"no scan dirs under {base}")

    all_tiles, n_dir_problems = [], 0
    for d in dirs:
        tiles, problems = check_scan_dir(d, args.preset, args.benchmark, args.sample, args.tolerance)
        all_tiles += tiles
        bad = sum(bool(t["errors"]) for t in tiles)
        n_dir_problems += bool(problems)
        print(f"{d}: {len(tiles)} tiles checked, {bad} with errors" + (f"; DIR PROBLEMS: {problems}" if problems else ""))
        for t in [t for t in tiles if t["errors"]][:5]:
            print(f"   {t['filename']}: {t['errors']}")

    df = pd.DataFrame(all_tiles)
    if df.empty:
        raise SystemExit("no tiles checked")
    df["tile_id"] = df["scan_dir"] + "/" + df["filename"]
    dup_ids = int(df["tile_id"].duplicated().sum())
    dup_coords = int(df.dropna(subset=["lat"]).duplicated(subset=["lat", "lon"]).sum()) if "lat" in df else 0
    n_err = int(df["errors"].map(bool).sum())
    if "footprint_width_km" in df:
        print(f"footprint km: min {df['footprint_width_km'].min():.3f}, max {df['footprint_width_km'].max():.3f} "
              f"(sources: {df['footprint_source'].value_counts().to_dict()})")
    print(f"TOTAL: {len(df)} tiles, {n_err} with errors, {dup_ids} duplicate tile ids, "
          f"{dup_coords} coordinates extracted more than once, {n_dir_problems} dirs with problems")
    if args.report:
        df.assign(errors=df["errors"].map("; ".join)).to_csv(args.report, index=False)
    raise SystemExit(1 if (n_err or dup_ids or dup_coords or n_dir_problems) else 0)


if __name__ == "__main__":
    main()
