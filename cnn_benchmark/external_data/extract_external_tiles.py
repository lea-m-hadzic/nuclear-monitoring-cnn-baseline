"""Extract external layered-fence training tiles with Qiqi's UNMODIFIED script, one run per coordinate.

For each row of the coordinate CSV (schema: intake.py) this runs
  scripts/scan_tiles_overscan_crop.py --lat <lat> --lon <lon> --radius_km <r> <preset flags> --output <dir>
from the pinned vendor checkout, into
  <output_root>/<preset>/<facility_id>/<point_id>/   (tiles/*.png, manifest.csv, preview.html from the vendor)
plus our own files in the same directory:
  point.json              the intake row (incl. coordinate_type) the tile was centred on
  run.log                 vendor stdout/stderr, key-redacted
  extraction_record.json  command, preset flags, vendor commit, timings, statuses, request estimate
and appends one line per point to <output_root>/extraction_log.jsonl.

Safety:
  * the API key is read only from $GOOGLE_MAPS_STATIC_API_KEY and passed to the child process via its environment;
    it never appears on a command line, in a log, or in a manifest (manifests are sanitized after every run);
  * points whose record says "complete" and whose tiles are all present are skipped;
  * --dry-run prints the plan and request/cost estimate and makes NO API calls (the vendor's own --dry_run
    flag would still fetch every image, so it is never used);
  * there is no default preset: --preset is required (see README.md for why it matters).

    python extract_external_tiles.py --csv external_facilities.csv --preset auto_mosaic_1p9km --dry-run
    python extract_external_tiles.py --csv external_facilities.csv --preset auto_mosaic_1p9km --max-facilities 2
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from estimate_cost import RADIUS_CENTRE_ONLY, estimate
from geometry import REPO_ROOT, load_config, n_tiles, preset_flags, tile_geometry, vendor_script_path
from intake import load_intake
from secrets_guard import redact, sanitize_manifest, scan_dir_for_secrets
from vendor_runner import COUNT_TAG

RECORD = "extraction_record.json"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def api_key_or_exit(cfg: dict) -> str:
    key = os.environ.get(cfg["api_key_env"], "")
    if not key:
        raise SystemExit(f"${cfg['api_key_env']} is not set in this process's environment. Export it in the shell "
                         f"that runs this command (it is never read from or written to a file). Nothing was fetched.")
    return key


def vendor_state(cfg: dict) -> dict:
    vdir = REPO_ROOT / cfg["vendor"]["path"]
    head = subprocess.run(["git", "-C", str(vdir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(vdir), "status", "--porcelain"], capture_output=True, text=True).stdout
    return {"commit": head, "expected_commit": cfg["vendor"]["commit"], "modified_files": dirty.split("\n") if dirty.strip() else []}


def check_vendor(cfg: dict, allow_mismatch: bool) -> dict:
    if not vendor_script_path(cfg).exists():
        raise SystemExit(f"vendor script missing: {vendor_script_path(cfg)} (clone {cfg['vendor']['repo_url']} into "
                         f"{cfg['vendor']['path']} and check out {cfg['vendor']['commit']})")
    st = vendor_state(cfg)
    problems = []
    if st["commit"] != st["expected_commit"]:
        problems.append(f"vendor HEAD {st['commit'][:10]} != pinned {st['expected_commit'][:10]}")
    if st["modified_files"]:
        problems.append(f"vendor checkout has local modifications: {st['modified_files'][:5]}")
    if problems and not allow_mismatch:
        raise SystemExit("; ".join(problems) + " (use --allow-vendor-mismatch to override deliberately)")
    return st


def point_dir(output_root: Path, preset: str, point: dict) -> Path:
    return output_root / preset / point["facility_id"] / point["point_id"]


RUNNER = Path(__file__).resolve().parent / "vendor_runner.py"


def build_command(point: dict, preset: str, radius_km: float, out: Path) -> list[str]:
    # vendor_runner.py executes the unmodified vendor script and meters its Static Maps requests
    return [sys.executable, str(RUNNER), str(vendor_script_path()),
            "--lat", f"{float(point['latitude']):.8f}", "--lon", f"{float(point['longitude']):.8f}",
            "--radius_km", f"{max(radius_km, RADIUS_CENTRE_ONLY)}",
            *preset_flags(preset),
            "--output", str(out), "--region_name", "region"]   # "region" keeps benchmark-style filenames


def is_complete(out: Path, expected_tiles: int) -> bool:
    rec = out / RECORD
    if not rec.exists():
        return False
    try:
        ok = json.loads(rec.read_text()).get("status") == "complete"
    except json.JSONDecodeError:
        return False
    return ok and len(list((out / "tiles").glob("*.png"))) >= expected_tiles and (out / "manifest.csv").exists()


def extract_point(point: dict, preset: str, radius_km: float, out: Path, key: str, vendor: dict,
                  timeout_s: int = 1800) -> dict:
    """Run the vendor script for one coordinate. Returns the extraction record (also written to disk)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "point.json").write_text(json.dumps(point, indent=1, default=str))
    cmd = build_command(point, preset, radius_km, out)
    geom = tile_geometry(preset, float(point["latitude"]))
    expected = n_tiles(max(radius_km, RADIUS_CENTRE_ONLY), geom_tile_km(preset))
    started, t0 = now(), time.time()
    env = dict(os.environ)          # carries the key to the child; nothing else does
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s, cwd=out)
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        rc, stdout, stderr = -1, e.stdout or "", f"timeout after {timeout_s}s"
    (out / "run.log").write_text(redact(f"$ {' '.join(cmd)}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n", key))

    metered = [int(ln.split("=", 1)[1]) for ln in stdout.splitlines() if ln.startswith(COUNT_TAG)]
    manifest = out / "manifest.csv"
    statuses, n_redacted, mgeom = {}, 0, {}
    if manifest.exists():
        n_redacted = sanitize_manifest(manifest, key)
        m = pd.read_csv(manifest, dtype=str)
        statuses = m["status"].value_counts().to_dict()
        d = m[m["status"] == "downloaded"]
        if len(d) and "mode" in d and (d["mode"] == "auto_mosaic").all():
            r0 = d.iloc[0]
            mgeom = {"zoom": int(r0["zoom"]), "grid_n": int(r0["mosaic_grid_n"]), "crop_px": int(r0["crop_px"]),
                     "native_mpp": float(r0["native_mpp"]),
                     "footprint_width_km": round(int(r0["crop_px"]) * float(r0["native_mpp"]) / 1000, 4),
                     "tile_image_px": int(r0["tile_image_px"])}
    leaks = [str(p.relative_to(out)) for p in scan_dir_for_secrets(out, key)]
    n_png = len(list((out / "tiles").glob("*.png"))) if (out / "tiles").exists() else 0
    ok = rc == 0 and not leaks and n_png >= expected and not statuses.get("error")
    record = {
        "status": "complete" if ok else ("secret_leak" if leaks else "incomplete"),
        "point": point, "preset": preset, "preset_flags": preset_flags(preset),
        "radius_km_requested": radius_km, "radius_km_passed": max(radius_km, RADIUS_CENTRE_ONLY),
        "command": cmd, "vendor": vendor, "returncode": rc,
        "started_utc": started, "finished_utc": now(), "elapsed_s": round(time.time() - t0, 1),
        "expected_tiles": expected, "png_tiles": n_png, "manifest_status_counts": statuses,
        "manifest_cells_redacted": n_redacted, "files_still_containing_key": leaks,
        "geometry": {"zoom": geom.zoom, "grid_n": geom.grid_n, "footprint_width_km": round(geom.footprint_width_km, 4),
                     "effective_mpp": round(geom.effective_mpp, 5), "requests_per_tile_new": geom.requests_per_tile},
        "requests_estimate_upper": geom.requests_per_tile * statuses.get("downloaded", expected),
        "requests_metered": metered[-1] if metered else None,
        "manifest_geometry": mgeom,
    }
    (out / RECORD).write_text(json.dumps(record, indent=1, default=str))
    return record


def geom_tile_km(preset: str) -> float:
    flags = preset_flags(preset)
    return float(flags[flags.index("--tile_km") + 1]) if "--tile_km" in flags else load_config()["tile_km"]


def load_plan(path: Path, roles: list[str] | None) -> list[dict]:
    """Tile plan from dataset_plan.py: each planned tile is extracted as a single centred tile at its centre."""
    t = pd.read_csv(path)
    t = t[t["status"] == "planned"]
    if roles:
        t = t[t["role"].isin(roles)]
    pts = []
    for r in t.to_dict("records"):
        pts.append({**{k: v for k, v in r.items() if pd.notna(v)}, "point_id": r["tile_id"],
                    "latitude": float(r["center_lat"]), "longitude": float(r["center_lon"])})
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="coordinate CSV (schema: intake.py); one tile/grid per coordinate")
    src.add_argument("--plan", type=Path, help="tile_plan.csv from dataset_plan.py (one centred tile per row)")
    ap.add_argument("--preset", required=True, choices=sorted(load_config()["presets"]))
    ap.add_argument("--roles", nargs="+", help="with --plan: only these roles (positive, candidate_negative)")
    ap.add_argument("--radius-km", type=float, default=0.0, help="--csv only; 0 = one tile centred on each coordinate")
    ap.add_argument("--output-root", type=Path, default=None, help="default: output_root in extraction_config.yaml")
    ap.add_argument("--max-facilities", type=int, default=None, help="only the first N facility_ids (input order)")
    ap.add_argument("--max-requests", type=int, default=None,
                    help="hard budget of metered Static Maps requests for this run; no new tile starts once the "
                         "metered total plus the next tile's planned requests would exceed it")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="print plan + estimate; no API calls, nothing written")
    ap.add_argument("--allow-vendor-mismatch", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if args.plan:
        if args.radius_km:
            raise SystemExit("--radius-km is not used with --plan (each plan row is one centred tile)")
        points = load_plan(args.plan, args.roles)
    else:
        points = load_intake(args.csv).to_dict("records")
    if args.max_facilities is not None:
        keep = list(dict.fromkeys(p["facility_id"] for p in points))[:args.max_facilities]
        points = [p for p in points if p["facility_id"] in keep]
    output_root = args.output_root or (REPO_ROOT / cfg["output_root"])
    vendor = check_vendor(cfg, args.allow_vendor_mismatch)
    expected = n_tiles(max(args.radius_km, RADIUS_CENTRE_ONLY), geom_tile_km(args.preset))

    todo = [p for p in points if not is_complete(point_dir(output_root, args.preset, p), expected)]
    est = estimate([p["latitude"] for p in todo], args.radius_km, args.preset, geom_tile_km(args.preset)) if todo else None
    print(f"{len(points)} tiles/points from {len({p['facility_id'] for p in points})} facilities; "
          f"{len(points) - len(todo)} already complete; {len(todo)} to run with preset {args.preset}")
    if est:
        print(f"estimate for the remaining: {est['output_tiles']} tiles, <= {est['requests']} requests, "
              f"~${est['approx_cost_usd']} ({est['pricing_basis']}), ~{est['approx_png_gb']} GB PNG")

    if args.dry_run:
        for p in todo[:20]:
            print("DRY-RUN", " ".join(build_command(p, args.preset, args.radius_km,
                                                    point_dir(output_root, args.preset, p))))
        if len(todo) > 20:
            print(f"DRY-RUN ... and {len(todo) - 20} more")
        return

    key = api_key_or_exit(cfg)
    log_path = output_root / "extraction_log.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    state = {"metered": 0, "ok": 0, "done": 0, "stop": None}

    def run_one(p):
        g = tile_geometry(args.preset, float(p["latitude"]))
        with lock:
            if state["stop"]:
                return None
            if args.max_requests is not None and state["metered"] + g.requests_per_tile * expected > args.max_requests:
                state["stop"] = f"request budget {args.max_requests} reached (metered {state['metered']})"
                return None
        out = point_dir(output_root, args.preset, p)
        rec = extract_point(p, args.preset, args.radius_km, out, key, vendor)
        with lock:
            state["done"] += 1
            state["metered"] += rec["requests_metered"] or 0
            state["ok"] += rec["status"] == "complete"
            with open(log_path, "a") as f:
                f.write(json.dumps({k: rec[k] for k in ("status", "preset", "started_utc", "finished_utc", "elapsed_s",
                                                        "png_tiles", "expected_tiles", "manifest_status_counts",
                                                        "requests_metered", "requests_estimate_upper")}
                                   | {"point_id": p["point_id"], "facility_id": p["facility_id"],
                                      "dir": str(out.relative_to(output_root))}) + "\n")
            planned = g.requests_per_tile * expected
            if rec["status"] == "secret_leak":
                state["stop"] = f"API key still present in {rec['files_still_containing_key']} under {out}"
            elif rec["requests_metered"] is not None and rec["requests_metered"] > planned:
                state["stop"] = (f"{p['point_id']}: metered {rec['requests_metered']} requests > planned {planned}; "
                                 f"stopping on unexpected request volume")
            elif rec["status"] == "complete" and "--auto_mosaic" in preset_flags(args.preset):
                mg = rec["manifest_geometry"]
                if not mg or abs(mg["footprint_width_km"] - geom_tile_km(args.preset)) > 0.005 or mg["tile_image_px"] != 2560 \
                        or (mg["zoom"], mg["grid_n"]) != (g.zoom, g.grid_n):
                    state["stop"] = f"{p['point_id']}: unexpected geometry {mg} (planned z{g.zoom} grid{g.grid_n})"
            if state["done"] % 25 == 0 or rec["status"] != "complete":
                print(f"[{state['done']}/{len(todo)}] {p['point_id']}: {rec['status']} "
                      f"(metered {rec['requests_metered']} req; run total {state['metered']})", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(run_one, todo))
    print(f"{state['ok']}/{len(todo)} complete this run; {state['metered']} metered Static Maps requests. Log: {log_path}")
    if state["stop"]:
        raise SystemExit(f"STOPPED: {state['stop']}")


if __name__ == "__main__":
    main()
