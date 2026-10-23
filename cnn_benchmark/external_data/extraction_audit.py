"""Extraction / preprocessing audit: re-extract a small representative tile set with Qiqi's intended path
(`--auto_mosaic --output_px 2560`, preset auto_mosaic_1p9km, all other flags unchanged) into a FRESH output root and
compare each fresh tile with the tile the frozen CNN caches were built from. Reads labels/manifests; changes nothing.

    python extraction_audit.py extract     # Static Maps calls (key from $GOOGLE_MAPS_STATIC_API_KEY; never printed)
    python extraction_audit.py analyse     # no API calls -> results/qa/extraction_audit/

Fresh root: external_imagery/extraction_audit_2026-10-22/ (git-ignored). `extract` refuses to run if it exists, so no
tile from an earlier extraction mode can be found on disk and reused by the vendor's "tile already exists" path.
Per tile: vendor-selected zoom / grid / crop, PNG size, footprint = crop_px * native_mpp, GSD = footprint / 2560;
existing tile: its recorded mode / zoom / footprint where recoverable, plus an image comparison of fresh vs cached
1280 px arrays (same decode_reduce): NCC at scale 1 and the best-matching relative scale (detects a footprint mismatch).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parent
sys.path[:0] = [str(HERE), str(PKG / "src")]
FRESH = REPO / "external_imagery" / "extraction_audit_2026-10-22"
PRESET = "auto_mosaic_1p9km"
OUT = PKG.parent / "results" / "qa" / "extraction_audit"   # extraction_audit_table.csv is git-ignored (latitudes)
DATA = REPO / "data"
BENCH = {   # (region, tile_id, manifest holding the original extraction row)
    "Saratov": (["r21_c21", "r1_c21"], DATA / "saratov_40km_tiles_z16_mosaic 4" / "manifest.csv"),
    "Natanz": (["r1_c12", "r5_c16", "r20_c20"], DATA / "natanz_30km_tiles_auto_mosaic 3" / "manifest.csv"),
    "Fordow": (["r15_c15", "r25_c10"], DATA / "fordow_30km_tiles_auto_mosaic" / "manifest.csv"),
}
EXTERNAL = ["ext_kudankulam__pos", "ext_hamaoka__pos", "ext_braidwood__pos", "ext_ivg_1m__pos",
            "ext_fissile_material_storage_facility_fmsf__pos", "ext_kola__pos"]


def sample() -> list[dict]:
    pts = []
    for region, (ids, man) in BENCH.items():
        m = pd.read_csv(man)
        for t in ids:
            r, c = (int(x[1:]) for x in t.split("_"))
            row = m[(m.row == r) & (m.col == c)].iloc[0]
            pts.append({"point_id": f"{region}__{t}", "facility_id": f"audit_{region.lower()}", "kind": "benchmark",
                        "region": region, "tile_id": t, "latitude": float(row.center_lat), "longitude": float(row.center_lon),
                        "orig": {"mode": row.get("mode", "legacy_mosaic2x2") if isinstance(row.get("mode"), str) else "legacy_mosaic2x2",
                                 "status": row.status, "message": row.message, "zoom": row.zoom,
                                 "grid": row.get("mosaic_grid_n", 2.0 if region == "Saratov" else np.nan),
                                 "crop_px": row.crop_px, "mpp": row.get("native_mpp", row.returned_m_per_px),
                                 "tile_px": row.tile_image_px, "timestamp": row.timestamp_utc}})
    meta = pd.read_csv(REPO / "external_imagery/dataset/external_tiles_metadata_refined.csv").set_index("tile_id")
    for t in EXTERNAL:
        r = meta.loc[t]
        pts.append({"point_id": t, "facility_id": r.facility_id, "kind": "external", "region": "external", "tile_id": t,
                    "latitude": float(r.center_lat), "longitude": float(r.center_lon),
                    "orig": {"mode": "auto_mosaic", "status": "downloaded", "message": "ok", "zoom": r.source_zoom,
                             "grid": r.mosaic_grid, "crop_px": r.crop_px, "mpp": r.native_mpp, "tile_px": 2560,
                             "timestamp": ""}})
    return pts


def extract():
    import extract_external_tiles as X
    from geometry import load_config
    if FRESH.exists():
        raise SystemExit(f"{FRESH} already exists: refusing (the audit needs a never-used output root)")
    cfg = load_config()
    key = X.api_key_or_exit(cfg)
    vend = X.check_vendor(cfg, allow_mismatch=False)
    FRESH.mkdir(parents=True)
    recs = []
    for p in sample():
        out = X.point_dir(FRESH, PRESET, p)
        rec = X.extract_point({k: v for k, v in p.items() if k != "orig"}, PRESET, 0.0, out, key, vend)
        recs.append({"point_id": p["point_id"], "status": rec["status"], "requests": rec["requests_metered"],
                     "manifest_status": rec["manifest_status_counts"]})
        print(recs[-1], flush=True)
    (FRESH / "audit_extraction_summary.json").write_text(json.dumps(recs, indent=1))


def existing_array(p) -> np.ndarray:
    if p["kind"] == "benchmark":
        return np.load(PKG / "cache/decoded_1280px" / f"{p['region']}__{p['tile_id']}.npy")
    return np.load(PKG / "cache/external_decoded_1280px" / f"{p['tile_id']}.npy")


def existing_png_size(p):
    if p["kind"] == "external":
        m = pd.read_csv(REPO / "external_imagery/dataset/external_tiles_metadata_refined.csv").set_index("tile_id")
        with Image.open(REPO / m.loc[p["tile_id"], "png_path"]) as im:
            return im.size
    import decoded_cache
    cm = pd.read_csv(PKG / "cache/decoded_1280px/cache_manifest.csv")
    loc = cm[(cm.region == p["region"]) & (cm.tile_id == p["tile_id"])].source_locator.iloc[0]
    root = REPO / ("NEW_natanz" if p["region"] == "Natanz" else "data")
    loc = loc if Path(loc).is_absolute() else str((root / loc).resolve()) if "::" not in loc else loc
    try:
        raw = decoded_cache._read_source(loc if "::" not in loc or Path(loc.split("::")[0]).is_absolute()
                                         else str(root / loc.split("::")[0]) + "::" + loc.split("::")[1])
    except Exception as e:      # locator layout differs: report rather than guess
        return f"unread ({type(e).__name__})"
    with Image.open(io.BytesIO(raw)) as im:
        return im.size


def gray(a):
    g = a.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.asarray(Image.fromarray(g).resize((640, 640), Image.BILINEAR))


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def best_scale(fresh, old):
    """Relative footprint (old / fresh) that best aligns the centres: old zoomed by s vs fresh, central 320 px."""
    f, o = gray(fresh), gray(old)
    win = slice(160, 480)
    res = []
    for s in np.round(np.arange(0.70, 1.46, 0.01), 2):
        n = int(round(640 * s))
        z = np.asarray(Image.fromarray(o).resize((n, n), Image.BILINEAR))
        c0 = (n - 320) // 2
        if c0 < 0:
            continue
        res.append((ncc(f[win, win], z[c0:c0 + 320, c0:c0 + 320]), s))
    return max(res)


def analyse():
    import decoded_cache
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sample():
        d = FRESH / PRESET / p["facility_id"] / p["point_id"]
        rec = json.loads((d / "extraction_record.json").read_text())
        m = pd.read_csv(d / "manifest.csv").iloc[0]
        png = next((d / "tiles").glob("*.png"))
        raw = png.read_bytes()
        with Image.open(io.BytesIO(raw)) as im:
            size = im.size
        fresh = decoded_cache.decode_reduce(raw)
        old = existing_array(p)
        fp = float(m.crop_px) * float(m.native_mpp) / 1000
        o = p["orig"]
        ofp = (float(o["crop_px"]) * float(o["mpp"]) / 1000) if pd.notna(o["crop_px"]) and pd.notna(o["mpp"]) else np.nan
        sc_ncc, sc = best_scale(fresh, old)
        rows.append({
            "tile": p["point_id"], "lat": round(p["latitude"], 3),
            "fresh_status": f"{m.status}/{m.message}", "fresh_zoom": int(m.zoom), "fresh_grid": int(m.mosaic_grid_n),
            "fresh_px": f"{size[0]}x{size[1]}", "fresh_footprint_km": round(fp, 4), "fresh_gsd_m": round(fp * 1000 / size[0], 4),
            "model_input_gsd_m": round(fp * 1000 / 1280, 3),
            "orig_mode": o["mode"], "orig_status": f"{o['status']}/{o['message']}",
            "orig_zoom": o["zoom"], "orig_grid": o["grid"], "orig_png_px": existing_png_size(p),
            "orig_footprint_km": round(ofp, 4) if pd.notna(ofp) else "not recorded",
            "orig_gsd_m": round(ofp * 1000 / 2560, 4) if pd.notna(ofp) else "",
            "ncc_scale1": round(ncc(gray(fresh)[160:480, 160:480], gray(old)[160:480, 160:480]), 3),
            "best_rel_scale": sc, "ncc_best": round(sc_ncc, 3),
            "requests": rec["requests_metered"], "fresh_record_status": rec["status"],
        })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "extraction_audit_table.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    {"extract": extract, "analyse": analyse}[sys.argv[1]]()
