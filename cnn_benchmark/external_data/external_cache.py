"""Decoded 1280 px cache for the external ablation tiles (same decode as the benchmark cache: src/decoded_cache.py
decode_reduce = RGB + exact 2x box reduce of the 2560 px PNG). Source PNGs are only read.

    python external_cache.py      # every tile in the private ablation manifests -> cache/external_decoded_1280px/

Each entry is verified (shape 1280x1280x3, uint8) and recorded with source + cache SHA-256 in cache_manifest.csv.
Writes are atomic and the build is resumable.
"""
from __future__ import annotations

import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import decoded_cache  # noqa: E402
from common import PACKAGE_DIR  # noqa: E402
from geometry import REPO_ROOT  # noqa: E402

CACHE = PACKAGE_DIR / "cache" / "external_decoded_1280px"
MANIFESTS = REPO_ROOT / "external_imagery" / "dataset" / "ablation_manifests"     # private copies (png_path)


def one(args) -> dict:
    tile_id, png = args
    dst = CACHE / f"{tile_id}.npy"
    raw = (REPO_ROOT / png).read_bytes()
    rec = {"tile_id": tile_id, "png_path": png, "source_sha256": hashlib.sha256(raw).hexdigest(), "error": ""}
    try:
        if not dst.exists():
            arr = decoded_cache.decode_reduce(raw)
            tmp = dst.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, dst)
        arr = np.load(dst)
        if arr.shape != (1280, 1280, 3) or arr.dtype != np.uint8:
            raise ValueError(f"bad cache array {arr.shape} {arr.dtype}")
        rec["cache_sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception as e:                       # corrupt PNG / decode failure: recorded, never substituted
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    tiles = pd.concat([pd.read_csv(f)[["tile_id", "png_path"]] for f in sorted(MANIFESTS.glob("*.csv"))]).drop_duplicates("tile_id")
    with ProcessPoolExecutor(8) as ex:
        recs = list(ex.map(one, zip(tiles.tile_id, tiles.png_path), chunksize=8))
    man = pd.DataFrame(recs)
    man.to_csv(CACHE / "cache_manifest.csv", index=False)
    bad = man[man.error != ""]
    print(f"{len(man)} tiles cached in {CACHE} ({len(bad)} errors)")
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
