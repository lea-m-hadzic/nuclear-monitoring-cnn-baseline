"""Full-resolution (2560 px) caches for the C-only resolution sensitivity. Source PNGs are only read; frozen caches untouched.

    python fullres_cache.py external    # C manifest + shared val tiles -> cache/external_decoded_2560px/
    python fullres_cache.py benchmark   # all 2,940 benchmark parent tiles -> cache/decoded_2560px/
    python fullres_cache.py fill-external | fill-benchmark   # + geometric watermark fill -> *_2560px_filled/

Filled caches: the frozen fill (watermark_mask.fill_rgb, per-tile geometric mask rendered at 2560, sigma 12 px = the
1280 runs' 6 px at the same physical scale) applied ONCE and stored. The fill is deterministic and happens before
augmentation, so training / scoring from the filled cache gives byte-identical inputs to filling on the fly; it only
avoids ~2.4 s of CPU per tile per epoch.

Each tile is decoded from the SAME source file recorded in the frozen 1280 cache manifest:
  * source SHA-256 must equal the frozen manifest's source_sha256 (same extracted imagery);
  * array = RGB uint8 2560 x 2560, no reduction;
  * reduce check: PIL Image.reduce(2) of that array must equal the frozen 1280 cache array exactly, i.e. the only
    difference from the frozen input is the 2x box reduction.
Writes cache_manifest.csv (source + cache SHA-256, reduce check) per cache. Atomic writes; resumable.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))
import decoded_cache  # noqa: E402
import image_index  # noqa: E402

REPO = PKG.parent
SIZE = 2560


def decode_full(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        if im.size != (SIZE, SIZE):
            raise ValueError(f"unexpected source size {im.size}")
        return np.asarray(im, dtype=np.uint8)


def one(job) -> dict:
    key, src, want_sha, frozen_npy, dst = job
    rec = {"key": key, "source": src, "error": ""}
    try:
        raw = decoded_cache._read_source(src)
        rec["source_sha256"] = hashlib.sha256(raw).hexdigest()
        if rec["source_sha256"] != want_sha:
            raise ValueError("source SHA-256 differs from the frozen 1280 cache manifest")
        dst = Path(dst)
        if dst.exists():
            arr = np.load(dst)
        else:
            arr = decode_full(raw)
            tmp = dst.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, dst)
        if arr.shape != (SIZE, SIZE, 3) or arr.dtype != np.uint8:
            raise ValueError(f"bad array {arr.shape} {arr.dtype}")
        reduced = np.asarray(Image.fromarray(arr).reduce(2), dtype=np.uint8)
        rec["reduce_equals_frozen_1280"] = bool(np.array_equal(reduced, np.load(frozen_npy)))
        rec["cache_sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception as e:                           # recorded, never substituted
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def fill_one(job) -> dict:
    key, src, dst = job
    from watermark_geometry import MaskBank
    from watermark_mask import fill_rgb
    mb = fill_one.mb = getattr(fill_one, "mb", None) or MaskBank(PKG / "cache" / "watermark_geometry" / "geometry.csv", n=SIZE)
    dst = Path(dst)
    if not dst.exists():
        arr = fill_rgb(np.load(src), mb.get(key), 12.0)
        tmp = dst.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, dst)
    return {"key": key, "filled_sha256": hashlib.sha256(np.load(dst).tobytes()).hexdigest()}


def fill(which: str):
    src = PKG / "cache" / ("external_decoded_2560px" if which == "external" else "decoded_2560px")
    out = PKG / "cache" / (src.name + "_filled")
    m = pd.read_csv(src / "cache_manifest.csv")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, str(src / f"{k}.npy"), str(out / f"{k}.npy")) for k in m.key]
    with ProcessPoolExecutor(8) as ex:
        recs = list(ex.map(fill_one, jobs, chunksize=4))
    f = pd.DataFrame(recs).merge(m[["key", "cache_sha256"]], on="key")
    f["fill"] = "watermark_mask.fill_rgb(sigma=12) with MaskBank(n=2560)"
    f.to_csv(out / "cache_manifest.csv", index=False)
    print(f"fill-{which}: {len(f)} tiles -> {out}")


def main():
    which = sys.argv[1]
    if which.startswith("fill-"):
        return fill(which.split("-", 1)[1])
    if which == "external":
        frozen = PKG / "cache" / "external_decoded_1280px"
        out = PKG / "cache" / "external_decoded_2560px"
        m = pd.read_csv(frozen / "cache_manifest.csv")
        need = set(pd.read_csv(PKG / "outputs/ablation/manifests/C.csv").tile_id) | set(
            pd.read_csv(PKG / "outputs/ablation/manifests/shared_val.csv").tile_id)
        m = m[m.tile_id.isin(need)]
        if len(m) != len(need):
            raise SystemExit("some C / shared-val tiles are missing from the frozen external cache manifest")
        jobs = [(t, str(REPO / p), s, str(frozen / f"{t}.npy"), str(out / f"{t}.npy"))
                for t, p, s in zip(m.tile_id, m.png_path, m.source_sha256)]
    else:
        frozen = PKG / "cache" / "decoded_1280px"
        out = PKG / "cache" / "decoded_2560px"
        m = pd.read_csv(frozen / "cache_manifest.csv")
        root = image_index.image_root_from(None)
        jobs = [(f"{r}__{t}", str(root / loc), s, str(frozen / f), str(out / f))
                for r, t, loc, s, f in zip(m.region, m.tile_id, m.source_locator, m.source_sha256, m.cache_file)]
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(8) as ex:
        recs = list(ex.map(one, jobs, chunksize=4))
    man = pd.DataFrame(recs)
    man["source"] = man.source.map(lambda p: os.path.relpath(p.split("::")[0], REPO) + ("::" + p.split("::", 1)[1] if "::" in p else ""))
    man.to_csv(out / "cache_manifest.csv", index=False)
    bad = man[(man.error != "") | (man.reduce_equals_frozen_1280 != True)]  # noqa: E712
    print(f"{which}: {len(man)} tiles -> {out} | errors {int((man.error != '').sum())} | "
          f"reduce==frozen1280 {int((man.reduce_equals_frozen_1280 == True).sum())}/{len(man)}")  # noqa: E712
    if len(bad):
        print(bad.head().to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
