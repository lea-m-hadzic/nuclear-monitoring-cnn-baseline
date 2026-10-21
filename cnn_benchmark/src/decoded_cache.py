"""Non-destructive pre-decoded 1280 px image cache.

Each resolved z16 parent tile is decoded once from its source PNG (extracted file or zip member, read-only),
converted to RGB, reduced with PIL's exact 2x box filter (`Image.reduce(2)`: deterministic, the same
operation as the on-the-fly loader), and stored as an uncompressed uint8 array (1280, 1280, 3):

    <cache_dir>/<region>__<tile_id>.npy        one file per parent tile (exact one-to-one by name)
    <cache_dir>/cache_manifest.csv             AOI, tile ID, source filename/locator, source bytes + sha256,
                                               cache sha256, shape, dtype

Unresolved tiles (currently 530 Natanz) are skipped and listed, never substituted. Writes are atomic
(tmp + rename) and the build is resumable.

  python src/decoded_cache.py build            # default cache_dir from config
  python src/decoded_cache.py verify           # full check of every entry (+ re-decode sample)
"""
from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
from PIL import Image

import image_index
from common import OUTPUT_DIR, PACKAGE_DIR, REPO_ROOT, load_config

SIZE = 1280
FACTOR = 2
MANIFEST = "cache_manifest.csv"


def cache_dir_from(arg: str | None = None) -> Path:
    val = arg or os.environ.get("CNN_BENCH_DECODED_CACHE") or load_config().get("decoded_cache_dir")
    if not val:
        raise FileNotFoundError("decoded_cache_dir not configured")
    p = Path(val).expanduser()
    return p if p.is_absolute() else PACKAGE_DIR / p


def cache_name(region: str, tile_id: str) -> str:
    return f"{region}__{tile_id}.npy"


def _read_source(locator: str) -> bytes:
    """Raw bytes of the source PNG (file or zip member), for hashing and decoding."""
    if image_index.ZIP_SEP in locator:
        import zipfile
        z, member = locator.split(image_index.ZIP_SEP, 1)
        with zipfile.ZipFile(z) as zf:
            return zf.read(member)
    return Path(locator).read_bytes()


def decode_reduce(raw: bytes) -> np.ndarray:
    import io
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        if im.size != (SIZE * FACTOR, SIZE * FACTOR):
            raise ValueError(f"unexpected source size {im.size}")
        return np.asarray(im.reduce(FACTOR), dtype=np.uint8)


def _build_one(job):
    region, tile_id, locator, out_dir = job
    out = Path(out_dir) / cache_name(region, tile_id)
    raw = _read_source(locator)
    arr = decode_reduce(raw)
    tmp = out.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, out)
    return dict(region=region, tile_id=tile_id, cache_file=out.name,
                source_filename=os.path.basename(locator.split(image_index.ZIP_SEP)[-1]), source_locator=locator,
                source_bytes=len(raw), source_sha256=hashlib.sha256(raw).hexdigest(),
                cache_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                height=arr.shape[0], width=arr.shape[1], channels=arr.shape[2], dtype=str(arr.dtype),
                reduce_factor=FACTOR, pil_version=PIL.__version__)


def _source_sha256(path: str) -> str:
    return hashlib.sha256(_read_source(path)).hexdigest()


def build(cache_dir: Path, workers: int = 8) -> pd.DataFrame:
    """Add missing tiles; re-point (without rebuilding) tiles whose resolved source moved but whose source
    bytes are identical (SHA-256); rebuild a tile only if its source bytes changed."""
    root = image_index.image_root_from(None)
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    idx = image_index.resolve(pt, root)
    ok, skipped = idx[idx.status == "ok"], idx[idx.status != "ok"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    man_path = cache_dir / MANIFEST
    done = pd.read_csv(man_path) if man_path.exists() else pd.DataFrame(columns=["region", "tile_id"])
    loc_now = ok.set_index(["region", "tile_id"]).locator
    src_now = ok.set_index(["region", "tile_id"]).source

    # Existing entries whose resolved source location changed (e.g. a canonical re-upload).
    rebuild, repointed = set(), 0
    if len(done):
        key = list(zip(done.region, done.tile_id))
        moved = done[[k in loc_now.index and loc_now[k] != loc for k, loc in zip(key, done.source_locator)]]
        if len(moved):
            mk = list(zip(moved.region, moved.tile_id))
            with ProcessPoolExecutor(workers) as ex:
                shas = list(ex.map(_source_sha256, [str(root / loc_now[k]) for k in mk], chunksize=4))
            for i, (k, sha) in zip(moved.index, zip(mk, shas)):
                if sha == done.at[i, "source_sha256"]:
                    done.at[i, "source_locator"] = loc_now[k]
                    done.at[i, "source_filename"] = os.path.basename(loc_now[k].split(image_index.ZIP_SEP)[-1])
                    repointed += 1
                else:
                    rebuild.add(k)
    have = set(zip(done.region, done.tile_id)) if len(done) else set()
    todo = ok[[(r, t) not in have or (r, t) in rebuild or not (cache_dir / cache_name(r, t)).exists()
               for r, t in zip(ok.region, ok.tile_id)]]
    print(f"resolved {len(ok)} | already cached {len(ok) - len(todo)} (re-pointed to new identical source: "
          f"{repointed}) | to build {len(todo)} (source changed: {len(rebuild)}) | skipped unresolved "
          f"{len(skipped)} {skipped.groupby('region').size().to_dict()}", flush=True)
    jobs = [(r.region, r.tile_id, str(root / r.locator), str(cache_dir)) for r in todo.itertuples()]
    recs = []
    with ProcessPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(_build_one, jobs, chunksize=4), 1):
            recs.append(rec)
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    keep = done[~done.set_index(["region", "tile_id"]).index.isin([(j[0], j[1]) for j in jobs])] if len(done) else done
    man = pd.concat([keep, pd.DataFrame(recs)], ignore_index=True)
    man["source_locator"] = man.source_locator.map(lambda p: os.path.relpath(p.split(image_index.ZIP_SEP)[0], root)
                                                   + (image_index.ZIP_SEP + p.split(image_index.ZIP_SEP, 1)[1]
                                                      if image_index.ZIP_SEP in p else "")
                                                   if os.path.isabs(str(p)) else p)
    man["source"] = [src_now.get((r, t), "") for r, t in zip(man.region, man.tile_id)]
    man.sort_values(["region", "tile_id"]).to_csv(man_path, index=False)
    skipped[["region", "tile_id", "filename", "status"]].to_csv(cache_dir / "skipped_unresolved.csv", index=False)
    return man


def _verify_one(job):
    path, sha = job
    arr = np.load(path, mmap_mode="r")
    ok_shape = arr.shape == (SIZE, SIZE, 3) and arr.dtype == np.uint8
    return ok_shape, hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest() == sha


def verify(cache_dir: Path, redecode_sample: int = 24, workers: int = 8) -> dict:
    root = image_index.image_root_from(None)
    man = pd.read_csv(cache_dir / MANIFEST)
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    idx = image_index.resolve(pt, root)
    ok = idx[idx.status == "ok"]
    files = {p.name for p in cache_dir.glob("*.npy")}
    res = dict(
        manifest_rows=len(man),
        unique_keys=int((~man.duplicated(["region", "tile_id"])).sum()),
        unique_cache_files=int(man.cache_file.nunique()),
        manifest_equals_resolved=set(zip(man.region, man.tile_id)) == set(zip(ok.region, ok.tile_id)),
        source_filenames_match_metadata=bool((man.merge(pt[["region", "tile_id", "source_image_relpath"]],
                                                         on=["region", "tile_id"]).pipe(
            lambda d: d.source_filename == d.source_image_relpath.str.split("/").str[-1])).all()),
        names_match_keys=bool((man.cache_file == [cache_name(r, t) for r, t in zip(man.region, man.tile_id)]).all()),
        files_on_disk=len(files), orphan_files=sorted(files - set(man.cache_file))[:5],
        missing_files=sorted(set(man.cache_file) - files)[:5],
        source_locators_match_index=bool((man.merge(ok, on=["region", "tile_id"]).pipe(
            lambda d: d.source_locator == d.locator)).all()),
        unique_source_sha256=int(man.source_sha256.nunique()),
        counts_by_region=man.groupby("region").size().to_dict())
    with ProcessPoolExecutor(workers) as ex:
        chk = list(ex.map(_verify_one, [(str(cache_dir / f), s) for f, s in zip(man.cache_file, man.cache_sha256)],
                          chunksize=16))
    res["shape_dtype_ok"] = int(sum(c[0] for c in chk))
    res["cache_sha256_ok"] = int(sum(c[1] for c in chk))
    sample = man.sample(min(redecode_sample, len(man)), random_state=0)
    if "source" in man:      # always include re-decode checks from every source (e.g. canonical Natanz)
        sample = pd.concat([sample] + [g.sample(min(4, len(g)), random_state=0) for _, g in man.groupby("source")]
                           ).drop_duplicates(["region", "tile_id"])
    same = 0
    for r in sample.itertuples():
        raw = _read_source(str(root / r.source_locator))
        same += hashlib.sha256(raw).hexdigest() == r.source_sha256 and \
            np.array_equal(decode_reduce(raw), np.load(cache_dir / r.cache_file))
    res["redecode_identical"] = f"{same}/{len(sample)}"
    res["disk_bytes"] = int(sum((cache_dir / f).stat().st_size for f in files))
    res["passed"] = (res["manifest_rows"] == res["unique_keys"] == res["unique_cache_files"] == len(files)
                     == res["shape_dtype_ok"] == res["cache_sha256_ok"] and res["manifest_equals_resolved"]
                     and res["names_match_keys"] and res["source_filenames_match_metadata"] and res["source_locators_match_index"]
                     and res["unique_source_sha256"] == len(man) and same == len(sample))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--cache-dir")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    d = cache_dir_from(args.cache_dir)
    if args.cmd == "build":
        build(d, args.workers)
    res = verify(d, workers=args.workers)
    for k, v in res.items():
        print(f"{k}: {v}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
