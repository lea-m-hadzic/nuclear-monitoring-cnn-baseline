"""Pre-decoded 1280 px cache: mapping, coverage, manifest, exact equivalence with the PNG path, git hygiene.

Skipped when the cache has not been built (python src/decoded_cache.py build).
"""
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch

import decoded_cache
import image_index
from common import PACKAGE_DIR
from dataset import FullResImagesMissing, ParentTileDataset, resolve_image_paths

CACHE = decoded_cache.cache_dir_from(None)
pytestmark = pytest.mark.skipif(not (CACHE / decoded_cache.MANIFEST).exists(), reason="decoded cache not built")


@pytest.fixture(scope="module")
def man():
    return pd.read_csv(CACHE / decoded_cache.MANIFEST)


@pytest.fixture(scope="module")
def resolved(parent_table):
    return image_index.resolve(parent_table, image_index.image_root_from(None))


def test_manifest_columns(man):
    need = {"region", "tile_id", "cache_file", "source_filename", "source_locator", "source_bytes",
            "source_sha256", "cache_sha256", "height", "width", "channels", "dtype", "reduce_factor"}
    assert need <= set(man.columns)
    assert (man.height == 1280).all() and (man.width == 1280).all() and (man.channels == 3).all()
    assert (man.dtype == "uint8").all() and (man.reduce_factor == 2).all()


def test_one_file_per_parent(man, parent_table):
    assert not man.duplicated(["region", "tile_id"]).any()
    assert not man.cache_file.duplicated().any()
    assert (man.cache_file == man.region + "__" + man.tile_id + ".npy").all()
    assert not man.source_sha256.duplicated().any()               # no two parents share one source image
    files = {p.name for p in CACHE.glob("*.npy")}
    assert files == set(man.cache_file)                            # no orphans, nothing missing
    meta = man.merge(parent_table[["region", "tile_id", "source_image_relpath"]], on=["region", "tile_id"])
    assert len(meta) == len(man)
    assert (meta.source_filename == meta.source_image_relpath.str.split("/").str[-1]).all()


def test_coverage_by_aoi(man, resolved):
    counts = man.groupby("region").size().to_dict()
    assert counts == {"Saratov": 1387, "Fordow": 777, "Natanz": 776} and len(man) == 2940
    assert (man[man.region == "Natanz"].source == "image_sources.Natanz").all()
    ok = resolved[resolved.status == "ok"]
    assert set(zip(man.region, man.tile_id)) == set(zip(ok.region, ok.tile_id))   # Natanz: resolved tiles only
    skipped = pd.read_csv(CACHE / "skipped_unresolved.csv")
    assert set(zip(skipped.region, skipped.tile_id)) == set(zip(*resolved[resolved.status != "ok"][["region", "tile_id"]].T.values))


def test_cached_tensors_identical_to_png_path(parent_table, man, resolved):
    kinds = resolved.set_index(["region", "tile_id"]).kind
    picks = []
    for region in ["Saratov", "Fordow", "Natanz"]:
        g = man[man.region == region]
        picks += list(g.sample(2, random_state=1).itertuples(index=False))
    zip_backed = [k for k in kinds.index if kinds[k] == "zip" and k in set(zip(man.region, man.tile_id))][:1]
    keys = [(p.region, p.tile_id) for p in picks] + zip_backed
    tiles = parent_table.set_index(["region", "tile_id"]).loc[keys].reset_index()
    a = ParentTileDataset(tiles, resolve_image_paths(tiles, None), 1280, train=False)
    b = ParentTileDataset(tiles, resolve_image_paths(tiles, None, decoded_cache=CACHE), 1280, train=False)
    for i in range(len(tiles)):
        xa, ya, ka = a[i]
        xb, yb, kb = b[i]
        assert ka == kb and torch.equal(ya, yb)
        assert torch.equal(xa, xb), f"cached tensor differs for {ka}"     # exact: same uint8 box-reduce


def test_cache_refuses_unresolved_tiles(parent_table):
    """Synthetic: tiles absent from the cache manifest must be refused, never substituted."""
    tiles = parent_table.head(2).assign(tile_id=["r999_c998", "r999_c999"])
    with pytest.raises(FullResImagesMissing, match="not in decoded cache"):
        resolve_image_paths(tiles, None, decoded_cache=CACHE)


def test_cache_is_git_ignored(man):
    root = PACKAGE_DIR.parent
    probe = [str((CACHE / decoded_cache.MANIFEST).relative_to(root)), str((CACHE / man.cache_file.iloc[0]).relative_to(root))]
    r = subprocess.run(["git", "-C", str(root), "check-ignore", *probe], capture_output=True, text=True)
    assert r.stdout.split() == probe
    staged = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--name-only"], capture_output=True, text=True)
    assert not any("cache/" in f or f.endswith(".npy") for f in staged.stdout.split())


def test_cached_arrays_spot_check(man):
    for f in man.cache_file.sample(5, random_state=2):
        arr = np.load(CACHE / f, mmap_mode="r")
        assert arr.shape == (1280, 1280, 3) and arr.dtype == np.uint8
