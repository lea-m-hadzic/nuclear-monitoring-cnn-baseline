"""Full-resolution cache tests. Skipped when the configured image_root is absent (e.g. on other machines).

Natanz resolves from the canonical re-upload configured in `image_sources` (776/776). History: the first
shared Natanz folder held only 246/776 benchmark tiles plus 261 off-grid tiles; it is no longer used.
"""
import numpy as np
import pandas as pd
import pytest

import image_index
from dataset import FullResImagesMissing, ParentTileDataset, resolve_image_paths

EXPECTED = {"Saratov": 1387, "Natanz": 776, "Fordow": 777}

try:
    ROOT = image_index.image_root_from(None)
    HAVE_CACHE = ROOT.is_dir() and any(ROOT.glob("*/tiles/*.png"))
except FileNotFoundError:
    HAVE_CACHE = False
pytestmark = pytest.mark.skipif(not HAVE_CACHE, reason="full-resolution image cache not present")


@pytest.fixture(scope="module")
def index(parent_table):
    return image_index.resolve(parent_table, ROOT)


@pytest.fixture(scope="module")
def headers(index):
    return image_index.check_images(index, ROOT, deep=False)


def test_index_covers_every_parent_once(index, parent_table):
    assert len(index) == len(parent_table)
    assert not index.duplicated(["region", "tile_id"]).any()
    assert index.groupby("region").size().to_dict() == EXPECTED


@pytest.mark.parametrize("region", ["Saratov", "Fordow", "Natanz"])
def test_every_parent_resolves_to_exactly_one_image(index, region):
    g = index[index.region == region]
    assert (g.status == "ok").all(), g.status.value_counts().to_dict()
    assert (g.n_extracted + g.n_zipped).ge(1).all()
    assert ((g.n_extracted <= 1) & (g.n_zipped <= 1)).all()     # never two candidate copies of the same kind


def test_natanz_resolves_completely_from_canonical_source(index):
    nat = index[index.region == "Natanz"]
    assert len(nat) == 776 and (nat.status == "ok").all()
    assert (nat.source == "image_sources.Natanz").all()                 # never the old mixed folder
    assert nat.locator.str.contains("natanz_30km_tiles_auto_mosaic_reupload").all()
    assert (index[index.region != "Natanz"].source == "image_root").all()


def test_no_image_maps_to_multiple_parents(index):
    ok = index[index.status == "ok"]
    assert not ok.locator.duplicated().any()
    assert not ok.duplicated(["scan_dir", "filename"]).any()


def test_resolved_files_are_the_parent_tiles_themselves(index, parent_table):
    """Filename must encode the parent's own row, col, centre and 1.90 km width: no quadrant/other-grid image."""
    ok = index[index.status == "ok"].merge(parent_table, on=["region", "tile_id"])
    m = ok.filename.str.extract(image_index.NAME_RE)
    assert m[0].notna().all() and m[5].isna().all()                       # parsable, no "(1)" copies
    assert (m[0].astype(int) == ok.row).all() and (m[1].astype(int) == ok.col).all()
    assert np.allclose(m[2].astype(float), ok.center_lat, atol=1e-5)
    assert np.allclose(m[3].astype(float), ok.center_lon, atol=1e-5)
    assert (m[4] == "1.90").all()
    assert not ok.filename.str.contains("zoom|__z|quadrant|_NW|_NE|_SW|_SE").any()


def test_image_dimensions(headers):
    ok = headers[headers.status == "ok"]
    assert ok.read_error.fillna("").eq("").all()
    assert (ok.width == image_index.EXPECTED_PX).all() and (ok.height == image_index.EXPECTED_PX).all()
    assert (ok["mode"] == "RGB").all()


def test_loader_on_fullres_file_and_zip_member(index, parent_table):
    sar = parent_table[parent_table.region == "Saratov"]
    kinds = index[(index.region == "Saratov") & (index.status == "ok")].set_index("tile_id").kind
    pick = pd.concat([sar[sar.tile_id.isin(kinds.index[kinds == k])].head(1) for k in ("file", "zip")])
    pick = pd.concat([pick, sar[sar.exclude_from_training | (sar.human_target == 1)].head(1)])
    paths = resolve_image_paths(pick, None)
    assert any("::" in p for p in paths) and any("::" not in p for p in paths)
    ds = ParentTileDataset(pick, paths, input_size=640, train=True)
    for i, (_, r) in enumerate(pick.iterrows()):
        x, y, key = ds[i]
        assert x.shape == (3, 640, 640) and x.dtype.is_floating_point
        assert key == f"Saratov|{r.tile_id}"
        expected = -1.0 if pd.isna(r.training_target) else float(r.training_target)
        assert y.item() == expected


def test_loader_refuses_unresolved_tiles(parent_table):
    """Synthetic: point two real parents at filenames that exist nowhere -> must be refused, not substituted."""
    tiles = parent_table[parent_table.region == "Natanz"].head(2).copy()
    tiles["source_image_relpath"] = tiles.source_image_relpath.str.replace(".png", "_DOES_NOT_EXIST.png", regex=False)
    with pytest.raises(FullResImagesMissing, match="not resolvable"):
        resolve_image_paths(tiles, None)


def test_dataset_targets_ignore_vlm_scores(parent_table, index):
    sar = parent_table[(parent_table.region == "Saratov")].head(20)
    paths = pd.Series(["unused"] * len(sar), index=sar.index)
    a = ParentTileDataset(sar, paths, 640, train=False).targets
    scrambled = sar.assign(vlm_score_single_pass=-sar.vlm_score_single_pass, vlm_score_adaptive_max=99)
    assert ParentTileDataset(scrambled, paths, 640, train=False).targets == a


def test_training_blocked_while_cache_incomplete(monkeypatch):
    """Synthetic: mark one parent of another AOI unresolved -> the completeness guard must fire before training."""
    import train
    real = image_index.resolve

    def one_missing(pt, root):
        idx = real(pt, root)
        idx.loc[idx.index[idx.region == "Natanz"][0], "status"] = "missing"
        return idx
    monkeypatch.setattr(image_index, "resolve", one_missing)
    with pytest.raises(FullResImagesMissing, match="Training is blocked"):
        train.main(["--split", "saratov_spatial_cv", "--fold", "0", "--epochs", "1"])
