"""Tests for the ResNet ablation setup: manifests, shared val set, geometric watermark mask + fill, dataset, metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "external_data"))

MAN = PKG / "outputs" / "ablation" / "manifests"
needs_manifests = pytest.mark.skipif(not (MAN / "checks.json").exists(), reason="manifests not built")


@needs_manifests
def test_manifest_invariants():
    m = {c: pd.read_csv(MAN / f"{c}.csv") for c in ("A", "B", "C", "C_matched")}
    pos = {c: set(df[df.label == 1].tile_id) for c, df in m.items()}
    assert len({frozenset(p) for p in pos.values()}) == 1 and len(pos["A"]) == 287
    negs = {c: df[df.label == 0].split.value_counts().sort_index().to_dict() for c, df in m.items()}
    assert all(v == {"train": 126, "val": 31} for v in negs.values())
    assert set(m["A"].role[m["A"].label == 0]) == {"nearby"} and set(m["B"].role[m["B"].label == 0]) == {"hard"}
    c = m["C"][m["C"].label == 0]
    assert c.role.value_counts().to_dict() == {"hard": 79, "nearby": 78}
    assert set(c[c.role == "nearby"].tile_id) <= set(m["A"].tile_id)                    # nested in A
    assert set(m["C_matched"][m["C_matched"].role == "hard"].tile_id) == set(c[c.role == "hard"].tile_id)
    for df in m.values():
        assert not df.tile_id.duplicated().any()
        assert (df.groupby("split_group_id").split.nunique() == 1).all()
        assert (df.groupby("facility_id").split.nunique() == 1).all()


def test_fill_rgb_only_changes_masked_pixels():
    from watermark_mask import fill_rgb
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), bool); mask[30:34, 10:50] = True
    out = fill_rgb(arr, mask)
    assert out.dtype == np.uint8 and out.shape == arr.shape
    assert np.array_equal(out[~mask], arr[~mask]) and not np.array_equal(out[mask], arr[mask])
    flat = np.full((64, 64, 3), 77, np.uint8)
    assert np.array_equal(fill_rgb(flat, mask), flat)                                    # constant image stays constant


@needs_manifests
def test_shared_val_set_identical_and_leak_free():
    sv = pd.read_csv(MAN / "shared_val.csv")
    assert sv.role.value_counts().to_dict() == {"positive": 56, "nearby": 31, "hard": 31} and (sv.split == "val").all()
    for c in ("A", "B", "C", "C_matched"):
        m = pd.read_csv(MAN / f"{c}.csv")
        tr = m[m.split == "train"]
        assert not set(sv.tile_id) & set(tr.tile_id) and not set(sv.split_group_id) & set(tr.split_group_id)
        assert not set(sv.facility_id) & set(tr.facility_id)
        if c != "C_matched":
            assert set(m[m.split == "val"].tile_id) <= set(sv.tile_id)


def _template_mosaic(grid: int) -> np.ndarray:
    from watermark_geometry import RECTS, SOURCE_PX
    big = np.zeros((grid * SOURCE_PX,) * 2, bool)
    for (u0, u1), (v0, v1) in RECTS.values():
        for r in range(grid):
            for c in range(grid):
                big[r * SOURCE_PX + v0:r * SOURCE_PX + v1, c * SOURCE_PX + u0:c * SOURCE_PX + u1] = True
    return big


@pytest.mark.parametrize("grid,crop", [(2, 2560), (3, 2561), (3, 3819), (4, 3873), (4, 5107)])
def test_geometric_mask_covers_overlay_after_crop_and_resize(grid, crop):
    """Overlay drawn on every source image, centre-cropped and area-resized to 1280 px, lies inside the mask."""
    from PIL import Image
    from watermark_geometry import geometry_mask
    big = _template_mosaic(grid)
    off = (grid * 1280 - crop) // 2
    im = Image.fromarray((big[off:off + crop, off:off + crop] * 255).astype(np.uint8)).resize((1280, 1280), Image.BOX)
    overlay = np.asarray(im) > 0
    m = geometry_mask(grid, float(crop))
    assert overlay.any() and (overlay & ~m).sum() == 0
    assert m.mean() < 1.5 * overlay.mean() + 0.005                       # template + 3 px margin, nothing more


def test_legacy_mask_rows_and_cols():
    from watermark_geometry import geometry_mask
    m = geometry_mask(2, 2560.0)                                           # Saratov: 2x2 z16, no crop, 2x reduce
    rows = np.where(m.any(1))[0]
    assert set(rows) == set(range(616, 642)) | set(range(1256, 1280))      # source v 1232..1283 in both image rows
    assert m[630, 0] and m[630, 700] and not m[600, 700] and not m[630, 100] and m[1270, 1279]


def test_mask_bank_lookup_and_dataset_uses_per_tile_mask(tmp_path):
    from train_ablation import NpyTileDataset
    from watermark_geometry import MaskBank, geometry_mask
    bank = MaskBank.__new__(MaskBank)
    bank.geo = {"a": (2, 2560.0), "b": (4, 3873.0)}
    with pytest.raises(KeyError):
        bank.get("zzz")
    arr = np.random.default_rng(0).integers(0, 256, (1280, 1280, 3), dtype=np.uint8)
    for t in "ab":
        np.save(tmp_path / f"{t}.npy", arr)
    ds, plain = NpyTileDataset(["a", "b"], [1, 0], tmp_path, bank, train=False), NpyTileDataset(["a", "b"], [1, 0], tmp_path, None, train=False)
    for i, t in enumerate("ab"):
        diff = (ds[i][0] != plain[i][0]).any(0).numpy()
        m = geometry_mask(*bank.geo[t])
        assert diff[~m].sum() == 0 and diff[m].mean() > 0.9


def test_built_geometry_table_if_present():
    from watermark_geometry import OUT, MaskBank, mask_sha, geometry_mask
    if not (OUT / "geometry.csv").exists():
        pytest.skip("geometry table not built")
    g = pd.read_csv(OUT / "geometry.csv")
    assert not g.key.duplicated().any()
    assert set(g.domain) == {"external", "Saratov", "Natanz", "Fordow"}
    sar = g[g.domain == "Saratov"]
    assert (sar.layout == "legacy_mosaic2x2").all() and (sar.grid == 2).all() and (sar.crop_px == 2560).all()
    assert set(g[g.domain == "Natanz"].grid) == {3} and set(g[g.domain == "Fordow"].grid) == {4}
    r = g.sample(20, random_state=0)
    assert all(mask_sha(geometry_mask(int(a), float(b))) == s for a, b, s in zip(r.grid, r.crop_px, r.mask_sha256))
    if (MAN / "shared_val.csv").exists():
        keys = set(g.key)
        for f in MAN.glob("*.csv"):
            assert set(pd.read_csv(f).tile_id) <= keys                     # every ablation tile has a geometry
    assert MaskBank().table_sha256


def test_dataset_applies_mask_before_normalisation(tmp_path):
    from train_ablation import NpyTileDataset
    arr = np.zeros((1280, 1280, 3), np.uint8); arr[:, :640] = 200
    np.save(tmp_path / "t.npy", arr)
    mask = np.zeros((1280, 1280), bool); mask[600:700, 600:700] = True
    x_masked, y, k = NpyTileDataset(["t"], [1], tmp_path, mask, train=False)[0]
    x_plain, _, _ = NpyTileDataset(["t"], [1], tmp_path, None, train=False)[0]
    assert x_masked.shape == (3, 1280, 1280) and k == "t" and float(y) == 1.0
    diff = (x_masked != x_plain).any(0).numpy()
    assert diff[~mask].sum() == 0 and diff[mask].any()


def test_metrics():
    from evaluate_ablation import auroc, pr_auc
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.4, 0.35, 0.8])
    assert auroc(y, s) == pytest.approx(0.75)
    assert pr_auc(y, s) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert auroc(np.array([1, 1]), np.array([0.2, 0.3])) != auroc(np.array([1, 1]), np.array([0.2, 0.3]))   # nan
