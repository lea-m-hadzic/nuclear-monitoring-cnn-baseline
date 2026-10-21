"""CNN scaffolding: clear failure without full-res images; dataset/model shapes (thumbnails = plumbing only)."""
import pytest
import torch

import image_index
from dataset import FullResImagesMissing, ParentTileDataset, augment, resolve_image_paths
from model import build_model


def test_fails_clearly_without_fullres(parent_table, tmp_path, monkeypatch):
    tiles = parent_table.head(3)
    monkeypatch.delenv("CNN_BENCH_IMAGE_ROOT", raising=False)
    monkeypatch.setattr(image_index, "load_config", lambda: {"image_root": None})
    with pytest.raises(FullResImagesMissing, match="not configured"):
        resolve_image_paths(tiles, None)
    with pytest.raises(FullResImagesMissing, match="does not exist"):
        resolve_image_paths(tiles, str(tmp_path / "nope"))
    with pytest.raises(FullResImagesMissing, match="not resolvable"):
        resolve_image_paths(tiles, str(tmp_path))


def test_thumbnail_dataset_and_model_forward(parent_table):
    tiles = parent_table[parent_table.training_target.notna()].head(4)
    paths = resolve_image_paths(tiles, None, smoke_test_thumbnails=True)
    ds = ParentTileDataset(tiles, paths, input_size=64, train=True)
    x, y, key = ds[0]
    assert x.shape == (3, 64, 64) and y.item() in (0.0, 1.0) and "|" in key
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.stack([ds[i][0] for i in range(2)]))
    assert out.shape == (2, 1)


def test_augment_preserves_content():
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4)
    for _ in range(10):
        y = augment(x)
        assert y.shape == x.shape and torch.equal(y.flatten().sort().values, x.flatten().sort().values)
