"""Parent-tile image dataset.

Full-resolution tiles are located by image_index.resolve(): exact (scan_dir, original filename) match from
the VLM CSV `image_path`, in extracted "<scan_dir>[ N]/tiles/" folders or directly inside the zip parts.
The 320 px report thumbnails can be used ONLY for pipeline smoke tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import image_index
from common import REPO_ROOT

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SMOKE_TEST_LABEL = "PIPELINE_SMOKE_TEST_ONLY"
SANITY_LABEL = "FULLRES_SANITY_CHECK_ONLY"


class FullResImagesMissing(FileNotFoundError):
    pass


def resolve_image_paths(tiles: pd.DataFrame, image_root: str | None, smoke_test_thumbnails: bool = False,
                        decoded_cache: str | Path | None = None) -> pd.Series:
    """Map parent tiles to image files; raise FullResImagesMissing with a clear message if absent.

    decoded_cache: directory built by decoded_cache.py; tiles are then read from <region>__<tile_id>.npy
    (exact 2x box-reduced 1280 px copies of the source PNGs). Every requested tile must be in the cache.
    """
    if decoded_cache is not None and not smoke_test_thumbnails:
        d = Path(decoded_cache)
        man_path = d / "cache_manifest.csv"
        if not man_path.exists():
            raise FullResImagesMissing(f"decoded cache not built: {man_path} missing (run `python src/decoded_cache.py build`)")
        man = pd.read_csv(man_path, usecols=["region", "tile_id", "cache_file"])
        keyed = man.set_index(["region", "tile_id"]).cache_file
        keys = list(zip(tiles.region, tiles.tile_id))
        missing = [k for k in keys if k not in keyed.index or not (d / keyed[k]).is_file()]
        if missing:
            raise FullResImagesMissing(f"{len(missing)}/{len(keys)} tiles not in decoded cache {d}, e.g. {missing[:3]}")
        return pd.Series([str(d / keyed[k]) for k in keys], index=tiles.index)
    if smoke_test_thumbnails:
        paths = tiles.thumbnail_relpath.map(lambda p: REPO_ROOT / p)
    else:
        try:
            root = image_index.image_root_from(image_root)
        except FileNotFoundError:
            raise FullResImagesMissing(
                "Full-resolution image cache is not configured. Set `image_root` in configs/default.yaml, "
                "$CNN_BENCH_IMAGE_ROOT, or pass --image-root. Expected files: <image_root>/<scan_dir>[ N]/tiles/<name>.png "
                f"(or zip parts), e.g. {tiles.source_image_relpath.iloc[0]}. (Use --smoke-test-thumbnails only to test plumbing.)")
        if not root.is_dir():
            raise FullResImagesMissing(f"image_root does not exist: {root}")
        idx = image_index.resolve(tiles, root).set_index(["region", "tile_id"])
        idx = idx.reindex(list(zip(tiles.region, tiles.tile_id)))
        bad = idx[idx.status != "ok"]
        if len(bad):
            per = bad.groupby(level="region").status.value_counts().to_dict()
            raise FullResImagesMissing(
                f"{len(bad)}/{len(idx)} tile images not resolvable in {root} ({per}), e.g. {bad.filename.head(3).tolist()}. "
                "Run `python src/image_index.py` for the full integrity report.")
        # absolute locators: a file path, or '<zip path>::<member>'
        return pd.Series([str(root / loc) for loc in idx.locator], index=tiles.index)
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FullResImagesMissing(f"{len(missing)}/{len(paths)} tile images not found, e.g. {missing[:3]}")
    return paths


def augment(x: torch.Tensor) -> torch.Tensor:
    """Geospatially sensible augmentation: random 90-degree rotation and horizontal/vertical flips."""
    x = torch.rot90(x, int(torch.randint(0, 4, ())), dims=(1, 2))
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(2,))
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(1,))
    return x


class ParentTileDataset(Dataset):
    def __init__(self, tiles: pd.DataFrame, paths: pd.Series, input_size: int, train: bool,
                 target_col: str = "training_target"):
        self.keys = (tiles.region + "|" + tiles.tile_id).tolist()
        self.paths = [str(p) for p in paths]
        self.targets = tiles[target_col].fillna(-1).astype(np.float32).tolist() if target_col in tiles else [-1.0] * len(tiles)
        self.input_size = input_size
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.paths[i].endswith(".npy"):                                # pre-decoded 1280 px cache
            arr = np.load(self.paths[i])
            if arr.shape[0] != self.input_size:
                raise ValueError(f"decoded cache holds {arr.shape[0]} px tiles; input_size is {self.input_size}")
            x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
            if self.train:
                x = augment(x)
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            return x, torch.tensor(self.targets[i]), self.keys[i]
        with image_index.open_image(self.paths[i], REPO_ROOT) as im:    # absolute locators ignore the root
            im = im.convert("RGB")
            w = im.size[0]
            if im.size != (self.input_size, self.input_size):
                if w == im.size[1] and w % self.input_size == 0:
                    im = im.reduce(w // self.input_size)                    # exact box filter (antialiased, fast)
                else:
                    im = im.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR, reducing_gap=2.0)
            x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if self.train:
            x = augment(x)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, torch.tensor(self.targets[i]), self.keys[i]
