"""SUPERSEDED for the ablation (see watermark_geometry.py): the old fixed union mask is kept only for provenance;
fill_rgb below is still THE fill for every mask.

The shared watermark mask, persisted, plus the RGB fill used for every training and evaluation image.

The mask is NOT redefined here: it is rebuilt with src/artifact_audit.py's own functions and parameters
(same tile sample: --n 120, --seed 20261008 per AOI; per-AOI frequency of high-pass pixels > HP_T; threshold 0.5;
binary dilation 3; union of Saratov | Fordow | Natanz) and saved once, with its SHA-256 and area, so every
run uses the identical array. Fill = artifact_audit.fill_masked (normalized-convolution Gaussian, sigma 6), applied
to each RGB channel independently, on the 1280 px uint8 image BEFORE augmentation and normalization.

    python watermark_mask.py build      # writes cache/watermark_union_mask_1280.npy (+ .json)
    python watermark_mask.py coverage   # how much of the external tiles' watermark rows the mask covers
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import artifact_audit as aa  # noqa: E402
import image_index  # noqa: E402
from common import OUTPUT_DIR, PACKAGE_DIR  # noqa: E402
from scipy import ndimage  # noqa: E402

MASK_PATH = PACKAGE_DIR / "cache" / "watermark_union_mask_1280.npy"
AUDIT_N, AUDIT_SEED, AUDIT_THRESH, AUDIT_DILATION = 120, 20261008, 0.5, 3   # artifact_audit.py defaults used for the report
REGIONS = ["Saratov", "Fordow", "Natanz"]


def build() -> dict:
    root = image_index.image_root_from(None)
    idx = pd.read_csv(OUTPUT_DIR / "image_index.csv")
    idx = idx[idx.status == "ok"]
    sample = pd.concat([g.sample(min(AUDIT_N, len(g)), random_state=AUDIT_SEED) for _, g in idx.groupby("region")])
    sample = sample[sample.region.isin(REGIONS)]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(aa._tile_stats, [str(root / l) for l in sample.locator], chunksize=4))
    freq = {r: np.mean([x[0] for x, reg in zip(res, sample.region) if reg == r], axis=0) for r in REGIONS}
    masks = {r: ndimage.binary_dilation(freq[r] > AUDIT_THRESH, iterations=AUDIT_DILATION) for r in REGIONS}
    union = masks["Saratov"] | masks["Fordow"] | masks["Natanz"]
    MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(MASK_PATH, union)
    info = {"shape": list(union.shape), "area_pct": round(100 * float(union.mean()), 2),
            "per_aoi_area_pct": {r: round(100 * float(masks[r].mean()), 2) for r in REGIONS},
            "sha256": hashlib.sha256(union.tobytes()).hexdigest(), "source": "src/artifact_audit.py functions",
            "params": {"n": AUDIT_N, "seed": AUDIT_SEED, "thresh": AUDIT_THRESH, "dilation": AUDIT_DILATION,
                       "hp_sigma": aa.HP_SIGMA, "hp_t": aa.HP_T, "size": aa.SIZE}}
    MASK_PATH.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return info


def load_mask() -> tuple[np.ndarray, str]:
    m = np.load(MASK_PATH)
    sha = hashlib.sha256(m.tobytes()).hexdigest()
    expected = json.loads(MASK_PATH.with_suffix(".json").read_text())["sha256"]
    if sha != expected:
        raise RuntimeError(f"watermark mask changed on disk ({sha[:12]} != {expected[:12]})")
    return m, sha


def fill_rgb(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """artifact_audit.fill_masked applied per channel; uint8 in, uint8 out."""
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        out[..., c] = np.clip(np.rint(aa.fill_masked(arr[..., c].astype(np.float32), mask)), 0, 255).astype(np.uint8)
    return out


def coverage() -> dict:
    """Share of each external tile's predicted watermark strip rows that fall inside the fixed mask.

    auto_mosaic crops a centred `crop_px` window out of grid x 1280 source pixels; the Google logo/attribution strip
    sits at the bottom of every 1280 px source image, i.e. at output rows (k*1280 - off) * 1280 / crop_px for the
    1280 px input (off = (grid*1280 - crop_px) / 2)."""
    m, _ = load_mask()
    row_cov = m.any(axis=1)                                   # rows where the fixed mask has any pixel
    from geometry import REPO_ROOT, load_config
    d = REPO_ROOT / load_config()["dataset"]["dataset_dir"]
    meta = pd.concat([pd.read_csv(d / "external_tiles_metadata_refined.csv")] +
                     [pd.read_csv(f) for f in sorted(d.glob("hard_negative_metadata_private_*.csv"))], ignore_index=True)
    meta = meta.dropna(subset=["crop_px", "mosaic_grid"]).drop_duplicates("tile_id")
    rows = []
    for r in meta.itertuples():
        grid, crop = int(r.mosaic_grid), float(r.crop_px)
        off = (grid * 1280 - crop) / 2
        ys = [(k * 1280 - off) * 1280 / crop for k in range(1, grid + 1)]
        ys = [y for y in ys if 4 <= y <= 1280]
        hit = [bool(row_cov[max(int(y) - 10, 0):min(int(y), 1280)].mean() > 0.5) for y in ys]
        rows.append({"tile_id": r.tile_id, "n_strips": len(ys), "covered": sum(hit)})
    c = pd.DataFrame(rows)
    return {"tiles": int(len(c)), "strips": int(c.n_strips.sum()), "strips_covered": int(c.covered.sum()),
            "pct_strips_covered": round(100 * c.covered.sum() / max(c.n_strips.sum(), 1), 1),
            "tiles_fully_covered": int((c.covered == c.n_strips).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "coverage"])
    a = ap.parse_args()
    print(json.dumps(build() if a.cmd == "build" else coverage(), indent=1))


if __name__ == "__main__":
    main()
