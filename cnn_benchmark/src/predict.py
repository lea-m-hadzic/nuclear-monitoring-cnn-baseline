"""Inference export and out-of-fold assembly.

  # score arbitrary tiles with a checkpoint
  python src/predict.py score --checkpoint outputs/runs/saratov_spatial_cv_fold0/best.pt \
      --regions Saratov --image-root /path/to/IMINT-Images --out scores.csv
  # combine per-fold test predictions into one score file (each tile scored exactly once)
  python src/predict.py assemble --runs-dir outputs/runs --split saratov_spatial_cv --out cnn_scores_saratov_cv.csv

Output columns: region, tile_id, cnn_score (+ provenance columns).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import OUTPUT_DIR, load_config
import decoded_cache
from dataset import SMOKE_TEST_LABEL, ParentTileDataset, resolve_image_paths
from model import build_model, pick_device
from train import predict


def score(args) -> None:
    cfg = load_config()
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    pt = pt[pt.region.isin(args.regions)]
    cache = decoded_cache.cache_dir_from(None) if args.decoded_cache else None
    paths = resolve_image_paths(pt, args.image_root, args.smoke_test_thumbnails, cache)
    device = pick_device(args.device)
    ck = torch.load(args.checkpoint, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    size = args.input_size or ck["meta"]["args"]["input_size"]
    dl = DataLoader(ParentTileDataset(pt, paths, size, train=False), batch_size=args.batch_size)
    out = predict(model, dl, device).drop(columns="target")
    # propagate any non-scientific label of the checkpoint (smoke test / sanity check) to its predictions
    label = SMOKE_TEST_LABEL if args.smoke_test_thumbnails else ck["meta"].get("run_label", "full_resolution")
    out.assign(checkpoint=str(args.checkpoint), run_label=label).to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} tiles) [{label}]")


def assemble(args) -> None:
    runs = sorted(Path(args.runs_dir).glob(f"{args.split}_fold*/test_predictions.csv"))
    if not runs:
        raise FileNotFoundError(f"no {args.split}_fold*/test_predictions.csv under {args.runs_dir}")
    d = pd.concat([pd.read_csv(p) for p in runs], ignore_index=True)
    dup = d.duplicated(["region", "tile_id"])
    if dup.any():
        raise ValueError(f"{dup.sum()} tiles scored by more than one fold (expected out-of-fold scores)")
    labels = set(d.run_label)
    if len(labels) > 1:
        raise ValueError(f"mixed run labels {labels}")
    d.to_csv(args.out, index=False)
    print(f"Assembled {len(runs)} folds, {len(d)} tiles, run_label={labels.pop()} -> {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--regions", nargs="+", default=load_config()["regions"])
    s.add_argument("--image-root", default=None)
    s.add_argument("--smoke-test-thumbnails", action="store_true")
    s.add_argument("--decoded-cache", action="store_true", help="read the pre-decoded 1280 px cache")
    s.add_argument("--input-size", type=int)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--device", default="auto")
    s.add_argument("--out", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--runs-dir", default=str(OUTPUT_DIR / "runs"))
    a.add_argument("--split", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    score(args) if args.cmd == "score" else assemble(args)


if __name__ == "__main__":
    main()
