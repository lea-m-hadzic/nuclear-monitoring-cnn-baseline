"""Measure loader-only and end-to-end training throughput (s/img) for source PNGs vs the decoded cache.

  python src/throughput.py --source png
  python src/throughput.py --source cache
Same fixed sample of Saratov fold-0 training tiles, 1280 px, batch 4, ResNet-50 (random init; weights do not
affect speed). Results are appended to outputs/throughput.csv.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

import decoded_cache
from common import OUTPUT_DIR
from dataset import ParentTileDataset, resolve_image_paths
from model import build_model, pick_device
from train import load_fold


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["png", "cache"], required=True)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 6])
    args = ap.parse_args(argv)
    _, d = load_fold("saratov_spatial_cv", 0)
    tiles = d[(d.role == "train") & ~d.exclude_from_training].sample(args.n, random_state=0)
    cache = decoded_cache.cache_dir_from(None) if args.source == "cache" else None
    paths = resolve_image_paths(tiles, None, decoded_cache=cache)
    rows = []
    for nw in args.workers:
        dl = DataLoader(ParentTileDataset(tiles, paths, 1280, True), batch_size=4, num_workers=nw, shuffle=True)
        t = time.time(); n = 0
        for x, _, _ in dl:
            n += len(x)
        rows.append(dict(source=args.source, test="loader_only", workers=nw, s_per_img=(time.time() - t) / n))
    dev = pick_device("auto")
    m = build_model(pretrained=False).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-4)
    lf = torch.nn.BCEWithLogitsLoss()
    nw = max(args.workers)
    dl = DataLoader(ParentTileDataset(tiles, paths, 1280, True), batch_size=4, num_workers=nw, shuffle=True)
    for epoch in range(2):                       # epoch 0 includes warm-up / worker start
        t = time.time(); n = 0; tg = 0.0
        for x, y, _ in dl:
            t1 = time.time()
            opt.zero_grad(); lf(m(x.to(dev)).squeeze(1), y.to(dev)).backward(); opt.step()
            if dev.type == "mps":
                torch.mps.synchronize()
            tg += time.time() - t1; n += len(x)
        tot = time.time() - t
        rows.append(dict(source=args.source, test=f"train_end_to_end_epoch{epoch}", workers=nw,
                         s_per_img=tot / n, gpu_step_share=tg / tot))
    r = pd.DataFrame(rows).assign(n_tiles=args.n, device=str(dev))
    print(r.round(3).to_string(index=False))
    out = OUTPUT_DIR / "throughput.csv"
    r.to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
