"""Train the ResNet-50 parent-tile classifier on one fold of a split manifest and score its test tiles.

Target: `training_target` from outputs/parent_tiles.csv (human layered-fencing signature; tiles with
exclude_from_training are dropped from train/val but still scored at test time). VLM scores are never read.

Examples
  python src/train.py --split saratov_spatial_cv --fold 0 --image-root /path/to/IMINT-Images
  python src/train.py --split loao --fold 2 --image-root /path/to/IMINT-Images
  python src/train.py --split saratov_spatial_cv --fold 0 --smoke-test-thumbnails --epochs 1 \
      --max-batches 2 --input-size 224 --no-pretrained      # PIPELINE_SMOKE_TEST_ONLY
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import OUTPUT_DIR, load_config
import decoded_cache
import image_index
from dataset import SANITY_LABEL, SMOKE_TEST_LABEL, FullResImagesMissing, ParentTileDataset, resolve_image_paths
from model import build_model, pick_device, weighted_bce


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Tile-level AP; used ONLY for checkpoint selection, never as a reported metric."""
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    hits = np.cumsum(y)
    return float((hits / np.arange(1, len(y) + 1))[y == 1].mean())


def load_fold(split: str, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pt_path, man_path = OUTPUT_DIR / "parent_tiles.csv", OUTPUT_DIR / "splits" / f"{split}.csv"
    for p, cmd in [(pt_path, "src/build_parent_table.py"), (man_path, "src/splits.py")]:
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run `python {cmd}` first")
    pt = pd.read_csv(pt_path)
    man = pd.read_csv(man_path)
    man = man[man.fold == fold]
    if man.empty:
        raise ValueError(f"fold {fold} not in {man_path}")
    return pt, man.merge(pt, on=["region", "tile_id"], validate="one_to_one")


def device_memory_gb(device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    return float("nan")


def weighted_bce_from_probs(y: np.ndarray, p: np.ndarray, pos_weight: float) -> float:
    """Same weighted BCE as training, computed from sigmoid outputs (monitoring only)."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.mean(-(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p))))


def run_out_dir(args) -> Path:
    if args.smoke_test_thumbnails:
        return OUTPUT_DIR / f"_{SMOKE_TEST_LABEL}" / f"{args.split}_fold{args.fold}"
    if args.sanity_check:
        return OUTPUT_DIR / f"_{SANITY_LABEL}" / f"{args.split}_fold{args.fold}"
    if args.out_dir:
        return Path(args.out_dir)
    return (Path(args.out_root) if args.out_root else OUTPUT_DIR / "runs") / f"{args.split}_fold{args.fold}"


@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    keys, scores, ys = [], [], []
    for x, y, k in loader:
        logit = model(x.to(device)).squeeze(1).float().cpu()
        keys += list(k); scores += torch.sigmoid(logit).tolist(); ys += y.tolist()
    reg, tile = zip(*(k.split("|") for k in keys)) if keys else ((), ())
    return pd.DataFrame({"region": reg, "tile_id": tile, "cnn_score": scores, "target": ys})


def main(argv=None) -> None:
    cfg = load_config()
    tc = cfg["training"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["saratov_spatial_cv", "loao"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--image-root", default=None, help="default: $CNN_BENCH_IMAGE_ROOT, then config image_root")
    ap.add_argument("--out-dir")
    ap.add_argument("--epochs", type=int, default=tc["epochs"])
    ap.add_argument("--batch-size", type=int, default=tc["batch_size"])
    ap.add_argument("--input-size", type=int, default=tc["input_size"])
    ap.add_argument("--lr", type=float, default=tc["lr"])
    ap.add_argument("--weight-decay", type=float, default=tc["weight_decay"])
    ap.add_argument("--num-workers", type=int, default=None,
                    help="default: training.num_workers (PNG) or training.num_workers_decoded_cache (--decoded-cache)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=cfg["splits"]["seed"])
    ap.add_argument("--smoke-test-thumbnails", action="store_true",
                    help=f"use 320px thumbnails; output is labelled {SMOKE_TEST_LABEL} and kept out of results")
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches per epoch (smoke tests)")
    ap.add_argument("--sanity-check", action="store_true",
                    help=f"full-res plumbing/runtime check: outputs labelled {SANITY_LABEL}, kept out of results")
    ap.add_argument("--allow-incomplete-cache", action="store_true",
                    help="permit full-res training while another AOI's images are unresolved (requires --sanity-check)")
    ap.add_argument("--out-root", default=None, help="parent dir for run outputs (default outputs/runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate guards, image/cache resolution, sample counts and dataloaders (one batch each), "
                         "then exit without training or writing outputs")
    ap.add_argument("--decoded-cache", action="store_true",
                    help="read the pre-decoded 1280 px cache (src/decoded_cache.py) instead of decoding PNGs")
    args = ap.parse_args(argv)

    if args.allow_incomplete_cache and not args.sanity_check:
        ap.error("--allow-incomplete-cache is only permitted together with --sanity-check")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    run_label = SMOKE_TEST_LABEL if args.smoke_test_thumbnails else SANITY_LABEL if args.sanity_check else "full_resolution"
    pt, d = load_fold(args.split, args.fold)
    if not args.smoke_test_thumbnails:
        full = image_index.resolve(pt, image_index.image_root_from(args.image_root))
        unresolved = (full.status != "ok").groupby(full.region).sum()
        if unresolved.any() and not args.allow_incomplete_cache:
            raise FullResImagesMissing(
                f"Full-resolution cache incomplete: unresolved parent tiles per AOI {unresolved.to_dict()}. Training is "
                "blocked until every benchmark parent resolves (see `python src/image_index.py`). "
                "Use --allow-incomplete-cache only for an explicitly labelled sanity check.")
    if args.num_workers is None:
        args.num_workers = tc["num_workers_decoded_cache"] if args.decoded_cache else tc["num_workers"]
    if args.decoded_cache and args.input_size != 1280:
        ap.error("--decoded-cache holds 1280 px tiles; use --input-size 1280")
    cache = decoded_cache.cache_dir_from(None) if args.decoded_cache else None
    paths = resolve_image_paths(d, args.image_root, args.smoke_test_thumbnails, cache)   # fails clearly if absent
    out = run_out_dir(args)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    d = d.assign(path=paths.values)
    usable = ~d.exclude_from_training.astype(bool)
    tr, va, te = d[(d.role == "train") & usable], d[(d.role == "val") & usable], d[d.role == "test"]
    n_pos, n_neg = int(tr.training_target.sum()), int((tr.training_target == 0).sum())
    if n_pos == 0:
        raise ValueError("no positive training tiles in this fold")
    device = pick_device(args.device)
    mk = lambda df, train, shuffle: DataLoader(
        ParentTileDataset(df, df.path, args.input_size, train), batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    dl_tr, dl_va, dl_te = mk(tr, True, True), mk(va, False, False), mk(te, False, False)

    if args.dry_run:
        report = dict(split=args.split, fold=args.fold, regions_by_role=d.groupby("role").region.unique()
                      .map(lambda r: sorted(r)).to_dict(), n_train=len(tr), n_train_pos=n_pos, n_val=len(va),
                      n_val_pos=int(va.training_target.sum()), n_test=len(te), n_buffer=int((d.role == "buffer").sum()),
                      n_excluded_from_training=int((~usable & d.role.isin(["train", "val"])).sum()),
                      image_source="decoded_cache_1280" if args.decoded_cache else "source_png",
                      n_paths_resolved=int(paths.notna().sum()), num_workers=args.num_workers)
        for name, dl in (("train", dl_tr), ("val", dl_va), ("test", dl_te)):
            x, y, k = next(iter(dl))
            report[f"{name}_batch"] = f"{tuple(x.shape)} {x.dtype}, targets {y.tolist()}"
        m = build_model(pretrained=False).to(device).eval()
        with torch.no_grad():
            report["forward_logits_shape"] = tuple(m(x.to(device)).shape)
        print(json.dumps(report, indent=1, default=str))
        print("DRY RUN OK: guards passed, all images resolved, dataloaders initialised; no training performed.")
        return
    model = build_model(pretrained=tc["pretrained"] and not args.no_pretrained).to(device)
    loss_fn = weighted_bce(n_neg / n_pos, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    meta = dict(run_label=run_label, split=args.split, fold=args.fold, n_train=len(tr), n_val=len(va), n_test=len(te),
                n_train_pos=n_pos, n_train_neg=n_neg, pos_weight=n_neg / n_pos, device=str(device),
                image_source="decoded_cache_1280" if args.decoded_cache else ("thumbnails" if args.smoke_test_thumbnails else "source_png"),
                args={k: v for k, v in vars(args).items()})
    (out / "run_config.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ["run_label", "split", "fold", "n_train", "n_val", "n_test", "n_train_pos", "device"]}))

    best, log = -np.inf, out / "metrics.jsonl"
    pw = n_neg / n_pos
    for epoch in range(args.epochs):
        t0 = time.time(); model.train(); losses = []; peak = 0.0
        for b, (x, y, _) in enumerate(dl_tr):
            if args.max_batches and b >= args.max_batches:
                break
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)).squeeze(1), y.to(device))
            loss.backward(); opt.step(); losses.append(loss.item())
            if b % 20 == 0:
                peak = max(peak, device_memory_gb(device))
            if b % 50 == 0:
                print(f"  epoch {epoch} batch {b}/{len(dl_tr)} loss {np.mean(losses[-50:]):.4f}", flush=True)
        sched.step()
        t_train = time.time() - t0
        v = predict(model, dl_va, device) if len(va) else pd.DataFrame(columns=["target", "cnn_score"])
        v_ap = average_precision(v.target.to_numpy(), v.cnn_score.to_numpy()) if len(v) else float("nan")
        q = max(len(losses) // 4, 1)
        rec = dict(epoch=epoch, train_loss=float(np.mean(losses)) if losses else None,
                   train_loss_first_quarter=float(np.mean(losses[:q])) if losses else None,
                   train_loss_last_quarter=float(np.mean(losses[-q:])) if losses else None,
                   val_loss_weighted=weighted_bce_from_probs(v.target.to_numpy(), v.cnn_score.to_numpy(), pw) if len(v) else None,
                   val_tile_ap=v_ap, train_seconds=round(t_train, 1), val_seconds=round(time.time() - t0 - t_train, 1),
                   seconds=round(time.time() - t0, 1), peak_device_memory_gb=round(peak, 2), run_label=run_label)
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec)
        torch.save({"model": model.state_dict(), "epoch": epoch, "meta": meta}, out / "last.pt")
        sel = v_ap if np.isfinite(v_ap) else -epoch   # fall back to latest epoch if val AP undefined
        if sel > best:
            best = sel
            torch.save({"model": model.state_dict(), "epoch": epoch, "meta": meta}, out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    pred = predict(model, dl_te, device).drop(columns="target")
    pred = pred.assign(split=args.split, fold=args.fold, run_label=run_label)
    pred.to_csv(out / "test_predictions.csv", index=False)
    print(f"Wrote {out / 'test_predictions.csv'} ({len(pred)} tiles) [{run_label}]")


if __name__ == "__main__":
    main()
