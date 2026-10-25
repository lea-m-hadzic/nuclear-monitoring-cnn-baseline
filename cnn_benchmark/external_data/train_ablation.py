"""Train ResNet-50 on one ablation condition/seed (external tiles only) with the repo's existing recipe.

Identical to src/train.py for model (torchvision ResNet-50, ImageNet V2 weights, single logit), loss (BCE with
pos_weight = n_neg / n_pos of the training split), optimiser (AdamW, lr/wd from configs/default.yaml), schedule
(cosine over epochs_max), augmentation (random 90-degree rotation + horizontal/vertical flips), input 1280 px and
ImageNet normalisation. Added here: tile-ID manifest input, the per-tile geometric watermark mask (filled before
augmentation), checkpoint selection / early stopping on the SHARED validation set (identical tiles for every condition;
PR-AUC, with AUROC and loss recorded), seeded data order, and run provenance (manifest / mask / val-set hashes).

    python train_ablation.py --condition A --seed 0                    # full run (NOT launched without approval)
    python train_ablation.py --condition A --seed 0 --dry-run batch    # one forward/backward batch, nothing saved
    python train_ablation.py --condition A --seed 0 --dry-run epoch    # one epoch + val, outputs under _DRY_RUN_ONLY
    python train_ablation.py --condition C --seed 0 --no-mask --robustness   # C_unmasked mask-sensitivity run
    python train_ablation.py --condition C_matched --seed 0 --robustness     # C_matched sampling-sensitivity run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))
from common import load_config  # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD, augment  # noqa: E402
from model import build_model, pick_device, weighted_bce  # noqa: E402
from train import average_precision, device_memory_gb  # noqa: E402
import freeze_ablation  # noqa: E402
from watermark_geometry import MaskBank  # noqa: E402
from watermark_mask import fill_rgb  # noqa: E402

DRY_LABEL = "DRY_RUN_ONLY"
RUNS_ROBUSTNESS = "outputs/ablation/runs_robustness"   # + "_<backend>" (mps / cuda): robustness runs; frozen runs/ untouched
# Full-resolution sensitivity (--input-px 2560; condition C only): same imagery decoded WITHOUT the 2x reduction, the same
# watermark rule rendered at 2560 with the fill sigma doubled (same physical smoothing). Frozen 1280 runs are untouched.
# The fill is precomputed once into *_filled caches (byte-identical to filling on the fly; see fullres_cache.py).
FULLRES = {2560: {"external_cache": "cache/external_decoded_2560px_filled", "benchmark_cache": "cache/decoded_2560px_filled",
                  "runs_dir": "outputs/ablation/runs_fullres_2560", "fill_sigma": 12.0, "run_label": "fullres_2560"}}


def ablation_config() -> dict:
    return yaml.safe_load((HERE / "ablation_config.yaml").read_text())


class NpyTileDataset(Dataset):
    """1280 px uint8 tiles from a decoded cache; optional mask fill BEFORE augmentation + normalisation.

    `mask` is None, one (1280, 1280) bool array for every tile, or a per-tile source with .get(tile_id) (MaskBank)."""

    def __init__(self, ids, labels, cache_dir: Path, mask: np.ndarray | None, train: bool, size: int = 1280,
                 fill_sigma: float = 6.0):
        self.paths = [str(cache_dir / f"{t}.npy") for t in ids]
        self.ids, self.labels, self.mask, self.train = list(ids), [float(x) for x in labels], mask, train
        self.size, self.fill_sigma = size, fill_sigma

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        arr = np.load(self.paths[i])
        if arr.shape != (self.size, self.size, 3):
            raise ValueError(f"{self.paths[i]}: {arr.shape}")
        if self.mask is not None:
            arr = fill_rgb(arr, self.mask.get(self.ids[i]) if isinstance(self.mask, MaskBank) else self.mask, self.fill_sigma)
        x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
        if self.train:
            x = augment(x)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, torch.tensor(self.labels[i]), self.ids[i]


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.min() == y.max():
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yy = y[o]
    return float((np.cumsum(yy) / np.arange(1, len(yy) + 1))[yy == 1].mean())


@torch.no_grad()
def predict(model, loader, device, loss_fn=None) -> tuple[pd.DataFrame, float]:
    """Scores for every tile, plus the mean per-tile loss (same weighted BCE as training) when loss_fn is given."""
    model.eval()
    ids, scores, ys, loss_sum = [], [], [], 0.0
    for x, y, k in loader:
        logit = model(x.to(device)).squeeze(1)
        if loss_fn is not None:
            loss_sum += float(loss_fn(logit, y.to(device))) * len(y)
        scores += torch.sigmoid(logit.float().cpu()).tolist()
        ys += y.tolist(); ids += list(k)
    return pd.DataFrame({"tile_id": ids, "score": scores, "label": ys}), loss_sum / max(len(ids), 1)


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    return g


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    tc, ac = load_config()["training"], ablation_config()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--condition", required=True, choices=["A", "B", "C", "C_matched"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--no-mask", action="store_true", help="mask-sensitivity runs (A_unmasked, C_unmasked) only")
    ap.add_argument("--robustness", action="store_true",
                    help="robustness matrix: outputs to runs_robustness_dir, inputs checked against freeze_robustness.json")
    ap.add_argument("--dry-run", choices=["batch", "epoch"], default=None)
    ap.add_argument("--input-px", type=int, choices=[1280, 2560], default=1280,
                    help="2560 = full-resolution sensitivity (condition C only); 1280 = every frozen run")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override only if memory forces it (documented in the sensitivity spec); default from config")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=tc["num_workers_decoded_cache"])
    a = ap.parse_args(argv)
    if a.no_mask and a.condition not in ("A", "C"):
        ap.error("--no-mask is only defined for the A_unmasked / C_unmasked sensitivity runs")
    if not a.dry_run and (a.no_mask or a.condition == "C_matched") and not a.robustness:
        ap.error("C_matched and unmasked runs belong to the robustness matrix: pass --robustness")
    fr = FULLRES.get(a.input_px)
    if fr and (a.condition != "C" or a.no_mask or a.robustness):
        ap.error("--input-px 2560 is the full-resolution sensitivity: condition C, masked, not --robustness")
    batch_size = a.batch_size or tc["batch_size"]

    man_path = PKG / ac["data"]["manifests_dir"] / f"{a.condition}.csv"
    man = pd.read_csv(man_path)
    cache = PKG / (fr["external_cache"] if fr else ac["data"]["external_cache"])
    if ac["watermark_mask"] not in ("geometric", "none"):
        raise SystemExit(f"watermark_mask must be geometric or none, got {ac['watermark_mask']!r}")
    mask = None if (a.no_mask or ac["watermark_mask"] == "none") else MaskBank(PKG / ac["data"]["mask_geometry"], n=a.input_px)
    fill_sigma = fr["fill_sigma"] if fr else 6.0
    sv_path = PKG / ac["data"]["shared_val"]
    va = pd.read_csv(sv_path)                                    # shared checkpoint-selection set, same for every condition
    name = a.condition + ("_unmasked" if a.no_mask else "")
    out = PKG / (ac["dry_run_dir"] if a.dry_run else fr["runs_dir"] if fr else f"{RUNS_ROBUSTNESS}_{pick_device(a.device).type}" if a.robustness else ac["runs_dir"]) / name / f"seed{a.seed}"
    g = seed_all(a.seed)
    device = pick_device(a.device)
    if device.type == "cuda":                     # same fp32 arithmetic as the frozen MPS runs: no TF32 shortcuts
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tr = man[man.split == "train"]
    if set(va.tile_id) & set(tr.tile_id) or set(va.split_group_id) & set(tr.split_group_id) or (va.split != "val").any():
        raise SystemExit("shared validation set overlaps this condition's training split")
    missing = [t for t in list(tr.tile_id) + list(va.tile_id) if not (cache / f"{t}.npy").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} tiles missing from {cache}, e.g. {missing[:3]}")
    n_pos, n_neg = int(tr.label.sum()), int((tr.label == 0).sum())
    ds_mask = None if fr else mask                                # fullres: geometric fill already applied in the cache
    mk = lambda df, train: DataLoader(NpyTileDataset(df.tile_id, df.label, cache, ds_mask, train, a.input_px, fill_sigma), batch_size=batch_size,
                                      shuffle=train, generator=g if train else None, num_workers=a.num_workers)
    dl_tr, dl_va = mk(tr, True), mk(va, False)
    model = build_model(pretrained=tc["pretrained"]).to(device)
    loss_fn = weighted_bce(n_neg / n_pos, device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs = 1 if a.dry_run == "epoch" else ac["epochs_max"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ac["epochs_max"])
    frozen = None if a.dry_run else freeze_ablation.check()      # main runs: inputs must match the freeze record
    frozen_rob = freeze_ablation.check_robustness() if (a.robustness and not a.dry_run) else None
    meta = {"run_label": DRY_LABEL if a.dry_run else fr["run_label"] if fr else "ablation", "condition": name, "seed": a.seed,
            "input_px": a.input_px, "mask_fill_sigma_px": fill_sigma, "mask_prefilled_in_cache": bool(fr),
            "external_cache_manifest_sha256": sha(cache / "cache_manifest.csv") if fr else None,
            "freeze": frozen, "freeze_robustness": frozen_rob, "git_at_launch": freeze_ablation.git_state(),
            "backend": {"torch": str(torch.__version__), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda"
                        else str(device), "tf32": False if device.type == "cuda" else None, "precision": "fp32"},
            "manifest": str(man_path.relative_to(PKG)), "manifest_sha256": sha(man_path),
            "watermark_mask": "none" if mask is None else "geometric",
            "watermark_template_sha256": None if mask is None else mask.template_sha256,
            "watermark_geometry_sha256": None if mask is None else mask.table_sha256,
            "shared_val": str(sv_path.relative_to(PKG)), "shared_val_sha256": sha(sv_path),
            "n_train": len(tr), "n_train_pos": n_pos, "n_train_neg": n_neg,
            "n_val": len(va), "n_val_pos": int(va.label.sum()), "val_roles": va.role.value_counts().to_dict(),
            "pos_weight": n_neg / n_pos,
            "training": {k: tc[k] for k in ("input_size", "batch_size", "lr", "weight_decay", "pretrained")}
                        | ({"input_size": a.input_px, "batch_size": batch_size} if fr else {}),
            "ablation": {k: ac[k] for k in ("epochs_max", "early_stopping", "watermark_mask")}, "device": str(device)}

    if a.dry_run == "batch":
        t0 = time.time()
        x, y, _ = next(iter(dl_tr))
        t_load = time.time() - t0
        model.train(); opt.zero_grad()
        loss = loss_fn(model(x.to(device)).squeeze(1), y.to(device)); loss.backward(); opt.step()
        meta.update(batch_shape=list(x.shape), batch_targets=y.tolist(), loss=float(loss), batch_load_s=round(t_load, 2),
                    batch_total_s=round(time.time() - t0, 2), input_mean=round(float(x.mean()), 4))
        print(json.dumps(meta, indent=1, default=str))
        print("DRY RUN (batch) OK: nothing saved")
        return meta

    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(meta, indent=1, default=str))
    es = ac["early_stopping"]
    best, best_epoch, stale = -np.inf, -1, 0
    for epoch in range(epochs):
        t0 = time.time(); model.train(); losses = []; peak = 0.0
        for b, (x, y, _) in enumerate(dl_tr):
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)).squeeze(1), y.to(device))
            loss.backward(); opt.step(); losses.append(loss.item())
            if b % 20 == 0:
                peak = max(peak, device_memory_gb(device))
        sched.step()
        t_train = time.time() - t0
        v, v_loss = predict(model, dl_va, device, loss_fn)
        v_ap = average_precision(v.label.to_numpy(), v.score.to_numpy())
        v_auc = auroc(v.label.to_numpy().astype(int), v.score.to_numpy())
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "shared_val_pr_auc": v_ap, "shared_val_auroc": v_auc,
               "shared_val_loss": v_loss, "train_seconds": round(t_train, 1),
               "val_seconds": round(time.time() - t0 - t_train, 1), "peak_device_memory_gb": round(peak, 2),
               "run_label": meta["run_label"]}
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if v_ap > best:
            best, best_epoch, stale, best_rec = v_ap, epoch, 0, rec
            torch.save({"model": model.state_dict(), "epoch": epoch, "meta": meta}, out / "best.pt")
            v.to_csv(out / "val_predictions_best.csv", index=False)
        else:
            stale += 1
        if epoch + 1 >= es["min_epochs"] and stale >= es["patience"]:
            break
    summary = {"best_epoch": best_epoch, "best_shared_val_pr_auc": best, "best_shared_val_auroc": best_rec["shared_val_auroc"],
               "best_shared_val_loss": best_rec["shared_val_loss"], "epochs_run": epoch + 1, "run_label": meta["run_label"]}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)
    return summary


if __name__ == "__main__":
    main()
