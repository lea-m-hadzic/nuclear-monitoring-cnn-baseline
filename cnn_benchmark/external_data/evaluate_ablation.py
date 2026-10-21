"""Score the held-out benchmark AOIs with an ablation checkpoint and evaluate it (no training on these AOIs).

    python evaluate_ablation.py --run outputs/ablation/runs/A/seed0                # all 2,940 benchmark parent tiles
    python evaluate_ablation.py --run ... --scores-only                            # scoring only
    python evaluate_ablation.py --run ... --max-tiles 16 --dry-run                 # plumbing check, not a result

Per run (written to <run>/benchmark/):
  scores.csv                     region, tile_id, score, human_target  (same per-tile geometric mask + fill as training)
  facility_<region>_*.csv        HEADLINE: facility retrieval vs review budget (src/evaluation.py footprint rule,
                                 provisional facility map), recall targets 50/75/90/100 %
  summary.json                   per-AOI + aggregate: tile AUROC, tile PR-AUC, review area to surface known facilities,
                                 score quantiles for human positives / negatives
Aggregate = pooled over all tiles of the three AOIs (tile metrics) and per-AOI means; facility retrieval stays per AOI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))
import facility_mapping as fm  # noqa: E402
from common import OUTPUT_DIR, load_config  # noqa: E402
from evaluation import evaluate_parent_scores  # noqa: E402
from model import build_model, pick_device  # noqa: E402
from train_ablation import NpyTileDataset, ablation_config, auroc, pr_auc  # noqa: E402,F401
from watermark_geometry import MaskBank  # noqa: E402


def score_benchmark(run: Path, max_tiles: int | None, device_name: str) -> pd.DataFrame:
    cfg, ac = load_config(), ablation_config()
    # Safe (weights_only) loading; the only extra class allowed is TorchVersion, which C_matched seed 0 (robustness matrix) stored
    # in its metadata as torch.__version__; new checkpoints store a plain string.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    meta = ck["meta"]
    mask = None
    if meta.get("watermark_mask") == "geometric":           # same template, geometry table and fill as training
        mask = MaskBank(PKG / ac["data"]["mask_geometry"])
        if (mask.template_sha256, mask.table_sha256) != (meta["watermark_template_sha256"], meta["watermark_geometry_sha256"]):
            raise SystemExit("watermark template / geometry table differ from the ones used in training")
    elif meta.get("watermark_mask") != "none":
        raise SystemExit("checkpoint has no geometric-mask provenance (old fixed-mask run?)")
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    pt = pt[pt.region.isin(ac["evaluation"]["regions"])].reset_index(drop=True)
    cache = PKG / ac["data"]["benchmark_cache"]
    ids = pt.region + "__" + pt.tile_id
    missing = [i for i in ids if not (cache / f"{i}.npy").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} benchmark tiles missing from {cache}, e.g. {missing[:3]}")
    if max_tiles:
        pt, ids = pt.head(max_tiles), ids.head(max_tiles)
    device = pick_device(device_name)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    dl = DataLoader(NpyTileDataset(ids, pt[ac["evaluation"]["tile_label"]].fillna(0), cache, mask, train=False),
                    batch_size=load_config()["training"]["batch_size"])
    scores = []
    with torch.no_grad():
        for x, _, _ in dl:
            scores += torch.sigmoid(model(x.to(device)).squeeze(1).float().cpu()).tolist()
    return pt[["region", "tile_id", ac["evaluation"]["tile_label"]]].assign(score=scores, run_label=meta["run_label"],
                                                                            condition=meta["condition"], seed=meta["seed"])


def evaluate(s: pd.DataFrame, out: Path) -> dict:
    cfg, ac = load_config(), ablation_config()
    lab = ac["evaluation"]["tile_label"]
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    facilities, assignments = fm.reconstruct(cfg)
    mem = fm.parent_membership(assignments)
    res = {"regions": {}, "tile_label": lab, "facility_map": "PROVISIONAL (see facility_mapping_questions.md)"}
    for region in ac["evaluation"]["regions"]:
        r = s[s.region == region]
        y, sc = r[lab].fillna(0).astype(int).to_numpy(), r.score.to_numpy()
        fr = evaluate_parent_scores(s, pt, facilities, mem, region, aoi_area_km2=cfg["geometry"]["aoi_area_km2_nominal"][region],
                                    recall_targets=cfg["evaluation"]["recall_targets"])
        fr.curve.to_csv(out / f"facility_{region}_curve.csv", index=False)
        fr.first_hits.to_csv(out / f"facility_{region}_first_hits.csv", index=False)
        fr.recall_targets.to_csv(out / f"facility_{region}_recall_targets.csv", index=False)
        res["regions"][region] = {
            "n_tiles": int(len(r)), "n_human_positive": int(y.sum()), "tile_auroc": auroc(y, sc), "tile_pr_auc": pr_auc(y, sc),
            "facility": dict(fr.summary, recall_targets=fr.recall_targets.to_dict("records")),
            "score_quantiles": {k: np.round(r.score[(r[lab] == v)].quantile([.1, .5, .9]).to_numpy(), 4).tolist()
                                for k, v in (("human_positive", 1), ("human_negative", 0))}}
    y, sc = s[lab].fillna(0).astype(int).to_numpy(), s.score.to_numpy()
    res["aggregate"] = {"pooled_tile_auroc": auroc(y, sc), "pooled_tile_pr_auc": pr_auc(y, sc),
                        "mean_tile_auroc": float(np.nanmean([v["tile_auroc"] for v in res["regions"].values()])),
                        "mean_tile_pr_auc": float(np.nanmean([v["tile_pr_auc"] for v in res["regions"].values()]))}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="plumbing check (partial scoring + random-score facility eval)")
    ap.add_argument("--scores-only", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    run = a.run if a.run.is_absolute() else PKG / a.run
    out = run / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    s = score_benchmark(run, a.max_tiles, a.device)
    if a.dry_run:
        print(s.head().to_string(index=False))
        full = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
        full = full[full.region.isin(ablation_config()["evaluation"]["regions"])]
        rnd = full[["region", "tile_id", ablation_config()["evaluation"]["tile_label"]]].assign(
            score=np.random.default_rng(0).random(len(full)))
        res = evaluate(rnd, out)
        print("random-score plumbing check (NOT a result):", json.dumps(res["aggregate"]))
        return res
    s.to_csv(out / "scores.csv", index=False)
    if a.scores_only:
        return None
    res = evaluate(s, out)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["aggregate"], indent=1))
    return res


if __name__ == "__main__":
    main()
