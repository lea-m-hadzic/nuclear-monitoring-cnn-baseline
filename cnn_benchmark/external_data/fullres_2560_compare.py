"""Full-resolution (2560 px) sensitivity vs frozen C-1280: the pre-registered comparison (reads outputs only).

    python fullres_2560_compare.py      # -> results/robustness/fullres_2560/

Pre-registration: results/robustness/fullres_2560/FULLRES_2560_SPEC.md (metrics and materiality rule fixed there).
Inputs: frozen C-1280 = provenance/run_records/runs/C/seed{0,1,2}/; 2560 = provenance/run_records/fullres_2560/C/seed{0,1,2}/
(copies of outputs/ablation/runs_fullres_2560/C/, without checkpoints; the interrupted first seed-2 attempt is excluded).
Facility metrics come from each run's src/evaluation.py outputs (footprint rule, expected rank under ties); area to a
recall target not in the evaluator's recall_targets file (25 %) is read off the same curve, and the curve reading is
checked against the evaluator's own 50/75/90 % values.
Outputs: seed_level.csv, summary.csv, comparison_1280_vs_2560.csv, rank_overlap.csv, materiality.csv,
fig_fullres_2560_recall_vs_review_area.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from robustness_tables import run_metrics  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
RUNS = {"1280": REPO / "provenance" / "run_records" / "runs" / "C",
        "2560": REPO / "provenance" / "run_records" / "fullres_2560" / "C"}
OUT = REPO / "results" / "robustness" / "fullres_2560"
SEEDS = (0, 1, 2)
WHS = ("SAR-WHS-01", "SAR-WHS-02", "SAR-WHS-03")
METRICS = ["pooled_tile_auroc", "pooled_tile_pr_auc",
           "saratov_recall_at_10pct", "saratov_recall_at_20pct", "saratov_recall_at_30pct",
           "saratov_area_to_25pct", "saratov_area_to_50pct", "saratov_area_to_75pct", "saratov_area_to_90pct",
           *[f"first_hit_{w}" for w in WHS], "natanz_first_hit_pct", "fordow_first_hit_pct",
           "best_epoch", "epochs_run", "best_shared_val_pr_auc", "best_shared_val_auroc", "train_loss_at_best"]


def recall_at(curve: pd.DataFrame, pct: float) -> float:
    return float(curve[curve.pct_aoi_reviewed <= pct + 1e-9].facility_recall.iloc[-1])


def area_to(curve: pd.DataFrame, target: float) -> float:
    hit = curve[curve.facility_recall >= target - 1e-12]
    return float(hit.pct_aoi_reviewed.iloc[0]) if len(hit) else float("nan")


def seed_row(res: str, s: int) -> dict:
    d = RUNS[res] / f"seed{s}"
    b = d / "benchmark"
    curve = pd.read_csv(b / "facility_Saratov_curve.csv")
    rt = pd.read_csv(b / "facility_Saratov_recall_targets.csv").set_index("recall_target")
    for t in (0.5, 0.75, 0.9):                                   # curve reading == evaluator's own value
        assert abs(area_to(curve, t) - rt.loc[t, "pct_aoi_reviewed"]) < 1e-9, (res, s, t)
    fh = pd.read_csv(b / "facility_Saratov_first_hits.csv").set_index("facility_id")
    ts = json.loads((d / "train_summary.json").read_text())
    log = pd.read_json(d / "metrics.jsonl", lines=True).set_index("epoch")
    rm = run_metrics(d)
    return {"resolution": res, "seed": s, **{k: rm[k] for k in ("pooled_tile_auroc", "pooled_tile_pr_auc",
                                                                 "natanz_first_hit_pct", "fordow_first_hit_pct")},
            "saratov_recall_at_10pct": recall_at(curve, 10), "saratov_recall_at_20pct": recall_at(curve, 20),
            "saratov_recall_at_30pct": recall_at(curve, 30),
            **{f"saratov_area_to_{int(t * 100)}pct": area_to(curve, t) for t in (0.25, 0.5, 0.75, 0.9)},
            **{f"first_hit_{w}": float(fh.loc[w, "first_hit_pct_aoi_expected"]) for w in WHS},
            "best_epoch": ts["best_epoch"], "epochs_run": ts["epochs_run"],
            "best_shared_val_pr_auc": ts["best_shared_val_pr_auc"], "best_shared_val_auroc": ts["best_shared_val_auroc"],
            "train_loss_at_best": float(log.loc[ts["best_epoch"], "train_loss"])}


def overlap(a: pd.Series, b: pd.Series, frac: float) -> float:
    k = max(1, int(round(frac * len(a))))
    return len(set(a.nlargest(k).index) & set(b.nlargest(k).index)) / k


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed = pd.DataFrame([seed_row(r, s) for r in RUNS for s in SEEDS])
    seed.to_csv(OUT / "seed_level.csv", index=False)

    summ = seed.groupby("resolution")[METRICS].agg(["mean", "std"])
    summ.columns = [f"{m}_{a if a == 'mean' else 'sd'}" for m, a in summ.columns]
    summ.reset_index().to_csv(OUT / "summary.csv", index=False)

    lo, hi = seed[seed.resolution == "1280"].set_index("seed"), seed[seed.resolution == "2560"].set_index("seed")
    comp = []
    for m in METRICS:
        dlt = hi[m] - lo[m]
        comp.append({"metric": m, "c1280_mean": lo[m].mean(), "c1280_sd": lo[m].std(ddof=1),
                     "c2560_mean": hi[m].mean(), "c2560_sd": hi[m].std(ddof=1), "delta_mean": dlt.mean(),
                     **{f"delta_seed{s}": dlt[s] for s in SEEDS}})
    pd.DataFrame(comp).to_csv(OUT / "comparison_1280_vs_2560.csv", index=False)

    rows = []
    for s in SEEDS:
        sc = {r: pd.read_csv(RUNS[r] / f"seed{s}" / "benchmark" / "scores.csv").set_index(["region", "tile_id"]).score
              for r in RUNS}
        assert sc["1280"].index.equals(sc["2560"].index) or set(sc["1280"].index) == set(sc["2560"].index)
        b = sc["2560"].reindex(sc["1280"].index)
        for scope, sel in (("all 2,940 tiles", slice(None)), ("Saratov", "Saratov")):
            a_, b_ = (sc["1280"], b) if scope.startswith("all") else (sc["1280"].loc[sel], b.loc[sel])
            rows.append({"seed": s, "scope": scope, "n_tiles": len(a_), "spearman": spearmanr(a_, b_).statistic,
                         **{f"top{int(f * 100)}pct_overlap": overlap(a_, b_, f) for f in (0.01, 0.05, 0.10, 0.20)}})
    pd.DataFrame(rows).to_csv(OUT / "rank_overlap.csv", index=False)

    # Materiality rule (spec, fixed before training): |delta recall@20%| >= 0.10 OR |delta area-to-50%| >= 5 pp,
    # with the same direction in >= 2 of 3 seeds (per-seed delta = 2560 minus frozen 1280 of the same seed).
    mat = []
    for m, thr in (("saratov_recall_at_20pct", 0.10), ("saratov_area_to_50pct", 5.0)):
        dlt = hi[m] - lo[m]
        up, down = int((dlt >= thr).sum()), int((dlt <= -thr).sum())
        better = "up" if m.startswith("saratov_recall") else "down"
        material = up >= 2 or down >= 2
        direction = "" if not material else ("better" if (up >= 2) == (better == "up") else "worse")
        mat.append({"criterion": m, "threshold": thr, **{f"delta_seed{s}": dlt[s] for s in SEEDS},
                    "n_seeds_ge_threshold_up": up, "n_seeds_ge_threshold_down": down, "material": material,
                    "direction": direction})
    mat = pd.DataFrame(mat)
    mat.to_csv(OUT / "materiality.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    grid = np.linspace(0, 100, 1001)
    for res, col in (("1280", "#4a6fa5"), ("2560", "#c8553d")):
        ys = []
        for s in SEEDS:
            c = pd.read_csv(RUNS[res] / f"seed{s}" / "benchmark" / "facility_Saratov_curve.csv")
            y = np.array([recall_at(c, g) for g in grid]); ys.append(y)
            ax.step(grid, y, where="post", color=col, alpha=0.3, lw=0.9)
        ax.step(grid, np.mean(ys, axis=0), where="post", color=col, lw=2,
                label=f"C-{res} (mean of 3 seeds; thin = per seed)")
    ax.plot([0, 100], [0, 1], ls=":", color="grey", lw=1, label="random review")
    ax.axvline(20, color="grey", lw=0.6, ls="--")
    ax.set(xlabel="% of Saratov AOI reviewed", ylabel="Saratov facility recall", xlim=(0, 100), ylim=(0, 1.02),
           title="Input resolution sensitivity: condition C, 1280 vs 2560 px")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fullres_2560_recall_vs_review_area.png", dpi=200)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(comp).round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(mat.to_string(index=False))


if __name__ == "__main__":
    main()
