"""Aggregate the main ablation matrix (A/B/C x seeds 0-2) into paper tables and figures. Reads run outputs only.

    python aggregate_ablation.py      # -> results/main/abc_tables/{*.csv, *.md, *.png}

Tables (mean +/- sample std over seeds; per-seed values kept in the long CSVs):
  T1 checkpoint selection    best epoch, shared-val PR-AUC / AUROC / loss
  T2 tile discrimination     benchmark tile AUROC / PR-AUC per AOI, pooled over all tiles, and mean of AOIs
  T3 facility recall at equal review budget (% of AOI area reviewed, 1/2/5/10/20/30 %), per AOI + pooled
                             (pooled = facilities surfaced in all AOIs when each AOI is reviewed to the same %)
  T4 review area (% AOI) to reach 50/75/90/100 % facility recall, per AOI
  T5 review area (% AOI, expected under ties) to surface each known facility
  Random-ordering reference: 200 random score permutations through the same facility evaluation.
Figures: shared-val training curves; facility recall vs review budget per AOI; tile metrics per AOI with seeds;
per-facility first-hit review area.
Reversal flag: an AOI whose condition ranking on a metric differs from the pooled ranking, with the size of the
gap relative to the seed std.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))
import facility_mapping as fm  # noqa: E402
from common import OUTPUT_DIR, load_config  # noqa: E402
from evaluation import evaluate_parent_scores  # noqa: E402
from train_ablation import ablation_config, auroc, pr_auc  # noqa: E402

AC = ablation_config()
RUNS = PKG / AC["runs_dir"]
OUT = PKG.parent / "results" / "main" / "abc_tables"   # repo-level results/ (paper-facing); figures/ holds renamed copies
CONDS, SEEDS, REGIONS = AC["conditions"], AC["seeds"], AC["evaluation"]["regions"]
BUDGETS = [1, 2, 5, 10, 20, 30]
COLORS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "random": "#7f7f7f"}
LABEL = {"A": "A nearby", "B": "B hard", "C": "C mixed"}


def ms(x) -> str:
    x = pd.Series(x).dropna()
    return "n/a" if x.empty else f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.iloc[0]:.3f}"


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    head = [str(c) for c in d.columns]
    rows = [[str(v) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def recall_at(curve: pd.DataFrame, budget: float) -> tuple[int, float]:
    ok = curve[curve.pct_aoi_reviewed <= budget + 1e-9]
    r = ok.iloc[-1]
    return int(r.facilities_surfaced), float(r.facility_recall)


def load_runs():
    rows, curves, hits, epochs, rts = [], [], [], [], []
    for c in CONDS:
        for s in SEEDS:
            d = RUNS / c / f"seed{s}"
            if not (d / "benchmark" / "summary.json").exists():
                continue
            ts = json.loads((d / "train_summary.json").read_text())
            bs = json.loads((d / "benchmark" / "summary.json").read_text())
            rec = {"condition": c, "seed": s, **{k: ts[k] for k in ("best_epoch", "epochs_run", "best_shared_val_pr_auc",
                                                                      "best_shared_val_auroc", "best_shared_val_loss")}}
            for r in REGIONS:
                rec[f"{r}_tile_auroc"] = bs["regions"][r]["tile_auroc"]
                rec[f"{r}_tile_pr_auc"] = bs["regions"][r]["tile_pr_auc"]
                cv = pd.read_csv(d / "benchmark" / f"facility_{r}_curve.csv")
                curves.append(cv.assign(condition=c, seed=s, region=r))
                hits.append(pd.read_csv(d / "benchmark" / f"facility_{r}_first_hits.csv").assign(condition=c, seed=s, region=r))
                rts.append(pd.read_csv(d / "benchmark" / f"facility_{r}_recall_targets.csv").assign(condition=c, seed=s, region=r))
            for k, v in bs["aggregate"].items():
                rec[k] = v
            rows.append(rec)
            m = pd.read_json(d / "metrics.jsonl", lines=True)
            epochs.append(m.assign(condition=c, seed=s))
    return (pd.DataFrame(rows), pd.concat(curves), pd.concat(hits), pd.concat(epochs), pd.concat(rts))


def random_reference(n: int = 200) -> pd.DataFrame:
    """Facility recall at each budget and tile metrics for random score orderings (same evaluation code)."""
    cfg = load_config()
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    pt = pt[pt.region.isin(REGIONS)]
    facilities, assignments = fm.reconstruct(cfg)
    mem = fm.parent_membership(assignments)
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        s = pt[["region", "tile_id"]].assign(score=rng.random(len(pt)))
        for r in REGIONS:
            fr = evaluate_parent_scores(s, pt, facilities, mem, r, aoi_area_km2=cfg["geometry"]["aoi_area_km2_nominal"][r],
                                        recall_targets=cfg["evaluation"]["recall_targets"])
            for b in BUDGETS:
                k, rc = recall_at(fr.curve, b)
                out.append({"perm": i, "region": r, "budget": b, "surfaced": k, "recall": rc,
                            "n_fac": fr.summary["n_facilities"]})
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, curves, hits, epochs, rts = load_runs()
    if runs.empty:
        raise SystemExit("no finished runs")
    missing = [(c, s) for c in CONDS for s in SEEDS if not ((runs.condition == c) & (runs.seed == s)).any()]
    runs.to_csv(OUT / "runs_long.csv", index=False)
    epochs.to_csv(OUT / "training_curves_long.csv", index=False)
    md = [f"# ResNet-50 negative-composition ablation: main matrix\n",
          f"Runs: {len(runs)} of {len(CONDS) * len(SEEDS)}" + (f"; MISSING {missing}" if missing else "") + ".",
          "Values: mean ± sample std over seeds (n = 3). Facility map PROVISIONAL (facility_mapping_questions.md).\n"]

    # T1
    t1 = runs.groupby("condition").agg(**{k: (k, ms) for k in ("best_epoch", "epochs_run", "best_shared_val_pr_auc",
                                                               "best_shared_val_auroc", "best_shared_val_loss")})
    t1.to_csv(OUT / "T1_checkpoint_selection.csv")
    md += ["## T1 Checkpoint selection (shared validation set, 118 tiles)\n", md_table(t1), ""]

    # T2
    cols = [f"{r}_{m}" for r in REGIONS for m in ("tile_auroc", "tile_pr_auc")] + \
           ["pooled_tile_auroc", "pooled_tile_pr_auc", "mean_tile_auroc", "mean_tile_pr_auc"]
    t2 = runs.groupby("condition").agg(**{k: (k, ms) for k in cols}).T
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    prev = {r: pt[pt.region == r].human_target.fillna(0).mean() for r in REGIONS}
    prev["pooled"] = pt[pt.region.isin(REGIONS)].human_target.fillna(0).mean()
    t2["random"] = [("0.500" if "auroc" in k else f"{prev.get(k.split('_')[0], prev['pooled']):.3f}") if not k.startswith("mean")
                    else ("0.500" if "auroc" in k else f"{np.mean([prev[r] for r in REGIONS]):.3f}") for k in t2.index]
    t2.to_csv(OUT / "T2_tile_discrimination.csv")
    md += ["## T2 Benchmark tile discrimination (label: human_target)\n", md_table(t2), ""]

    # T3 facility recall at equal budget
    rec = []
    for (c, s, r), cv in curves.groupby(["condition", "seed", "region"]):
        nf = int(hits[(hits.condition == c) & (hits.seed == s) & (hits.region == r)].shape[0])
        for b in BUDGETS:
            k, rc = recall_at(cv, b)
            rec.append({"condition": c, "seed": s, "region": r, "budget": b, "surfaced": k, "recall": rc, "n_fac": nf})
    rec = pd.DataFrame(rec)
    pooled = rec.groupby(["condition", "seed", "budget"]).agg(surfaced=("surfaced", "sum"), n_fac=("n_fac", "sum")).reset_index()
    pooled = pooled.assign(region="pooled", recall=pooled.surfaced / pooled.n_fac)
    rec = pd.concat([rec, pooled])
    rec.to_csv(OUT / "facility_recall_at_budget_long.csv", index=False)
    rnd = random_reference()
    rp = rnd.groupby(["perm", "budget"]).agg(surfaced=("surfaced", "sum"), n_fac=("n_fac", "sum")).reset_index()
    rnd = pd.concat([rnd, rp.assign(region="pooled", recall=rp.surfaced / rp.n_fac)])
    rnd_mean = rnd.groupby(["region", "budget"]).recall.mean()
    t3 = rec.groupby(["region", "budget", "condition"]).recall.apply(ms).unstack("condition")
    t3["random (mean of 200)"] = [f"{rnd_mean[i]:.3f}" for i in t3.index]
    t3 = t3.reindex(REGIONS + ["pooled"], level=0)
    t3.to_csv(OUT / "T3_facility_recall_at_budget.csv")
    nfac = rec[rec.condition == CONDS[0]].groupby("region").n_fac.first().to_dict()
    md += ["## T3 Facility recall at equal review budget (% of AOI area reviewed)\n",
           "Known facilities in the primary denominator: " + ", ".join(f"{k} {v}" for k, v in nfac.items()) + ".\n",
           md_table(t3), ""]

    # T4 review area for recall targets
    t4 = rts.assign(pct=rts.pct_aoi_reviewed).groupby(["region", "recall_target", "condition"]).pct.apply(ms).unstack("condition")
    t4 = t4.reindex(REGIONS, level=0)
    t4.to_csv(OUT / "T4_review_area_for_recall_targets.csv")
    md += ["## T4 Review area (% of AOI) needed to reach a facility-recall target\n", md_table(t4), ""]

    # T5 per-facility first hit
    fac = fm.reconstruct(load_config())[0].set_index("facility_id")
    hits["facility"] = hits.facility_id.map(lambda f: f"{f} ({fac.facility_type.get(f, '?')})")
    t5 = hits.groupby(["region", "facility", "condition"]).first_hit_pct_aoi_expected.apply(ms).unstack("condition")
    t5 = t5.reindex(REGIONS, level=0)
    hits.to_csv(OUT / "facility_first_hits_long.csv", index=False)
    t5.to_csv(OUT / "T5_per_facility_review_area.csv")
    md += ["## T5 Review area (% of AOI, expected under ties) to surface each known facility\n", md_table(t5), ""]

    # Seed variability + reversals
    var = runs.groupby("condition")[["best_shared_val_pr_auc", "pooled_tile_auroc", "pooled_tile_pr_auc"] +
                                    [f"{r}_tile_auroc" for r in REGIONS]].agg(lambda x: x.max() - x.min())
    var.to_csv(OUT / "seed_ranges.csv")
    md += ["## Seed variability (max − min over seeds)\n", md_table(var.round(3)), ""]
    flags = []
    metrics = {"tile_auroc": lambda r: f"{r}_tile_auroc", "tile_pr_auc": lambda r: f"{r}_tile_pr_auc"}
    for mname, col in metrics.items():
        pooled_order = runs.groupby("condition")[f"pooled_{mname}"].mean().sort_values(ascending=False).index.tolist()
        for r in REGIONS:
            g = runs.groupby("condition")[col(r)]
            order = g.mean().sort_values(ascending=False).index.tolist()
            if order != pooled_order:
                gap = g.mean().max() - g.mean().min()
                flags.append({"metric": mname, "region": r, "aoi_order": " > ".join(order), "pooled_order": " > ".join(pooled_order),
                              "aoi_spread": round(gap, 3), "mean_seed_std": round(float(g.std().mean()), 3)})
    for b in (5, 10, 20):
        x = rec[rec.budget == b]
        po = x[x.region == "pooled"].groupby("condition").recall.mean().sort_values(ascending=False)
        for r in REGIONS:
            g = x[x.region == r].groupby("condition").recall
            o = g.mean().sort_values(ascending=False)
            if o.index.tolist() != po.index.tolist() and o.max() != o.min():
                flags.append({"metric": f"facility_recall@{b}%", "region": r, "aoi_order": " > ".join(o.index),
                              "pooled_order": " > ".join(po.index), "aoi_spread": round(o.max() - o.min(), 3),
                              "mean_seed_std": round(float(g.std().mean()), 3)})
    fl = pd.DataFrame(flags)
    fl.to_csv(OUT / "aoi_reversals.csv", index=False)
    md += ["## AOI-specific reversals (condition ranking differs from pooled)\n",
           md_table(fl, index=False) if len(fl) else "None.", ""]
    (OUT / "results_tables.md").write_text("\n".join(md))

    # Figures
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for (c, s), e in epochs.groupby(["condition", "seed"]):
        for a, k in zip(ax, ("shared_val_pr_auc", "shared_val_auroc", "shared_val_loss")):
            a.plot(e.epoch, e[k], color=COLORS[c], alpha=0.8, lw=1.2, label=LABEL[c] if s == 0 else None)
            b = runs[(runs.condition == c) & (runs.seed == s)].best_epoch.iloc[0]
            a.plot(b, e[e.epoch == b][k].iloc[0], "o", color=COLORS[c], ms=4)
    for a, k in zip(ax, ("Shared-val PR-AUC", "Shared-val AUROC", "Shared-val loss")):
        a.set_title(k); a.set_xlabel("epoch")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_training_curves.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(1, len(REGIONS) + 1, figsize=(15, 3.6))
    grid = np.linspace(0, 100, 401)
    for a, r in zip(ax, REGIONS + ["pooled"]):
        for c in CONDS:
            ys = []
            for s in SEEDS:
                if not ((runs.condition == c) & (runs.seed == s)).any():
                    continue
                if r == "pooled":
                    tot = sum(stepi(grid, *_step(curves, c, s, rr)) * nfac[rr] for rr in REGIONS)
                    ys.append(tot / sum(nfac[rr] for rr in REGIONS))
                elif ((curves.condition == c) & (curves.seed == s) & (curves.region == r)).any():
                    ys.append(stepi(grid, *_step(curves, c, s, r)))
            if ys:
                ys = np.array(ys)
                a.plot(grid, ys.mean(0), color=COLORS[c], label=LABEL[c], drawstyle="steps-post")
                a.fill_between(grid, ys.min(0), ys.max(0), color=COLORS[c], alpha=0.15, step="post")
        rr = rnd[rnd.region == r].groupby("budget").recall.mean()
        a.plot(rr.index, rr.values, "--", color=COLORS["random"], label="random (at budgets)")
        a.set_xscale("symlog", linthresh=1); a.set_xlim(0, 100); a.set_ylim(0, 1.02)
        a.set_title(f"{r} ({nfac[r]} facilit{'y' if nfac[r] == 1 else 'ies'})"); a.set_xlabel("% of AOI area reviewed")
    ax[0].set_ylabel("facility recall"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_facility_recall_vs_budget.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, m in zip(ax, ("tile_auroc", "tile_pr_auc")):
        xs = REGIONS + ["pooled"]
        for j, c in enumerate(CONDS):
            for i, r in enumerate(xs):
                v = runs[runs.condition == c][f"{r}_{m}"]
                a.scatter(np.full(len(v), i + (j - 1) * 0.22), v, color=COLORS[c], s=14, label=LABEL[c] if i == 0 else None)
                a.hlines(v.mean(), i + (j - 1) * 0.22 - 0.09, i + (j - 1) * 0.22 + 0.09, color=COLORS[c])
        if m == "tile_auroc":
            a.axhline(0.5, ls="--", color="grey", lw=0.8)
        else:
            for i, r in enumerate(xs):
                a.hlines(prev[r], i - 0.35, i + 0.35, ls="--", color="grey", lw=0.8)
        a.set_xticks(range(len(xs))); a.set_xticklabels(xs); a.set_title(m.replace("_", " "))
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_tile_metrics.png", dpi=160); plt.close(fig)

    h = hits.groupby(["region", "facility", "condition"]).first_hit_pct_aoi_expected.agg(["mean", "min", "max"]).reset_index()
    order = h.groupby(["region", "facility"])["mean"].mean().sort_values().reset_index()
    fig, a = plt.subplots(figsize=(8, 0.28 * len(order) + 1.2))
    for i, row in order.iterrows():
        for j, c in enumerate(CONDS):
            x = h[(h.facility == row.facility) & (h.condition == c)]
            if len(x):
                a.errorbar(x["mean"], i + (j - 1) * 0.22, xerr=[x["mean"] - x["min"], x["max"] - x["mean"]], fmt="o",
                           color=COLORS[c], ms=3, lw=0.8, label=LABEL[c] if i == 0 else None)
    a.set_yticks(range(len(order))); a.set_yticklabels([f"{r}: {f}" for r, f in zip(order.region, order.facility)], fontsize=6)
    a.set_xscale("symlog", linthresh=1); a.set_xlabel("% of AOI reviewed before the facility is surfaced (expected)")
    a.invert_yaxis(); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_per_facility_first_hit.png", dpi=160); plt.close(fig)
    print("\n".join(md))


def stepi(grid, x, y):
    """Right-continuous step function (recall only changes once a tile group is fully reviewed)."""
    return y[np.clip(np.searchsorted(x, grid, side="right") - 1, 0, len(y) - 1)]


def _step(curves, c, s, r):
    cv = curves[(curves.condition == c) & (curves.seed == s) & (curves.region == r)]
    return cv.pct_aoi_reviewed.to_numpy(), cv.facility_recall.to_numpy()


if __name__ == "__main__":
    main()
