"""Paper-facing summary tables for the frozen CNN baseline (reads frozen outputs only; trains / scores nothing).

    python paper_summary.py      # -> results/main/{abc_summary.csv, facility_metrics.csv, vlm_comparison.csv}

  facility_metrics.csv  one row per condition x seed (A, B, C; seeds 0-2): pooled tile AUROC / PR-AUC, Saratov
                        facility recall at 20 % of the AOI reviewed, area (% AOI) to 50 % Saratov recall, Natanz and
                        Fordow first-hit area (% AOI, expected under ties)
  abc_summary.csv       mean and sample SD over the 3 seeds of the same metrics, per condition
  vlm_comparison.csv    CNN condition C (mean +/- SD) vs the single-pass VLM on the same facility-level metrics
Inputs: provenance/run_records/runs/{A,B,C}/seed*/ (frozen per-run benchmark outputs) through robustness_tables.run_metrics
(the exact function used for the frozen paper tables), and the VLM evaluation in
cnn_benchmark/outputs/evaluations/vlm_single_pass__footprint_parent/ (src/evaluate_scores.py, same footprint rule).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from robustness_tables import run_metrics

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
RUNS = REPO / "provenance" / "run_records" / "runs"
VLM = PKG / "outputs" / "evaluations" / "vlm_single_pass__footprint_parent"
OUT = REPO / "results" / "main"
METRICS = ["pooled_tile_auroc", "pooled_tile_pr_auc", "saratov_recall_at_20pct", "saratov_area_to_50pct",
           "natanz_first_hit_pct", "fordow_first_hit_pct"]
NAMES = {"A": "A (nearby negatives)", "B": "B (hard negatives)", "C": "C (mixed negatives; main condition)"}


def vlm_metrics() -> dict:
    curve = pd.read_csv(VLM / "curve_Saratov.csv")
    rt = pd.read_csv(VLM / "recall_targets_Saratov.csv")
    fh = {r: pd.read_csv(VLM / f"first_hits_{r}.csv") for r in ("Natanz", "Fordow")}
    return {"saratov_recall_at_20pct": float(curve[curve.pct_aoi_reviewed <= 20 + 1e-9].facility_recall.iloc[-1]),
            "saratov_area_to_50pct": float(rt[rt.recall_target == 0.5].pct_aoi_reviewed.iloc[0]),
            "natanz_first_hit_pct": float(fh["Natanz"].first_hit_pct_aoi_expected.iloc[0]),
            "fordow_first_hit_pct": float(fh["Fordow"].first_hit_pct_aoi_expected.iloc[0])}


def main():
    rows = [{"condition": c, "seed": s, **{k: v for k, v in run_metrics(RUNS / c / f"seed{s}").items() if k in METRICS + ["device"]}}
            for c in ("A", "B", "C") for s in (0, 1, 2)]
    seed = pd.DataFrame(rows)
    seed.to_csv(OUT / "facility_metrics.csv", index=False)
    summ = []
    for c, g in seed.groupby("condition"):
        r = {"condition": NAMES[c], "n_seeds": len(g), "backend": "/".join(sorted(g.device.unique()))}
        for m in METRICS:
            r[f"{m}_mean"], r[f"{m}_sd"] = round(g[m].mean(), 4), round(g[m].std(ddof=1), 4)
        summ.append(r)
    pd.DataFrame(summ).to_csv(OUT / "abc_summary.csv", index=False)
    c = seed[seed.condition == "C"]
    v = vlm_metrics()
    comp = []
    for m in METRICS[2:]:
        comp.append({"metric": m, "cnn_C_mean": round(c[m].mean(), 4), "cnn_C_sd": round(c[m].std(ddof=1), 4),
                     "cnn_C_seed_values": ";".join(f"{x:.4f}" for x in c[m]), "vlm_single_pass": round(v[m], 4)})
    comp = pd.DataFrame(comp)
    a50 = comp.set_index("metric").loc["saratov_area_to_50pct"]
    comp["note"] = ""
    comp.loc[comp.metric == "saratov_area_to_50pct", "note"] = (
        f"CNN C needs {a50.cnn_C_mean / a50.vlm_single_pass:.2f}x the VLM's review area to reach 50% Saratov recall "
        "(n = 3 seeds; no significance claim)")
    comp.to_csv(OUT / "vlm_comparison.csv", index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
