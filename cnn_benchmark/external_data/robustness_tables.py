"""Paper tables for the CNN robustness checks, built from raw per-run outputs only (frozen results are read, never written).

    python robustness_tables.py     # -> results/robustness/paper_tables/

  table_nuisance_matching.{csv,md}   C [MPS, frozen] (reference) | C [CUDA control] | C_matched [CUDA]
  table_mask_robustness.{csv,md}     A mask on / off [MPS], C mask on / off [MPS]
  seed_level_metrics.csv             every run, every metric (the tables are mean +/- sample SD over seeds 0-2)
Metrics: pooled tile AUROC and PR-AUC (all 2,940 benchmark tiles; human_target), Saratov facility recall at 20 % of
the AOI reviewed, AOI area (%) reviewed to reach 50 % Saratov facility recall, Natanz and Fordow first-hit area (% of
AOI, expected under ties). Same evaluation code (src/evaluation.py) as the main ablation for every run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parents[1]
ABL = PKG / "outputs" / "ablation"
OUT = PKG.parent / "results" / "robustness" / "paper_tables"
RUNS = {  # label -> (run directory, backend, mask)
    "A [MPS, mask on, frozen]": (ABL / "runs" / "A", "MPS", "on"),
    "A [MPS, mask off]": (ABL / "runs_robustness_mps" / "A_unmasked", "MPS", "off"),
    "C [MPS, mask on, frozen]": (ABL / "runs" / "C", "MPS", "on"),
    "C [MPS, mask off]": (ABL / "runs_robustness_mps" / "C_unmasked", "MPS", "off"),
    "C [CUDA control]": (ABL / "runs_robustness_cuda" / "C", "CUDA (A100)", "on"),
    "C_matched [CUDA]": (ABL / "runs_robustness_cuda" / "C_matched", "CUDA (A100)", "on"),
}
METRICS = [("pooled_tile_auroc", "Pooled tile AUROC", 3), ("pooled_tile_pr_auc", "Pooled tile PR-AUC", 3),
           ("saratov_recall_at_20pct", "Saratov recall @20% area", 2), ("saratov_area_to_50pct", "Area to 50% Saratov recall (% AOI)", 1),
           ("natanz_first_hit_pct", "Natanz first-hit area (% AOI)", 2), ("fordow_first_hit_pct", "Fordow first-hit area (% AOI)", 2)]


def run_metrics(d: Path) -> dict:
    b = d / "benchmark"
    agg = json.loads((b / "summary.json").read_text())["aggregate"]
    curve = pd.read_csv(b / "facility_Saratov_curve.csv")
    rt = pd.read_csv(b / "facility_Saratov_recall_targets.csv")
    fh = {r: pd.read_csv(b / f"facility_{r}_first_hits.csv") for r in ("Natanz", "Fordow")}
    cfg = json.loads((d / "run_config.json").read_text())
    return {"pooled_tile_auroc": agg["pooled_tile_auroc"], "pooled_tile_pr_auc": agg["pooled_tile_pr_auc"],
            "saratov_recall_at_20pct": float(curve[curve.pct_aoi_reviewed <= 20 + 1e-9].facility_recall.iloc[-1]),
            "saratov_area_to_50pct": float(rt[rt.recall_target == 0.5].pct_aoi_reviewed.iloc[0]),
            "natanz_first_hit_pct": float(fh["Natanz"].first_hit_pct_aoi_expected.iloc[0]),
            "fordow_first_hit_pct": float(fh["Fordow"].first_hit_pct_aoi_expected.iloc[0]),
            "device": cfg.get("device"), "watermark_mask": cfg.get("watermark_mask"),
            "manifest_sha256": cfg.get("manifest_sha256", "")[:12]}


def table(df: pd.DataFrame, labels: list, notes: dict) -> pd.DataFrame:
    rows = []
    for lab in labels:
        g = df[df.run == lab]
        r = {"Condition": lab + notes.get(lab, ""), "Backend": RUNS[lab][1], "Seeds": len(g)}
        for key, name, dp in METRICS:
            r[name] = f"{g[key].mean():.{dp}f} ± {g[key].std(ddof=1):.{dp}f}"
        rows.append(r)
    return pd.DataFrame(rows)


def to_md(t: pd.DataFrame, italic_rows=()) -> str:
    head = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
    body = ""
    for _, r in t.iterrows():
        cells = [f"*{v}*" if r["Condition"].split(" (")[0] in italic_rows else str(v) for v in r.values]
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for lab, (d, backend, mask) in RUNS.items():
        for s in (0, 1, 2):
            rows.append({"run": lab, "seed": s, "backend": backend, "mask": mask, **run_metrics(d / f"seed{s}")})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "seed_level_metrics.csv", index=False)
    nm = ["C [MPS, mask on, frozen]", "C [CUDA control]", "C_matched [CUDA]"]
    t1 = table(df, nm, {"C [MPS, mask on, frozen]": " (reference; not part of the controlled comparison)"})
    t1.to_csv(OUT / "table_nuisance_matching.csv", index=False)
    mk = ["A [MPS, mask on, frozen]", "A [MPS, mask off]", "C [MPS, mask on, frozen]", "C [MPS, mask off]"]
    t2 = table(df, mk, {})
    t2.to_csv(OUT / "table_mask_robustness.csv", index=False)
    (OUT / "table_nuisance_matching.md").write_text(
        "**Robustness to nuisance matching.** Controlled comparison: C [CUDA control] vs C_matched [CUDA] (same backend, "
        "seeds, recipe; only the nearby-negative sampling differs). The italic MPS row is the frozen main-ablation C, "
        "shown for reference. Mean ± SD over 3 seeds; no significance claims.\n\n"
        + to_md(t1, italic_rows=("C [MPS, mask on, frozen]",)))
    (OUT / "table_mask_robustness.md").write_text(
        "**Robustness to imagery masking (all Apple MPS).** Mask on = frozen main-ablation runs; mask off = identical "
        "recipe with the geometric watermark mask disabled. Mean ± SD over 3 seeds; differences are within seed "
        "variability and do not change the A-vs-C interpretation.\n\n" + to_md(t2))
    pd.set_option("display.width", 250)
    print(t1.to_string(index=False)); print(); print(t2.to_string(index=False))


if __name__ == "__main__":
    main()
