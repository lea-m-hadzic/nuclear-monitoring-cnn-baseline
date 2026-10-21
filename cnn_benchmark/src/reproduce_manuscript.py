"""Reproduce the manuscript's facility-level numbers and write VLM baseline outputs.

Authority: `Manuscript 5.docx` (latest revision). Expected values below are transcribed from it.

Outputs (cnn_benchmark/outputs/):
  manuscript_reproduction.csv          every checked quantity: manuscript vs reproduced
  vlm_parent_scores.csv                VLM scores per z16 parent (single pass; adaptive = max over final outputs)
  vlm_facility_curve_saratov.csv       Saratov recall-vs-area curves (rule A and footprint rule)
  vlm_facility_first_hits.csv          per-facility first hits, all AOIs, all VLM methods
  vlm_facility_recall_vs_review_area.png   infrastructure-validation plot (Saratov z16)

Usage: python src/reproduce_manuscript.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import facility_mapping as fm
from build_parent_table import build as build_parent_table
from common import ensure_output_dir, load_config, load_processed
from evaluation import at_threshold, evaluate_parent_scores
from manuscript_rule import evaluate_manuscript_rule

TYPE_SHORT = {"icbm_silo": "silo", "warhead_storage": "warhead_storage",
              "warhead_rail_to_road_transfer": "transfer_point", "enrichment_plant": "enrichment"}

# (region, zoom, workflow) -> Table 2 / Table 4 values. tol = half a unit of the reported precision.
MANUSCRIPT_TABLES = {
    ("Saratov", 16, "single_pass"): dict(table="Table 2", surfaced=9, types={"silo": 7, "warhead_storage": 2},
                                         rate_pct=37.5, area=216.6, area_tol=0.05, pct=4.3, api_calls=2774),
    ("Saratov", 16, "adaptive"): dict(table="Table 2", surfaced=13, types={"silo": 10, "warhead_storage": 3},
                                      rate_pct=54.2, area=280, area_tol=0.5, pct=5.6, api_calls=3886),
    ("Saratov", 17, "single_pass"): dict(table="Table 2", surfaced=11,
                                         types={"silo": 9, "warhead_storage": 1, "transfer_point": 1},
                                         rate_pct=45.8, area=130.9, area_tol=0.05, pct=2.6, api_calls=11094),
    ("Saratov", 17, "adaptive"): dict(table="Table 2", surfaced=18,
                                      types={"silo": 14, "warhead_storage": 3, "transfer_point": 1},
                                      rate_pct=75.0, area=160, area_tol=0.5, pct=3.2, api_calls=13950),
    ("Natanz", 16, "single_pass"): dict(table="Table 4", surfaced=1, area=227.4, area_tol=0.05, pct=8.0, api_calls=1552),
    ("Natanz", 16, "adaptive"): dict(table="Table 4", surfaced=1, area=251.8, area_tol=0.05, pct=8.9, api_calls=2360),
    ("Fordow", 16, "single_pass"): dict(table="Table 4", surfaced=1, area=166.1, area_tol=0.05, pct=5.9, api_calls=1554),
    ("Fordow", 16, "adaptive"): dict(table="Table 4", surfaced=1, area=192.2, area_tol=0.05, pct=6.8, api_calls=2392),
}
# Table 3: Saratov z17 adaptive, threshold -> (facilities, rate %, area km2, % of area)
MANUSCRIPT_TABLE3 = {2: (20, 83.3, 194.9, 3.9), 3: (18, 75.0, 174.2, 3.5), 4: (18, 75.0, 160.4, 3.2),
                     5: (18, 75.0, 160.4, 3.2), 6: (11, 45.8, 85.5, 1.7), 7: (10, 41.7, 58.2, 1.2),
                     8: (1, 4.2, 2.0, 0.04)}
# Results-text counts of flagged outputs at score >= 5 (parents, quadrants/blocks) and follow-ups.
MANUSCRIPT_TEXT = {
    ("Saratov", 16): dict(flagged_parents=60, flagged_followups=70, triggered=139, followups=556),
    ("Saratov", 17): dict(flagged_parents=145, flagged_followups=131),
    ("Natanz", 16): dict(flagged_parents=63, flagged_followups=27, triggered=101, followups=404),
    ("Fordow", 16): dict(flagged_parents=46, flagged_followups=29, triggered=106, followups=419),
}


def _check(rows, source, quantity, expected, got, tol=0.0, note=""):
    ok = abs(float(got) - float(expected)) <= tol + 1e-9
    rows.append(dict(source=source, quantity=quantity, manuscript=expected, reproduced=got,
                     tolerance=tol, match=ok, note=note))


def reproduce(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = load_processed(cfg)
    facilities, _ = fm.reconstruct(cfg)
    ftype = facilities.set_index("facility_id").facility_type.map(TYPE_SHORT)
    rows = []
    for (region, zoom, wf), exp in MANUSCRIPT_TABLES.items():
        res, fac = evaluate_manuscript_rule(region, zoom, wf, cfg)
        st = at_threshold(res.curve, cfg["evaluation"]["vlm_threshold"])
        tag = f"{exp['table']} {region} z{zoom} {wf}"
        ids = [i for i in st.surfaced_ids.split(";") if i]
        _check(rows, tag, "facilities surfaced", exp["surfaced"], len(ids))
        if "rate_pct" in exp:
            _check(rows, tag, "facility retrieval %", exp["rate_pct"], round(100 * st.facility_recall, 1), 0.05)
        for t, n in exp.get("types", {}).items():
            _check(rows, tag, f"surfaced {t}", n, int((ftype.reindex(ids) == t).sum()))
        _check(rows, tag, "area flagged km2", exp["area"], round(st.area_reviewed_km2, 2), exp["area_tol"])
        _check(rows, tag, "% of AOI flagged", exp["pct"], round(st.pct_aoi_reviewed, 2), 0.05)
        sub = df[(df.region == region) & (df.zoom == zoom)]
        n_img = sub.stage.isin(["Original output", "Replaced parent"]).sum() if wf == "single_pass" else len(sub)
        _check(rows, tag, "OpenAI API calls (2 per image)", exp["api_calls"], 2 * int(n_img))
        if region == "Saratov":
            missed = fac.facility_id[~fac.facility_id.isin(ids)]
            _check(rows, tag, "facilities missed", 24 - exp["surfaced"], len(missed),
                   note="candidate 21st z16 silo excluded from denominator (see facility_mapping_questions.md)")

    res, _ = evaluate_manuscript_rule("Saratov", 17, "adaptive", cfg)
    for t, (n, rate, area, pct) in MANUSCRIPT_TABLE3.items():
        st = at_threshold(res.curve, t)
        tag = f"Table 3 Saratov z17 threshold {t}"
        _check(rows, tag, "facilities retrieved", n, int(st.facilities_surfaced))
        _check(rows, tag, "retrieval %", rate, round(100 * st.facility_recall, 1), 0.05)
        _check(rows, tag, "area km2", area, round(st.area_reviewed_km2, 2), 0.05)
        _check(rows, tag, "% of area", pct, round(st.pct_aoi_reviewed, 2), 0.05 if pct >= 0.1 else 0.005)

    for (region, zoom), exp in MANUSCRIPT_TEXT.items():
        sub = df[(df.region == region) & (df.zoom == zoom)]
        fin = sub[sub.included_final == 1]
        tag = f"Results text {region} z{zoom}"
        _check(rows, tag, "flagged parent tiles (score>=5)", exp["flagged_parents"],
               int(((fin.stage == "Original output") & (fin.score >= 5)).sum()))
        _check(rows, tag, "flagged quadrants/blocks (score>=5)", exp["flagged_followups"],
               int(((fin.stage != "Original output") & (fin.score >= 5)).sum()))
        if "triggered" in exp:
            _check(rows, tag, "parents sent to follow-up", exp["triggered"], int((sub.stage == "Replaced parent").sum()))
            _check(rows, tag, "follow-up images", exp["followups"],
                   int(sub.stage.isin(["Quadrant output", "Zoom-out mosaic block"]).sum()))

    # Narrative claims.
    nat, _ = evaluate_manuscript_rule("Natanz", 16, "adaptive", cfg)
    st7 = at_threshold(nat.curve, 7)
    _check(rows, "Results text Natanz", "area at threshold 7 km2", 97.5, round(st7.area_reviewed_km2, 2), 0.05)
    _check(rows, "Results text Natanz", "facility surfaced at threshold 7", 1, int(st7.facilities_surfaced))
    fo, _ = evaluate_manuscript_rule("Fordow", 16, "adaptive", cfg)
    _check(rows, "Results text Fordo", "best facility score (quadrant 6)", 6, fo.first_hits.best_score.iloc[0])
    _check(rows, "Results text Fordo", "facility surfaced at threshold 7", 0, int(at_threshold(fo.curve, 7).facilities_surfaced))
    z17, _ = evaluate_manuscript_rule("Saratov", 17, "adaptive", cfg)
    surfaced = set(at_threshold(z17.curve, 5).surfaced_ids.split(";"))
    sil = facilities[(facilities.facility_type == "icbm_silo") & facilities.in_primary_denominator]
    missed = sil[~sil.facility_id.isin(surfaced)]
    _check(rows, "Results text Saratov z17", "missed silos with a 'cut off' label", len(missed),
           int(missed.human_labels.str.contains("cut off").sum()),
           note="manuscript: all six missed silos were cut off at tile edges; missed without a 'cut off' label: "
                + ", ".join(f"{r.facility_id} (z17 labels: {r.human_labels})"
                            for r in missed[~missed.human_labels.str.contains("cut off")].itertuples()))
    return pd.DataFrame(rows)


def vlm_baselines(cfg: dict | None = None):
    """VLM scores evaluated under the common parent-footprint rule (and rule A for reference)."""
    cfg = cfg or load_config()
    pt = build_parent_table(cfg)
    facilities, assignments = fm.reconstruct(cfg)
    mem = fm.parent_membership(assignments)
    scores = pt[["region", "tile_id", "vlm_score_single_pass", "vlm_score_adaptive_max"]]
    curves, hits = [], []
    methods = {"vlm_single_pass__footprint_parent": "vlm_score_single_pass",
               "vlm_adaptive_max__footprint_parent": "vlm_score_adaptive_max"}
    for region in cfg["regions"]:
        aoi = cfg["geometry"]["aoi_area_km2_nominal"][region]
        for name, col in methods.items():
            s = scores.rename(columns={col: "score"})[["region", "tile_id", "score"]]
            r = evaluate_parent_scores(s, pt, facilities, mem, region, aoi_area_km2=aoi)
            curves.append(r.curve.assign(region=region, method=name, rule="footprint_parent_tiles"))
            hits.append(r.first_hits.assign(region=region, method=name, rule="footprint_parent_tiles"))
        r, _ = evaluate_manuscript_rule(region, 16, "adaptive", cfg)
        curves.append(r.curve.assign(region=region, method="vlm_adaptive__manuscript_rule_outputs",
                                     rule="manuscript_reasoning_validated"))
        hits.append(r.first_hits.assign(region=region, method="vlm_adaptive__manuscript_rule_outputs",
                                        rule="manuscript_reasoning_validated"))
    first_hits = pd.concat(hits).merge(facilities[["facility_id", "facility_type"]], on="facility_id")
    return scores, pd.concat(curves), first_hits


def plot_saratov(curves: pd.DataFrame, path, aoi: float) -> None:
    c = curves[curves.region == "Saratov"]
    style = {"vlm_single_pass__footprint_parent": ("VLM single pass, parent footprint rule", "#1f77b4", "-"),
             "vlm_adaptive_max__footprint_parent": ("VLM + quadrants (max), parent footprint rule", "#ff7f0e", "-"),
             "vlm_adaptive__manuscript_rule_outputs": ("VLM + quadrants, manuscript rule (output images)", "#2ca02c", "--")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, xmax in zip(axes, (100, 15)):
        for m, (lab, color, ls) in style.items():
            d = c[c.method == m]
            ax.step(d.pct_aoi_reviewed, d.facility_recall, where="post", color=color, ls=ls, label=lab)
        ax.scatter([100 * 280 / aoi, 100 * 216.6 / aoi], [13 / 24, 9 / 24], marker="x", color="k", zorder=5,
                   label="Manuscript Table 2 (z16, threshold 5)")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("% of AOI reviewed")
        sec = ax.secondary_xaxis("top", functions=(lambda p: p * aoi / 100, lambda x: 100 * x / aoi))
        sec.set_xlabel("Area reviewed (km²)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Facility recall (24 known targets)")
    axes[0].set_title("Full range", fontsize=9)
    axes[1].set_title("Zoom: first 15% of AOI", fontsize=9)
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Saratov z16: VLM baseline facility recall vs review area "
                 "(infrastructure validation; provisional facility map)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cfg = load_config()
    out = ensure_output_dir()
    rep = reproduce(cfg)
    rep.to_csv(out / "manuscript_reproduction.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Manuscript checks: {int(rep.match.sum())}/{len(rep)} match")
    bad = rep[~rep.match]
    if len(bad):
        print("\nDISCREPANCIES:")
        print(bad.to_string(index=False))

    scores, curves, hits = vlm_baselines(cfg)
    scores.to_csv(out / "vlm_parent_scores.csv", index=False)
    curves[curves.region == "Saratov"].to_csv(out / "vlm_facility_curve_saratov.csv", index=False)
    hits.to_csv(out / "vlm_facility_first_hits.csv", index=False)
    plot_saratov(curves, out / "vlm_facility_recall_vs_review_area.png", cfg["geometry"]["aoi_area_km2_nominal"]["Saratov"])

    print("\nVLM under the parent-footprint rule (z16; area needed for recall targets, km2):")
    for region in cfg["regions"]:
        for m in ["vlm_single_pass__footprint_parent", "vlm_adaptive_max__footprint_parent"]:
            h = hits[(hits.region == region) & (hits.method == m)]
            print(f"  {region:8s} {m:38s} median first-hit rank (expected) {h.first_hit_rank_expected.median():7.1f}"
                  f"  max first-hit area (group end) {h.first_hit_area_group_end_km2.max():7.1f} km2")
    print("Wrote manuscript_reproduction.csv, vlm_parent_scores.csv, vlm_facility_curve_saratov.csv, "
          "vlm_facility_first_hits.csv, vlm_facility_recall_vs_review_area.png")


if __name__ == "__main__":
    main()
