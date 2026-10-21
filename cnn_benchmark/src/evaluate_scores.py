"""Evaluate ANY parent-tile score file under the common facility-footprint rule.

Input CSV must contain region, tile_id and a score column, with every z16 parent tile of each evaluated
region scored (the evaluator refuses partial coverage).

  python src/evaluate_scores.py --scores outputs/vlm_parent_scores.csv --score-col vlm_score_single_pass \
      --name vlm_single_pass
  python src/evaluate_scores.py --scores cnn_scores_saratov_cv.csv --score-col cnn_score --regions Saratov \
      --name cnn_resnet50_saratov_cv

Writes <out-dir>/<name>/{curve,first_hits,recall_targets}_<region>.csv and summary.json. Score files
labelled PIPELINE_SMOKE_TEST_ONLY are routed to outputs/_PIPELINE_SMOKE_TEST_ONLY/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import facility_mapping as fm
from common import OUTPUT_DIR, load_config
from dataset import SANITY_LABEL, SMOKE_TEST_LABEL
from evaluation import TIE_POLICIES, evaluate_parent_scores


def main(argv=None) -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--name", required=True)
    ap.add_argument("--regions", nargs="+", default=cfg["regions"])
    ap.add_argument("--include-candidate-facilities", action="store_true")
    ap.add_argument("--tie-policy", default="group", choices=TIE_POLICIES)
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR / "evaluations"))
    args = ap.parse_args(argv)

    s = pd.read_csv(args.scores)
    labels = set(s.run_label) if "run_label" in s else set()
    special = next((lab for lab in (SMOKE_TEST_LABEL, SANITY_LABEL) if lab in labels), None)
    smoke = special is not None
    out = (OUTPUT_DIR / f"_{special}" / "evaluations" if smoke else Path(args.out_dir)) / args.name
    out.mkdir(parents=True, exist_ok=True)
    s = s.rename(columns={args.score_col: "score"})[["region", "tile_id", "score"]]

    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    facilities, assignments = fm.reconstruct(cfg)
    mem = fm.parent_membership(assignments)
    summary = dict(name=args.name, scores=str(args.scores), score_col=args.score_col, tie_policy=args.tie_policy,
                   include_candidate_facilities=args.include_candidate_facilities,
                   facility_map="PROVISIONAL (see facility_mapping_questions.md)",
                   run_label=special or "", regions={})
    for region in args.regions:
        r = evaluate_parent_scores(s, pt, facilities, mem, region,
                                   aoi_area_km2=cfg["geometry"]["aoi_area_km2_nominal"][region],
                                   include_candidates=args.include_candidate_facilities, tie_policy=args.tie_policy,
                                   recall_targets=cfg["evaluation"]["recall_targets"])
        r.curve.to_csv(out / f"curve_{region}.csv", index=False)
        r.first_hits.to_csv(out / f"first_hits_{region}.csv", index=False)
        r.recall_targets.to_csv(out / f"recall_targets_{region}.csv", index=False)
        summary["regions"][region] = dict(r.summary, recall_targets=r.recall_targets.to_dict("records"))
        print(f"{region}: {r.summary['n_facilities']} facilities")
        print(r.recall_targets[["recall_target", "n_reviewed", "area_reviewed_km2", "pct_aoi_reviewed",
                                "workload_reduction"]].to_string(index=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote {out}" + (f"  [{special}: not a result]" if smoke else ""))


if __name__ == "__main__":
    main()
