"""Rule A: the manuscript's reasoning-validated facility-surfacing rule, on VLM output images.

Manuscript (Methods > Metrics): a facility is surfaced when (1) it falls within at least one tile or
quadrant scored >= 5 and (2) the VLM reasoning correctly identifies its features in that tile/quadrant.

Operationalisation from the workbook:
  * Units are VLM output images. single_pass: every parent tile (original + later-replaced parents).
    adaptive: the final outputs ("Included in final?" == 1: un-zoomed parents, quadrants, zoom-out
    blocks); replaced parents are not reviewed.
  * A unit belongs to a facility if its human label names the facility (see facility_mapping). Rows
    scored >= 5 whose reasoning was rejected are labelled FP without a facility name, so condition (2)
    is carried by the TP labels. Replaced parents inherit their children's facilities (single pass).
  * Review area = sum of unit footprints (parent 3.61 km2, z16 quadrant 0.9025 km2, z17 parent 0.9025 km2,
    z17 quadrant 0.2256 km2; zoom-out block = parent size).
Evaluated with evaluation.evaluate_review(tie_policy="group") and read at score >= threshold.
"""
from __future__ import annotations

import pandas as pd

import facility_mapping as fm
from common import STAGE_ORIGINAL, STAGE_REPLACED, load_config, load_processed
from evaluation import ReviewResult, evaluate_review

WORKFLOWS = ("single_pass", "adaptive")


def manuscript_inputs(region: str, zoom: int, workflow: str, config: dict | None = None):
    cfg = config or load_config()
    if workflow not in WORKFLOWS:
        raise ValueError(workflow)
    df = load_processed(cfg)
    df = df[(df.region == region) & (df.zoom == zoom)]
    rows = df[df.stage.isin([STAGE_ORIGINAL, STAGE_REPLACED])] if workflow == "single_pass" else df[df.included_final == 1]
    units = rows.assign(area_km2=rows.width_km ** 2)[["unit_id", "area_km2"]]
    scores = rows[["unit_id", "score"]]
    facilities, assignments = fm.reconstruct(cfg)
    mem = fm.row_membership(df, assignments)
    mem = mem[mem.unit_id.isin(units.unit_id)]
    fac = facilities[(facilities.region == region) & facilities.in_primary_denominator]
    return scores, units, mem, fac


def evaluate_manuscript_rule(region: str, zoom: int, workflow: str, config: dict | None = None) -> tuple[ReviewResult, pd.DataFrame]:
    cfg = config or load_config()
    scores, units, mem, fac = manuscript_inputs(region, zoom, workflow, cfg)
    aoi = cfg["geometry"]["aoi_area_km2_nominal"][region]
    res = evaluate_review(scores, units, mem, fac.facility_id.tolist(), aoi_area_km2=aoi, tie_policy="group")
    return res, fac
