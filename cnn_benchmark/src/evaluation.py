"""Model-agnostic facility-level review evaluator.

A "unit" is anything an analyst would review (a z16 parent tile for CNN/VLM comparison; a VLM output
image for the manuscript rule). Given (unit_id, score), units are reviewed in descending score order and
we track how many known facilities have been surfaced. A facility is surfaced as soon as ANY unit in its
footprint (membership) has been reviewed; it is counted once.

Tie policy (scores are often integers for the VLM):
  * "group" (default): all units sharing a score are reviewed together, as one step. Curve points sit
    at the end of each tie group, which equals thresholding at that score (score >= t). First-hit
    rank/area are reported at the END of the group (conservative, threshold-consistent).
  * "optimistic" / "pessimistic": within each tie group, footprint units are reviewed first / last,
    one unit per step. Useful as bounds.
  * The first-hit table always also reports the expected rank/area under uniformly random
    tie-breaking: n_before + (g + 1) / (k + 1), where g = group size and k = the facility's units in
    the group (expected minimum of k random positions among g). Expected area assumes the mean unit
    area of the group (exact when unit areas are equal, as for parent tiles).
Units with equal scores are otherwise ordered by unit_id so results are deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIE_POLICIES = ("group", "optimistic", "pessimistic")


@dataclass
class ReviewResult:
    curve: pd.DataFrame
    first_hits: pd.DataFrame
    recall_targets: pd.DataFrame
    summary: dict = field(default_factory=dict)


def _validate(scores, units, membership, facilities):
    if units.unit_id.duplicated().any():
        raise ValueError("duplicate unit_id in units")
    if scores.unit_id.duplicated().any():
        raise ValueError("duplicate unit_id in scores")
    missing = set(units.unit_id) - set(scores.unit_id)
    if missing:
        raise ValueError(f"{len(missing)} units have no score, e.g. {sorted(missing)[:5]}")
    unknown = set(scores.unit_id) - set(units.unit_id)
    if unknown:
        raise ValueError(f"{len(unknown)} scored ids are not units, e.g. {sorted(unknown)[:5]}")
    if scores.score.isna().any():
        raise ValueError("NaN scores")
    bad = set(membership.unit_id) - set(units.unit_id)
    if bad:
        raise ValueError(f"membership references unknown units, e.g. {sorted(bad)[:5]}")
    empty = set(facilities) - set(membership.facility_id)
    if empty:
        raise ValueError(f"facilities without footprint units: {sorted(empty)}")


def evaluate_review(scores: pd.DataFrame, units: pd.DataFrame, membership: pd.DataFrame,
                    facilities, aoi_area_km2: float | None = None, tie_policy: str = "group",
                    recall_targets=(0.5, 0.75, 0.9, 1.0)) -> ReviewResult:
    """scores: [unit_id, score]; units: [unit_id, area_km2]; membership: [unit_id, facility_id].

    `facilities` is the recall denominator; membership rows for other facilities are ignored.
    """
    if tie_policy not in TIE_POLICIES:
        raise ValueError(f"tie_policy must be one of {TIE_POLICIES}")
    facilities = list(dict.fromkeys(facilities))
    membership = membership[membership.facility_id.isin(facilities)][["unit_id", "facility_id"]].drop_duplicates()
    _validate(scores, units, membership, facilities)
    aoi = float(aoi_area_km2 if aoi_area_km2 is not None else units.area_km2.sum())
    n_fac = len(facilities)

    d = units[["unit_id", "area_km2"]].merge(scores[["unit_id", "score"]], on="unit_id")
    fac_of = membership.groupby("unit_id").facility_id.apply(list).to_dict()
    d["in_footprint"] = d.unit_id.map(lambda u: u in fac_of)
    d = d.sort_values(["score", "unit_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    d["group"] = (d.score != d.score.shift()).cumsum()

    if tie_policy != "group":
        first = tie_policy == "optimistic"
        d = d.sort_values(["group", "in_footprint", "unit_id"], ascending=[True, not first, True],
                          kind="mergesort").reset_index(drop=True)
        d["step"] = np.arange(1, len(d) + 1)
    else:
        d["step"] = d["group"]

    # Curve.
    surfaced: set = set()
    rows = [dict(step=0, score_threshold=np.nan, n_reviewed=0, area_reviewed_km2=0.0, pct_aoi_reviewed=0.0,
                 facilities_surfaced=0, facility_recall=0.0, workload_reduction=1.0, newly_surfaced="")]
    n_cum, a_cum = 0, 0.0
    for step, g in d.groupby("step", sort=True):
        n_cum += len(g)
        a_cum += g.area_km2.sum()
        new = sorted({f for u in g.unit_id[g.in_footprint] for f in fac_of[u]} - surfaced)
        surfaced.update(new)
        rows.append(dict(step=int(step), score_threshold=float(g.score.min()), n_reviewed=n_cum,
                         area_reviewed_km2=a_cum, pct_aoi_reviewed=100 * a_cum / aoi,
                         facilities_surfaced=len(surfaced), facility_recall=len(surfaced) / n_fac,
                         workload_reduction=1 - a_cum / aoi, newly_surfaced=";".join(new)))
    curve = pd.DataFrame(rows)

    # First hits (all tie conventions reported).
    grp = d.groupby("group").agg(g_n=("unit_id", "size"), g_area=("area_km2", "sum"), g_score=("score", "first"))
    grp["n_before"] = grp.g_n.cumsum() - grp.g_n
    grp["area_before"] = grp.g_area.cumsum() - grp.g_area
    hits = []
    for fac in facilities:
        fu = set(membership.unit_id[membership.facility_id == fac])
        m = d[d.unit_id.isin(fu)]
        best = m.group.min()
        G = grp.loc[best]
        k = int((m.group == best).sum())
        g_n, mean_a = int(G.g_n), G.g_area / G.g_n
        exp_rank_in = (g_n + 1) / (k + 1)
        hits.append(dict(
            facility_id=fac, n_footprint_units=len(fu), best_score=float(G.g_score),
            first_hit_units=";".join(sorted(m.unit_id[m.group == best].astype(str))),
            tie_group_size=g_n, footprint_units_in_group=k,
            first_hit_rank_group_end=int(G.n_before + g_n),
            first_hit_area_group_end_km2=float(G.area_before + G.g_area),
            first_hit_rank_optimistic=int(G.n_before + 1),
            first_hit_rank_pessimistic=int(G.n_before + g_n - k + 1),
            first_hit_rank_expected=float(G.n_before + exp_rank_in),
            first_hit_area_expected_km2=float(G.area_before + exp_rank_in * mean_a)))
    first_hits = pd.DataFrame(hits)
    first_hits["first_hit_pct_aoi_group_end"] = 100 * first_hits.first_hit_area_group_end_km2 / aoi
    first_hits["first_hit_pct_aoi_expected"] = 100 * first_hits.first_hit_area_expected_km2 / aoi

    rt = []
    for t in recall_targets:
        hit = curve[curve.facility_recall >= t - 1e-12]
        r = hit.iloc[0] if len(hit) else None
        rt.append(dict(recall_target=t, reached=r is not None,
                       n_reviewed=None if r is None else int(r.n_reviewed),
                       area_reviewed_km2=None if r is None else float(r.area_reviewed_km2),
                       pct_aoi_reviewed=None if r is None else float(r.pct_aoi_reviewed),
                       workload_reduction=None if r is None else float(r.workload_reduction),
                       score_threshold=None if r is None else float(r.score_threshold)))
    summary = dict(n_units=len(d), total_unit_area_km2=float(d.area_km2.sum()), aoi_area_km2=aoi,
                   n_facilities=n_fac, tie_policy=tie_policy,
                   median_first_hit_rank_expected=float(first_hits.first_hit_rank_expected.median()),
                   mean_first_hit_rank_expected=float(first_hits.first_hit_rank_expected.mean()))
    return ReviewResult(curve, first_hits, pd.DataFrame(rt), summary)


def at_threshold(curve: pd.DataFrame, threshold: float) -> pd.Series:
    """State after reviewing every unit with score >= threshold (requires tie_policy='group')."""
    ok = curve[(curve.step == 0) | (curve.score_threshold >= threshold)]
    row = ok.iloc[-1].copy()
    row["surfaced_ids"] = ";".join(s for s in ok.newly_surfaced if s)
    return row


# ---------------------------------------------------------------------------------------------------
# Footprint rule on z16 parent tiles (common rule for CNN and VLM).

def footprint_inputs(parent_table: pd.DataFrame, facilities: pd.DataFrame, membership: pd.DataFrame,
                     region: str, include_candidates: bool = False):
    """Units / membership / denominator for one AOI under the parent-tile footprint rule."""
    units = parent_table[parent_table.region == region][["tile_id", "area_km2"]].rename(columns={"tile_id": "unit_id"})
    fac = facilities[facilities.region == region]
    sel = fac.known_target & (fac.in_primary_denominator | (include_candidates & fac.status.str.startswith("candidate")))
    denom = fac.facility_id[sel].tolist()
    mem = membership[(membership.region == region) & membership.facility_id.isin(denom)]
    return units, mem.rename(columns={"tile_id": "unit_id"})[["unit_id", "facility_id"]], denom


def evaluate_parent_scores(scores: pd.DataFrame, parent_table: pd.DataFrame, facilities: pd.DataFrame,
                           membership: pd.DataFrame, region: str, aoi_area_km2: float | None = None,
                           include_candidates: bool = False, tie_policy: str = "group",
                           recall_targets=(0.5, 0.75, 0.9, 1.0)) -> ReviewResult:
    """scores: [region, tile_id, score] for parent tiles (any model)."""
    units, mem, denom = footprint_inputs(parent_table, facilities, membership, region, include_candidates)
    s = scores[scores.region == region][["tile_id", "score"]].rename(columns={"tile_id": "unit_id"})
    return evaluate_review(s, units, mem, denom, aoi_area_km2, tie_policy, recall_targets)
