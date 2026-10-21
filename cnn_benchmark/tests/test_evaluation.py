"""Generic facility-level evaluator: synthetic correctness, tie policies, invariants on real VLM scores."""
import numpy as np
import pandas as pd
import pytest

from evaluation import at_threshold, evaluate_parent_scores, evaluate_review


def toy(scores):
    units = pd.DataFrame({"unit_id": list("abcdef"), "area_km2": [1.0] * 6})
    s = pd.DataFrame({"unit_id": list("abcdef"), "score": scores})
    mem = pd.DataFrame({"unit_id": ["b", "c", "d"], "facility_id": ["F1", "F2", "F2"]})
    return s, units, mem


def check_invariants(res, n_fac):
    c = res.curve
    assert (np.diff(c.n_reviewed) > 0).all()
    assert (np.diff(c.area_reviewed_km2) >= 0).all()
    assert (np.diff(c.facility_recall) >= 0).all()
    assert c.facilities_surfaced.max() <= n_fac
    assert c.facility_recall.iloc[-1] == pytest.approx(1.0)
    surfaced = [f for s in c.newly_surfaced if s for f in s.split(";")]
    assert len(surfaced) == len(set(surfaced)) == n_fac          # each facility counted exactly once


def test_basic_ranking_and_first_hits():
    s, u, m = toy([9, 8, 7, 1, 0, -1])
    r = evaluate_review(s, u, m, ["F1", "F2"], aoi_area_km2=6)
    check_invariants(r, 2)
    fh = r.first_hits.set_index("facility_id")
    assert fh.first_hit_rank_group_end.to_dict() == {"F1": 2, "F2": 3}
    assert fh.loc["F2", "n_footprint_units"] == 2
    assert at_threshold(r.curve, 7).facilities_surfaced == 2
    assert at_threshold(r.curve, 8).facilities_surfaced == 1
    rt = r.recall_targets.set_index("recall_target")
    assert rt.loc[0.5, "area_reviewed_km2"] == 2 and rt.loc[1.0, "area_reviewed_km2"] == 3
    assert rt.loc[1.0, "workload_reduction"] == pytest.approx(0.5)


def test_multi_tile_facility_counted_once():
    s, u, m = toy([1, 0, 9, 8, 0, 0])
    r = evaluate_review(s, u, m, ["F1", "F2"], aoi_area_km2=6, tie_policy="optimistic")
    # order: c(9)->F2, d(8) F2 again (not re-counted), a(1), then tie group 0 with b first -> F1
    assert r.curve.facilities_surfaced.tolist() == [0, 1, 1, 1, 2, 2, 2]


def test_tie_policies():
    s, u, m = toy([5, 5, 5, 5, 0, 0])            # b, c, d (facilities) tie with a
    g = evaluate_review(s, u, m, ["F1", "F2"], aoi_area_km2=6, tie_policy="group")
    assert g.curve.n_reviewed.tolist() == [0, 4, 6]        # one step per tie group
    opt = evaluate_review(s, u, m, ["F1", "F2"], aoi_area_km2=6, tie_policy="optimistic")
    pes = evaluate_review(s, u, m, ["F1", "F2"], aoi_area_km2=6, tie_policy="pessimistic")
    assert opt.curve.facilities_surfaced.tolist()[:3] == [0, 1, 2]
    assert pes.curve.facilities_surfaced.tolist()[:3] == [0, 0, 1]
    fh = g.first_hits.set_index("facility_id")
    # F1: 1 footprint unit in a group of 4 -> expected rank 5/2; F2: 2 units -> 5/3
    assert fh.loc["F1", "first_hit_rank_expected"] == pytest.approx(2.5)
    assert fh.loc["F2", "first_hit_rank_expected"] == pytest.approx(5 / 3)
    assert fh.loc["F2", "first_hit_rank_optimistic"] == 1 and fh.loc["F2", "first_hit_rank_pessimistic"] == 3
    assert fh.loc["F1", "first_hit_rank_group_end"] == 4


def test_area_weighting_and_pct():
    u = pd.DataFrame({"unit_id": ["p", "q"], "area_km2": [3.61, 0.9025]})
    s = pd.DataFrame({"unit_id": ["p", "q"], "score": [1, 2]})
    m = pd.DataFrame({"unit_id": ["p"], "facility_id": ["F"]})
    r = evaluate_review(s, u, m, ["F"], aoi_area_km2=100)
    row = r.first_hits.iloc[0]
    assert row.first_hit_area_group_end_km2 == pytest.approx(4.5125)
    assert row.first_hit_pct_aoi_group_end == pytest.approx(4.5125)


def test_input_validation():
    s, u, m = toy([1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="no score"):
        evaluate_review(s.iloc[:-1], u, m, ["F1"])
    with pytest.raises(ValueError, match="not units"):
        evaluate_review(pd.concat([s, pd.DataFrame({"unit_id": ["zz"], "score": [0]})]), u, m, ["F1"])
    with pytest.raises(ValueError, match="without footprint"):
        evaluate_review(s, u, m, ["F1", "F9"])
    with pytest.raises(ValueError, match="NaN"):
        evaluate_review(s.assign(score=np.nan), u, m, ["F1"])


@pytest.mark.parametrize("seed", range(5))
def test_invariants_random_scores(seed, parent_table, facilities, membership):
    rng = np.random.default_rng(seed)
    for region in ["Saratov", "Natanz", "Fordow"]:
        s = parent_table[["region", "tile_id"]].assign(score=rng.normal(size=len(parent_table)))
        r = evaluate_parent_scores(s, parent_table, facilities, membership, region)
        check_invariants(r, r.summary["n_facilities"])


def test_invariants_vlm_integer_scores(parent_table, facilities, membership):
    for col in ["vlm_score_single_pass", "vlm_score_adaptive_max"]:
        s = parent_table.rename(columns={col: "score"})[["region", "tile_id", "score"]]
        for region in ["Saratov", "Natanz", "Fordow"]:
            for tie in ["group", "optimistic", "pessimistic"]:
                r = evaluate_parent_scores(s, parent_table, facilities, membership, region, tie_policy=tie)
                check_invariants(r, r.summary["n_facilities"])
                fh = r.first_hits
                assert (fh.first_hit_rank_optimistic <= fh.first_hit_rank_expected).all()
                assert (fh.first_hit_rank_expected <= fh.first_hit_rank_pessimistic).all()
                assert (fh.first_hit_rank_pessimistic <= fh.first_hit_rank_group_end).all()


def test_candidate_facility_toggle(parent_table, facilities, membership):
    s = parent_table.rename(columns={"vlm_score_single_pass": "score"})[["region", "tile_id", "score"]]
    base = evaluate_parent_scores(s, parent_table, facilities, membership, "Saratov")
    incl = evaluate_parent_scores(s, parent_table, facilities, membership, "Saratov", include_candidates=True)
    assert base.summary["n_facilities"] == 24 and incl.summary["n_facilities"] == 25


def test_scores_must_cover_every_parent(parent_table, facilities, membership):
    s = parent_table[["region", "tile_id"]].assign(score=0.0).iloc[1:]
    with pytest.raises(ValueError):
        evaluate_parent_scores(s, parent_table, facilities, membership, "Saratov")
