"""Canonical parent table: counts, target provenance, aggregation rules, and flags."""
import numpy as np
import pandas as pd
import pytest

from build_parent_table import aggregate_parent_targets
from common import REPO_ROOT, load_raw_sheets, load_vlm_csv

EXPECTED_PARENTS = {"Saratov": 1387, "Natanz": 776, "Fordow": 777}
# Regression values for the human parent target under the documented aggregation rule.
EXPECTED_POSITIVES = {"Saratov": 102, "Natanz": 8, "Fordow": 36}
NATANZ_CONFLICTS = {"r3_c9", "r7_c14", "r8_c15", "r15_c17"}


def test_parent_counts_by_aoi(parent_table):
    assert parent_table.groupby("region").size().to_dict() == EXPECTED_PARENTS
    assert not parent_table.duplicated(["region", "tile_id"]).any()


def test_parent_ids_match_vlm_csv_manifest(parent_table, cfg):
    for region in EXPECTED_PARENTS:
        csv_ids = set(load_vlm_csv(region, cfg).tile_id)
        assert set(parent_table.tile_id[parent_table.region == region]) == csv_ids
    assert ((parent_table.row == parent_table.manifest_row) & (parent_table.col == parent_table.manifest_col)).all()


def test_no_child_row_becomes_a_parent_sample(parent_table, processed):
    assert parent_table.tile_id.str.fullmatch(r"r\d+_c\d+").all()
    z16 = processed[(processed.zoom == 16)]           # record IDs are unique within a region only
    key = lambda d: set(zip(d.region, d.record_id))
    parent_ids = key(z16[z16.stage.isin(["Original output", "Replaced parent"])])
    child_ids = key(z16[z16.stage.isin(["Quadrant output", "Zoom-out mosaic block"])])
    table_ids = set(zip(parent_table.region, parent_table.parent_record_id))
    assert table_ids == parent_ids
    assert not table_ids & child_ids
    used_children = {(r, int(x)) for r, s in zip(parent_table.region, parent_table.child_record_ids)
                     for x in str(s).split(";") if x and x != "nan"}
    assert used_children == child_ids


def test_target_does_not_depend_on_vlm_score(processed):
    z16 = processed[processed.zoom == 16]
    base = aggregate_parent_targets(z16)
    # 1) score column removed entirely; 2) scores scrambled
    no_score = aggregate_parent_targets(z16.drop(columns=["score"]))
    rng = np.random.default_rng(0)
    scrambled = aggregate_parent_targets(z16.assign(score=rng.permutation(z16.score.to_numpy())))
    for other in (no_score, scrambled):
        pd.testing.assert_series_equal(base.human_target, other.human_target)


def test_target_matches_tp_fn_semantics(parent_table):
    orig = parent_table[parent_table.target_source == "parent_label"]
    assert (orig.human_target == orig.parent_own_label.str.match(r"^(TP|FN)").astype(int)).all()
    quad = parent_table[parent_table.target_source == "quadrant_union"]
    any_pos = quad.quadrant_labels.str.contains(r"=(?:TP|FN)\b")
    assert (quad.human_target == any_pos.astype(int)).all()


def test_positive_counts_regression(parent_table):
    assert parent_table.groupby("region").human_target.sum().to_dict() == EXPECTED_POSITIVES


def test_label_conflicts_are_flagged_not_fixed(parent_table):
    c = parent_table[parent_table.label_conflict]
    assert set(zip(c.region, c.tile_id)) == {("Natanz", t) for t in NATANZ_CONFLICTS}
    assert c.exclude_from_training.all() and c.needs_author_confirmation.all()
    assert c.training_target.isna().all()
    assert (c.human_target == 1).all() and (c.parent_own_label == "TN").all()   # quadrant union kept, conflict visible


def test_zoomout_only_parents_flagged(parent_table):
    z = parent_table[parent_table.target_source == "replaced_parent_label_zoomout_only"]
    assert set(zip(z.region, z.tile_id)) == {("Natanz", "r15_c2"), ("Fordow", "r4_c7"), ("Fordow", "r9_c2")}
    assert z.needs_author_confirmation.all()


def test_aggregation_rules_on_synthetic_rows():
    def row(tile, rid, stage, label):
        return dict(region="X", parent_tile_id=tile, record_id=rid, stage=stage, human_label=label,
                    human_positive=label[:2] in ("TP", "FN"), label_detail=label[5:] if " - " in label else "",
                    row_tile_id=tile + ("" if stage in ("Original output", "Replaced parent") else ", NW"),
                    row=1, col=1)
    rows = pd.DataFrame([
        row("r1_c1", 1, "Original output", "FN - silo"),
        row("r1_c2", 2, "Replaced parent", "FN"), row("r1_c2", 3, "Quadrant output", "TN"),
        row("r1_c2", 4, "Quadrant output", "TP - prison"),
        row("r1_c3", 5, "Replaced parent", "TN"), row("r1_c3", 6, "Quadrant output", "FN - silo - cut off"),
        row("r1_c4", 7, "Replaced parent", "TN"), row("r1_c4", 8, "Zoom-out mosaic block", "FN - oil"),
        row("r1_c5", 9, "Original output", "FP - single"),
    ])
    t = aggregate_parent_targets(rows).set_index("tile_id")
    assert t.human_target.to_dict() == {"r1_c1": 1, "r1_c2": 1, "r1_c3": 1, "r1_c4": 0, "r1_c5": 0}
    assert t.label_conflict.to_dict() == {"r1_c1": False, "r1_c2": False, "r1_c3": True, "r1_c4": False, "r1_c5": False}
    assert t.loc["r1_c3", "ambiguous_tags"] == "cut_off"
    assert "zoomout_block_positive_not_aggregated" in t.loc["r1_c4", "aux_flags"]
    with pytest.raises(ValueError):
        aggregate_parent_targets(pd.concat([rows, rows.iloc[[0]]]))   # duplicated parent row


def test_workbook_raw_sheets_match_processed(processed, cfg):
    raw = load_raw_sheets(cfg)
    m = raw.merge(processed, left_on=["region", "zoom", "Record ID"], right_on=["region", "zoom", "record_id"])
    assert len(m) == len(raw) == len(processed)
    assert (m["Human label"].str.strip() == m.human_label).all()
    assert (m["Score"] == m.score).all() and (m["Tile ID"] == m.row_tile_id).all()


def test_image_references(parent_table):
    assert not parent_table.source_image_relpath.duplicated().any()
    assert all((REPO_ROOT / p).is_file() for p in parent_table.thumbnail_relpath)
