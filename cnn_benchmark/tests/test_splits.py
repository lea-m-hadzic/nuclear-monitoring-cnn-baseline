"""Candidate split manifests: leakage checks and determinism."""
import pandas as pd
import pytest

from splits import check_manifest, loao, saratov_spatial_cv


@pytest.fixture(scope="module")
def manifests(parent_table, membership, cfg):
    return saratov_spatial_cv(parent_table, membership, cfg), loao(parent_table, membership, cfg)


def test_manifests_pass_leakage_checks(manifests, parent_table, membership, cfg):
    for man in manifests:
        assert check_manifest(man, parent_table, membership, cfg["splits"]["buffer_tiles"]) == []


def test_checker_detects_leakage(manifests, parent_table, membership, cfg):
    cv = manifests[0].copy()
    # put one buffer tile into train -> adjacency violation must be detected
    i = cv.index[(cv.fold == 0) & (cv.role == "buffer")][0]
    cv.loc[i, "role"] = "train"
    assert any("within" in p for p in check_manifest(cv, parent_table, membership, 1))
    # split a multi-tile facility across roles -> must be detected
    cv = manifests[0].copy()
    tiles = membership[(membership.facility_id == "SAR-WHS-03")].tile_id
    j = cv.index[(cv.fold == 0) & cv.tile_id.isin(tiles)][0]
    cv.loc[j, "role"] = "train" if cv.loc[j, "role"] != "train" else "test"
    assert any("SAR-WHS-03" in p for p in check_manifest(cv, parent_table, membership, 1))
    # a z17 / quadrant id must be rejected
    bad = pd.concat([manifests[0], manifests[0].iloc[[0]].assign(tile_id="r10_c18, NW")])
    assert any("not z16 parents" in p for p in check_manifest(bad, parent_table, membership, 1))


def test_loao_structure(manifests):
    lo = manifests[1]
    for fold, test_region in enumerate(["Natanz", "Fordow", "Saratov"]):
        f = lo[lo.fold == fold]
        assert set(f.region[f.role == "test"]) == {test_region}
        assert set(f.role[f.region == test_region]) == {"test"}


def test_splits_deterministic(parent_table, membership, cfg, manifests):
    again = saratov_spatial_cv(parent_table, membership, cfg)
    pd.testing.assert_frame_equal(again, manifests[0])
