"""Offline tests for the hard-negative pilot helpers (no Overpass / Static Maps calls)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXT = Path(__file__).resolve().parents[1] / "external_data"
sys.path.insert(0, str(EXT))

import geometry  # noqa: E402
import hard_negative_pilot as hp  # noqa: E402
from geo import destination  # noqa: E402

needs_vendor = pytest.mark.skipif(not geometry.vendor_script_path().exists(), reason="vendor checkout missing")


def test_bbox_area_and_distance():
    b = {"minlat": 0.0, "maxlat": 0.009, "minlon": 0.0, "maxlon": 0.009}       # ~1 km x 1 km at the equator
    assert hp.bbox_area_ha(b) == pytest.approx(100, rel=0.02)
    assert hp.dist_to_bbox_m(0.0045, 0.0045, b) == 0.0                          # inside
    assert hp.dist_to_bbox_m(0.0045, 0.0180, b) == pytest.approx(1001, rel=0.01)


def test_excluded_nearby_uses_area_edges():
    lat, lon = 10.0, 10.0
    big_base = {"bounds": {"minlat": 9.0, "maxlat": 10.0, "minlon": 9.0, "maxlon": 10.0 - 0.01}, "tags": {"landuse": "military"}}
    assert hp.excluded_nearby(lat, lon, [big_base]).startswith("excluded_near:landuse=military")   # ~1.1 km from edge
    far = {"bounds": {"minlat": 9.0, "maxlat": 9.5, "minlon": 9.0, "maxlon": 9.5}, "tags": {"amenity": "prison"}}
    assert hp.excluded_nearby(lat, lon, [far]) == ""


@needs_vendor
def test_candidate_ok_rules():
    cfg = geometry.load_config()
    pos = pd.DataFrame({"latitude": [40.0], "longitude": [-100.0]})
    far = destination(40.0, -100.0, 0, 60_000)
    assert hp.candidate_ok(*far, "Big Gas Plant", pos, [], cfg) == ""
    assert hp.candidate_ok(*destination(40.0, -100.0, 0, 10_000), "x", pos, [], cfg) == "too_close_to_positive"
    assert hp.candidate_ok(*far, "Atomic Energy Works", pos, [], cfg) == "nuclear_like_name"
    acc = [{"latitude": far[0], "longitude": far[1]}]
    near_acc = destination(*far, 90, 1_000)
    assert hp.candidate_ok(*near_acc, "Depot", pos, acc, cfg) == "co_located_with_accepted"
    sar = cfg["evaluation_aois"]["Saratov"]
    assert hp.candidate_ok(sar["lat"], sar["lon"], "x", pos, [], cfg) == "near_Saratov_aoi"


def test_mixed_set_is_one_to_one_per_split():
    near = pd.DataFrame({"tile_id": [f"n{i}" for i in range(40)], "split": ["train"] * 30 + ["val"] * 10})
    hard = pd.DataFrame({"tile_id": [f"h{i}" for i in range(9)], "split": ["train"] * 6 + ["val"] * 3})
    mix = hp.mixed_set(100, near, hard, "seed")
    c = mix.assign(kind=mix.tile_id.str[0]).groupby(["split", "kind"]).size()
    assert c[("train", "h")] == c[("train", "n")] == 6 and c[("val", "h")] == c[("val", "n")] == 3
    assert mix.equals(hp.mixed_set(100, near, hard, "seed"))                                      # deterministic


def test_diag_row_refuses_without_confirmed_negatives():
    df = pd.DataFrame({"label_role": ["positive"] * 20 + ["hard_negative"] * 3, "split": ["train"] * 23})
    assert hp.diag_row("B", df, "")["status"].startswith("not run")
