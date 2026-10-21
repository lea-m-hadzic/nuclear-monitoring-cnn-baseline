"""Offline tests for the external dataset plan (ingest, jitter, negatives, split) and request metering. No API calls."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXT = Path(__file__).resolve().parents[1] / "external_data"
sys.path.insert(0, str(EXT))

import dataset_plan as dp  # noqa: E402
import geometry  # noqa: E402
import vendor_runner  # noqa: E402
from geo import chebyshev_m, destination, haversine_m, point_pixel_in_tile  # noqa: E402

needs_vendor = pytest.mark.skipif(not geometry.vendor_script_path().exists(), reason="vendor checkout missing")


def write_source(tmp_path, rows, bom=True, header='"Name","Latitude","Longitude"'):
    p = tmp_path / "src.csv"
    text = header + "\n" + "\n".join(f'"{n}",{a},{b}' for n, a, b in rows) + "\n"
    p.write_bytes(("﻿" if bom else "").encode() + text.encode())
    return p


# Synthetic coordinates only (open land, far from every evaluation AOI); NOT from the real list.
SYN = [("Alpha Site", "40.1000000000001", "-100.2"), ("Alpha Site Store", "40.1030", "-100.2050"),
       ("Beta", "40.1500", "-100.1000"), ("Gamma", "-30.25", "140.5")]


def test_load_source_preserves_strings_and_handles_bom(tmp_path):
    df = dp.load_source(write_source(tmp_path, SYN))
    assert df["latitude_original"].tolist()[0] == "40.1000000000001"          # exact string kept
    assert df["latitude"].iloc[0] == pytest.approx(40.1)
    assert df["source_row"].tolist() == [2, 3, 4, 5]
    with pytest.raises(SystemExit):
        dp.load_source(write_source(tmp_path, SYN, header='"Name","Lon","Lat"'))


def test_ids_and_collisions():
    df = dp.assign_ids(pd.DataFrame({"facility_name_original": ["Ascó", "Asco", "Plant 404/A"]}))
    assert df["facility_id"].tolist() == ["ext_asco", "ext_asco_2", "ext_plant_404_a"]
    assert df["id_collision"].tolist() == [True, True, False]


def test_site_groups_single_linkage():
    df = pd.DataFrame({"facility_id": ["ext_a", "ext_b", "ext_c", "ext_d"],
                       "latitude": [0.0, 0.0, 0.0, 10.0], "longitude": [0.0, 0.1, 0.2, 0.0]})   # ~11 km steps
    g = dp.split_groups(df, link_km=15)
    assert g.tolist() == ["sg_a", "sg_a", "sg_a", "sg_d"]               # chained a-b-c, d alone


def test_destination_roundtrip():
    lat, lon = destination(45.0, 10.0, 90.0, 1000.0)
    assert haversine_m(45.0, 10.0, lat, lon) == pytest.approx(1000.0, abs=0.01)


@needs_vendor
def test_fence_pixel_geometry():
    c = dp.fence_pixel(40.0, -100.0, 40.0, -100.0)
    assert (c["fence_px_col"], c["fence_px_row"]) == (1280.0, 1280.0)
    lat, lon = destination(40.0, -100.0, 90.0, 350.0)                       # 350 m east of the centre
    e = dp.fence_pixel(40.0, -100.0, lat, lon)
    assert e["fence_px_col"] - 1280 == pytest.approx(350 / (1900 / 2560), rel=0.003)
    assert e["fence_px_row"] == pytest.approx(1280, abs=0.5)
    assert e["fence_edge_margin_m"] == pytest.approx(950 - 350, abs=2)


@needs_vendor
def test_plan_invariants(tmp_path):
    pos, tiles, summary, pairs, _ = dp.build_plan(write_source(tmp_path, SYN))
    cfg = geometry.load_config()["dataset"]
    # source coordinates untouched; jitter within bounds and deterministic
    assert pos["latitude_original"].tolist() == [r[1] for r in SYN]
    d = haversine_m(pos.latitude, pos.longitude, pos.center_lat, pos.center_lon)
    assert ((d >= cfg["jitter"]["min_m"] - 0.01) & (d <= cfg["jitter"]["max_m"] + 0.01)).all()
    assert np.allclose(d, pos["jitter_m"], atol=0.01)
    assert pos["jitter_ok"].all()
    pos2, tiles2, _, _, _ = dp.build_plan(write_source(tmp_path, SYN))
    assert tiles.equals(tiles2)
    # the two "Alpha" rows are one split group, kept in one split with their negatives
    assert pos.loc[0, "split_group_id"] == pos.loc[1, "split_group_id"]
    assert (tiles.groupby("split_group_id")["split"].nunique() == 1).all()
    # negatives: placed, outside every fence exclusion zone, no footprint overlap with any tile
    neg = tiles[tiles.role == "candidate_negative"]
    assert (neg.status == "planned").all() and len(neg) == cfg["negatives"]["per_positive"] * len(pos)
    dmin = haversine_m(neg.center_lat.to_numpy()[:, None], neg.center_lon.to_numpy()[:, None],
                       pos.latitude.to_numpy()[None, :], pos.longitude.to_numpy()[None, :]).min(1)
    assert (dmin >= cfg["negatives"]["exclusion_km"] * 1000 - 0.5).all()
    assert summary["checks"]["negative_overlapping_any_tile"] == 0
    assert summary["checks"]["overlaps_across_splits"] == 0
    assert (neg.label_status == "candidate_negative").all()


@needs_vendor
def test_no_jitter_ablation(tmp_path):
    pos, tiles, _, _, _ = dp.build_plan(write_source(tmp_path, SYN), jitter=False)
    assert (pos.center_lat == pos.latitude).all() and (pos.jitter_m == 0).all()
    assert (pos.fence_px_col == 1280).all()


@needs_vendor
def test_aoi_proximity_flag_and_negative_exclusion(tmp_path):
    aoi = geometry.load_config()["evaluation_aois"]["Saratov"]
    lat, lon = destination(aoi["lat"], aoi["lon"], 0.0, (aoi["radius_km"] + 28) * 1000)   # just outside the buffer
    pos, tiles, summary, _, _ = dp.build_plan(write_source(tmp_path, [("Near Saratov", f"{lat:.6f}", f"{lon:.6f}")]))
    neg = tiles[(tiles.role == "candidate_negative") & (tiles.status == "planned")]
    d = haversine_m(neg.center_lat, neg.center_lon, aoi["lat"], aoi["lon"]) / 1000
    assert (d >= aoi["radius_km"] + geometry.load_config()["aoi_exclusion_buffer_km"]).all()
    assert "same_region_as_Saratov" in pos.loc[0, "flags"]


def test_vendor_runner_counts_only_staticmap_calls(tmp_path, monkeypatch):
    script = tmp_path / "fake_vendor.py"
    script.write_text("import requests\ns = requests.Session()\n"
                      "s.get('https://maps.googleapis.com/maps/api/staticmap?center=1,2&key=X')\n"
                      "s.get('https://maps.googleapis.com/maps/api/staticmap?center=3,4&key=X')\n"
                      "s.get('https://example.org/other')\nraise SystemExit(0)\n")
    monkeypatch.setattr(vendor_runner, "_orig_get", lambda self, url, *a, **k: None)   # no network
    monkeypatch.setattr(vendor_runner, "_count", 0)
    monkeypatch.setattr(sys, "argv", ["vendor_runner.py", str(script)])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = vendor_runner.main()
    assert rc == 0 and buf.getvalue().strip().endswith(f"{vendor_runner.COUNT_TAG}2")
    assert "key=" not in buf.getvalue()


def test_chebyshev_overlap_rule():
    lat, lon = destination(10.0, 10.0, 45.0, 2000.0)          # 2 km diagonal: ~1.41 km per axis -> overlap
    assert chebyshev_m(10.0, 10.0, lat, lon) < 1900
    lat, lon = destination(10.0, 10.0, 90.0, 2000.0)          # 2 km east -> no overlap
    assert chebyshev_m(10.0, 10.0, lat, lon) > 1900


def test_point_pixel_centre_is_centre():
    assert point_pixel_in_tile(51.0, 7.0, 51.0, 7.0, 17, 2, 3885, 2560) == (1280.0, 1280.0)


@needs_vendor
def test_ids_one_to_one_groups_separate_and_jitter_per_point(tmp_path):
    pos, tiles, summary, _, groups = dp.build_plan(write_source(tmp_path, SYN))
    assert pos["facility_id"].is_unique and len(pos) == len(SYN)                    # names never merged
    assert pos.loc[0, "split_group_id"] == pos.loc[1, "split_group_id"]               # co-located rows grouped
    assert pos.loc[0, "facility_id"] != pos.loc[1, "facility_id"]
    assert (pos.loc[0, "jitter_bearing_deg"], pos.loc[0, "jitter_m"]) != (pos.loc[1, "jitter_bearing_deg"], pos.loc[1, "jitter_m"])
    assert set(groups.columns) >= {"split_group_id", "facility_ids", "linking_pairs_lt_threshold"}
    neg = tiles[tiles.role == "candidate_negative"]
    per = neg.groupby("parent_point_id").candidate_subset.value_counts().unstack()
    assert (per["primary"] == 3).all() and (per["reserve"] == 1).all()
    assert set(neg.loc[neg.candidate_subset == "reserve", "tile_id"].str[-4:]) == {"neg4"}
