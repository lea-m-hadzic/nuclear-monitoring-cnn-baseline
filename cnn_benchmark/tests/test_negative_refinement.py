"""Offline tests: automatic rejection, nearby-negative selection, pixel statistics, hard-negative validation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

EXT = Path(__file__).resolve().parents[1] / "external_data"
sys.path.insert(0, str(EXT))

import geometry  # noqa: E402
import hard_negatives as hn  # noqa: E402
import negative_refinement as nr  # noqa: E402
from dataset_audit import PLACEHOLDER_RGB  # noqa: E402
from geo import destination  # noqa: E402


def feat(**kw):
    base = dict(decode_error="", placeholder_frac=0.0, black_frac=0.0, water_frac=0.0)
    return pd.Series({**base, **kw})


def test_auto_reject_reasons():
    assert nr.auto_reject(feat()) == []
    assert nr.auto_reject(feat(placeholder_frac=0.9, water_frac=0.0)) == ["full_placeholder"]
    assert nr.auto_reject(feat(placeholder_frac=0.1, black_frac=0.05)) == ["partial_placeholder", "black_nodata"]
    assert nr.auto_reject(feat(water_frac=0.7)) == ["mostly_water"]
    assert nr.auto_reject(feat(placeholder_frac=0.9, water_frac=0.9)) == ["full_placeholder"]   # not double-counted
    assert nr.auto_reject(feat(decode_error="OSError")) == ["corrupt"]


def test_select_nearby_fills_from_reserve_and_caps_at_three():
    rows = []
    for p, rejected in (("A__pos", {"A__neg2"}), ("B__pos", set()), ("C__pos", {"C__neg1", "C__neg2", "C__neg3", "C__neg4"})):
        f = p.split("__")[0]
        for k in range(1, 5):
            t = f"{f}__neg{k}"
            rows.append(dict(tile_id=t, parent_point_id=p, candidate_subset="primary" if k <= 3 else "reserve",
                             auto_reject=t in rejected))
    s = nr.select_nearby(pd.DataFrame(rows)).set_index("tile_id")
    assert s.loc[["A__neg1", "A__neg3", "A__neg4"], "nearby_negative_selected"].all()       # reserve fills the gap
    assert not s.loc["A__neg2", "nearby_negative_selected"]
    assert s.loc["A__neg4", "selection_rank"] == 3
    assert s.loc[["B__neg1", "B__neg2", "B__neg3"], "nearby_negative_selected"].all()
    assert not s.loc["B__neg4", "nearby_negative_selected"]                                  # cap at 3
    assert not s.loc[s.index.str.startswith("C__"), "nearby_negative_selected"].any()        # shortage stays explicit


def _png(tmp_path, arr, name):
    p = tmp_path / name
    Image.fromarray(arr.astype(np.uint8)).save(p)
    return str(p)


def test_tile_features_detect_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    land = np.clip(np.array([95, 110, 70]) + rng.normal(0, 18, (640, 640, 3)), 0, 255)   # textured green-brown land
    ph = np.zeros((640, 640, 3)); ph[:] = PLACEHOLDER_RGB
    black = land.copy(); black[:200, :200] = 0
    water = np.zeros((640, 640, 3)); water[:] = (20, 50, 90)
    f = {k: nr.tile_features(_png(tmp_path, a, f"{k}.png")) for k, a in
         (("land", land), ("ph", ph), ("black", black), ("water", water))}
    assert f["ph"]["placeholder_frac"] > 0.99 and f["land"]["placeholder_frac"] < 0.01
    assert f["black"]["black_frac"] == pytest.approx((200 / 640) ** 2, abs=0.01)
    assert f["water"]["water_frac"] > 0.95 and f["land"]["water_frac"] < 0.05
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    assert nr.tile_features(str(bad))["decode_error"]


def test_region_quotas_sum_and_proportional():
    pos = pd.DataFrame({"broad_region": ["europe"] * 6 + ["east_asia"] * 3 + ["south_asia"]})
    q = hn.region_quotas(pos, 10)
    assert q.sum() == 10 and q["europe"] == 6 and q["east_asia"] == 3


@pytest.mark.skipif(not geometry.vendor_script_path().exists(), reason="vendor checkout missing")
def test_validate_hard_negative_points():
    cfg = geometry.load_config()
    pos = pd.DataFrame({"facility_id": ["ext_x"], "latitude": [40.0], "longitude": [-100.0]})
    far_lat, far_lon = destination(40.0, -100.0, 90, 50_000)
    near_lat, near_lon = destination(40.0, -100.0, 90, 10_000)
    sar = cfg["evaluation_aois"]["Saratov"]
    pts = pd.DataFrame({"hn_id": ["a", "b", "c", "d"], "name": ["Big Coal Plant", "Steel works", "Atom Plant", "Depot"],
                        "category": ["thermal_power_plant"] * 4,
                        "latitude": [far_lat, near_lat, far_lat + 1, sar["lat"]],
                        "longitude": [far_lon, near_lon, far_lon, sar["lon"]]})
    v = hn.validate_points(pts, pos, cfg).set_index("hn_id")
    assert v.loc["a", "valid"]
    assert "too_close_to_positive" in v.loc["b", "flags"]
    assert "nuclear_like_name" in v.loc["c", "flags"]
    assert "inside_or_near_Saratov_aoi" in v.loc["d", "flags"]
    split = hn.assign_hard_negative_split(v.reset_index(), cfg)
    assert set(split) <= {"train", "val"}
