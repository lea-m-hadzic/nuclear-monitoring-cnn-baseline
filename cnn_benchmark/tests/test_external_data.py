"""Offline tests for the external-data tooling (no Google API calls). Geometry tests need the vendor checkout."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

EXT = Path(__file__).resolve().parents[1] / "external_data"
sys.path.insert(0, str(EXT))

import geometry  # noqa: E402
from intake import COLUMNS, validate_intake  # noqa: E402
from secrets_guard import contains_secret, redact, sanitize_manifest  # noqa: E402

needs_vendor = pytest.mark.skipif(not geometry.vendor_script_path().exists(), reason="vendor checkout missing")
FAKE_KEY = "AIza" + "0" * 35


@needs_vendor
@pytest.mark.parametrize("preset,lat,zoom,grid,req,footprint", [
    # values recorded in the benchmark manifests
    ("legacy_mosaic2x2_z16", 51.52, 16, 2, 5, 1.9025),    # Saratov: returned_m_per_px 0.74316, width 1.90249 km
    ("auto_mosaic_1p9km", 33.771261, 17, 3, 11, 1.9),     # Natanz: z17, mosaic_grid_n 3
    ("auto_mosaic_1p9km", 35.016697, 17, 4, 18, 1.9),     # Fordow: z17, mosaic_grid_n 4
])
def test_geometry_matches_benchmark_manifests(preset, lat, zoom, grid, req, footprint):
    g = geometry.tile_geometry(preset, lat)
    assert (g.zoom, g.grid_n, g.requests_per_tile) == (zoom, grid, req)
    assert g.footprint_width_km == pytest.approx(footprint, abs=5e-4)
    assert g.output_px == 2560


@needs_vendor
def test_legacy_footprint_depends_on_latitude():
    assert geometry.tile_geometry("legacy_mosaic2x2_z16", 35.0).footprint_width_km == pytest.approx(2.50, abs=0.01)
    assert geometry.tile_geometry("auto_mosaic_1p9km", 35.0).footprint_width_km == 1.9


@needs_vendor
def test_tile_counts():
    assert [geometry.n_tiles(r, 1.9) for r in (0.001, 1.9, 2.7, 4.0)] == [1, 5, 9, 13]


def test_tiered_cost():
    pricing = {"free_requests_per_month": 10000, "tiers_usd_per_1000": [[100000, 2.0], [None, 1.0]]}
    assert geometry.tiered_cost_usd(10000, pricing) == 0
    assert geometry.tiered_cost_usd(20000, pricing) == pytest.approx(20.0)
    assert geometry.tiered_cost_usd(1000, pricing, already_used_this_month=99500) == pytest.approx(1.5)


def test_intake_validation():
    good = pd.DataFrame([["p1", "f1", "Name", "XX", "10.5", "-20.25", "fence_segment", "src", ""]], columns=COLUMNS)
    assert validate_intake(good) == []
    bad = good.assign(point_id="has space", latitude="91", coordinate_type="fence")
    assert len(validate_intake(bad)) == 3
    assert any("duplicate point_id" in e for e in validate_intake(pd.concat([good, good])))


def test_secrets_guard(tmp_path):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center=1,2&zoom=16&key={FAKE_KEY}"
    assert FAKE_KEY not in redact(url) and "key=REDACTED" in redact(url)
    assert contains_secret(url) and not contains_secret(redact(url))
    m = tmp_path / "manifest.csv"
    m.write_text(f'row,url,message\n1,{url},"403 Client Error: Forbidden for url: {url}"\n')
    assert sanitize_manifest(m) == 2
    assert not contains_secret(m.read_text())
