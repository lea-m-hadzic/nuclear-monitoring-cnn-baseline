"""Small spherical / Web-Mercator helpers (no pyproj needed; accuracy is far better than a pixel at 1-10 km)."""
from __future__ import annotations

import hashlib
import math

import numpy as np

EARTH_R_M = 6371008.8                 # same mean radius as the vendor script's local_xy_to_latlon
WEB_MERC_R_M = 6378137.0              # Web Mercator sphere (Google tiles)
GOOGLE_BASE_MPP = 156543.03392        # m/px at zoom 0, equator, scale 1 (vendor constant)


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def destination(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    """Great-circle destination point."""
    p1, l1, b, d = math.radians(lat), math.radians(lon), math.radians(bearing_deg), dist_m / EARTH_R_M
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def local_dxdy_m(lat0, lon0, lat, lon):
    """East/north offsets (m) of (lat, lon) from (lat0, lon0) on the local tangent plane."""
    dlon = (np.asarray(lon) - lon0 + 540) % 360 - 180
    dx = np.radians(dlon) * EARTH_R_M * np.cos(np.radians(lat0))
    dy = np.radians(np.asarray(lat) - lat0) * EARTH_R_M
    return dx, dy


def chebyshev_m(lat0, lon0, lat, lon):
    """Axis-aligned separation; two square tiles of side s overlap iff this is < s."""
    dx, dy = local_dxdy_m(lat0, lon0, lat, lon)
    return np.maximum(np.abs(dx), np.abs(dy))


def merc_xy(lat, lon):
    lat = np.clip(np.asarray(lat, dtype=float), -85.05112878, 85.05112878)
    return WEB_MERC_R_M * np.radians(lon), WEB_MERC_R_M * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def point_pixel_in_tile(center_lat, center_lon, lat, lon, zoom: int, scale: int, crop_px: int, output_px: int):
    """(col, row) of (lat, lon) in an auto_mosaic output tile centred on (center_lat, center_lon).

    The source imagery is Web Mercator: at `zoom`, one returned pixel spans GOOGLE_BASE_MPP / 2**zoom / scale
    Mercator metres. auto_mosaic centre-crops `crop_px` source pixels and resizes them to `output_px`.
    """
    cx, cy = merc_xy(center_lat, center_lon)
    px, py = merc_xy(lat, lon)
    merc_per_px = GOOGLE_BASE_MPP / 2 ** zoom / scale
    k = output_px / crop_px
    return output_px / 2 + (px - cx) / merc_per_px * k, output_px / 2 - (py - cy) / merc_per_px * k


def unit_floats(*parts, n: int = 2) -> list[float]:
    """n deterministic floats in [0, 1) from a SHA-256 of the joined parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return [int.from_bytes(h[8 * i:8 * i + 8], "big") / 2 ** 64 for i in range(n)]
