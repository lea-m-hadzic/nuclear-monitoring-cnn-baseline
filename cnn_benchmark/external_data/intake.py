"""Schema and validation for the external coordinate list (`external_facilities.csv`).

One row per POINT. A facility with several layered-fence segments gets several rows sharing `facility_id`.

| column          | required | meaning |
|-----------------|----------|---------|
| point_id        | yes      | unique, filesystem-safe id of this coordinate (e.g. `<facility_id>_p01`) |
| facility_id     | yes      | groups points of one facility; the leakage unit for splits |
| facility_name   | yes      | human-readable name |
| country         | yes      | ISO 3166-1 alpha-2 preferred |
| latitude        | yes      | WGS84 decimal degrees, -90..90 |
| longitude       | yes      | WGS84 decimal degrees, -180..180 |
| coordinate_type | yes      | `fence_segment` (centred ON a layered-fence line), `facility_centroid`, or `other` |
| label_source    | yes      | who/what established the label (e.g. "Yanliang manual review 2026-10, Google Earth") |
| notes           | no       | free text (imagery date seen, fence description, caveats) |

Extraction centres one 1.9 km tile on (latitude, longitude). For `fence_segment` points the fence is therefore at
the tile centre by construction; that is recorded per tile so it can be used (or ignored) later.
"""
from __future__ import annotations

import re

import pandas as pd

REQUIRED = ["point_id", "facility_id", "facility_name", "country", "latitude", "longitude",
            "coordinate_type", "label_source"]
COLUMNS = REQUIRED + ["notes"]
COORDINATE_TYPES = {"fence_segment", "facility_centroid", "other"}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def load_intake(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    errors = validate_intake(df)
    if errors:
        raise SystemExit(f"{path}: {len(errors)} problem(s):\n  " + "\n  ".join(errors[:50]))
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    return df


def validate_intake(df: pd.DataFrame) -> list[str]:
    errors = [f"missing column {c!r}" for c in REQUIRED if c not in df.columns]
    if errors:
        return errors
    for i, r in df.iterrows():
        where = f"row {i + 2} ({r['point_id'] or '?'})"
        for c in REQUIRED:
            if not str(r[c]).strip():
                errors.append(f"{where}: empty {c}")
        for c in ("point_id", "facility_id"):
            if r[c] and not ID_RE.match(r[c]):
                errors.append(f"{where}: {c} {r[c]!r} must match {ID_RE.pattern} (used as a directory name)")
        try:
            lat, lon = float(r["latitude"]), float(r["longitude"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                errors.append(f"{where}: latitude/longitude out of range")
        except ValueError:
            errors.append(f"{where}: latitude/longitude must be decimal degrees")
        if r["coordinate_type"] and r["coordinate_type"] not in COORDINATE_TYPES:
            errors.append(f"{where}: coordinate_type must be one of {sorted(COORDINATE_TYPES)}")
    dup = df["point_id"][df["point_id"].duplicated()].unique()
    if len(dup):
        errors.append(f"duplicate point_id: {list(dup)[:10]}")
    coords = df[["latitude", "longitude"]].astype(str).agg(",".join, axis=1)
    if coords.duplicated().any():
        errors.append(f"duplicate coordinates: {coords[coords.duplicated()].tolist()[:10]}")
    return errors
