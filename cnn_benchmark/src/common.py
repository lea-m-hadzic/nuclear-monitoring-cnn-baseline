"""Shared paths, config loading, and read-only access to the source workbook / VLM CSVs."""
from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[1]          # cnn_benchmark/
REPO_ROOT = PACKAGE_DIR.parent                              # project root holding the source data
OUTPUT_DIR = PACKAGE_DIR / "outputs"
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "default.yaml"

POSITIVE_PREFIXES = ("TP", "FN")   # human "actual positive": layered-fencing signature present
LABEL_RE = re.compile(r"^(TP|FP|TN|FN)\b")
PARENT_RE = re.compile(r"^(r(\d+)_c(\d+))")

STAGE_ORIGINAL = "Original output"
STAGE_REPLACED = "Replaced parent"
STAGE_QUADRANT = "Quadrant output"
STAGE_ZOOMOUT = "Zoom-out mosaic block"


@functools.lru_cache(maxsize=None)
def load_config(path: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def source_path(rel: str) -> Path:
    return REPO_ROOT / rel


@functools.lru_cache(maxsize=None)
def _load_processed_cached(workbook: str) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name="Processed Data")
    df = df.rename(columns={
        "Case": "region", "Zoom": "zoom", "Record ID": "record_id", "Tile ID": "row_tile_id",
        "Latitude": "lat", "Longitude": "lon", "Score": "score", "Parent/Child": "parent_child",
        "Human label": "human_label", "Stage": "stage", "Included in final?": "included_final",
        "Outcome": "outcome", "Actual class": "actual_class", "Predicted class": "predicted_class",
        "Cut-off flag": "cutoff_flag", "Label detail": "label_detail", "Tile width (km)": "width_km",
        "Output area (km²)": "output_area_km2", "Review area (km²)": "review_area_km2",
    })
    df["region"] = df["region"].astype(str)
    df["human_label"] = df["human_label"].astype(str).str.strip()
    df["label_detail"] = df["label_detail"].fillna("").astype(str)
    df["label_prefix"] = df["human_label"].str.extract(LABEL_RE)[0]
    df["human_positive"] = df["label_prefix"].isin(POSITIVE_PREFIXES)
    m = df["row_tile_id"].str.extract(PARENT_RE)
    df["parent_tile_id"] = m[0]
    df["row"] = m[1].astype(int)
    df["col"] = m[2].astype(int)
    df["unit_id"] = df["region"] + "|z" + df["zoom"].astype(str) + "|" + df["record_id"].astype(str)
    return df


def load_processed(config: dict | None = None) -> pd.DataFrame:
    """The workbook's 'Processed Data' sheet with normalized column names (a fresh copy each call)."""
    cfg = config or load_config()
    return _load_processed_cached(str(source_path(cfg["sources"]["workbook"]))).copy()


@functools.lru_cache(maxsize=None)
def _load_raw_cached(workbook: str) -> pd.DataFrame:
    sheets = {("Saratov", 16): "Raw Saratov z16", ("Saratov", 17): "Raw Saratov z17",
              ("Natanz", 16): "Raw Natanz", ("Fordow", 16): "Raw Fordow"}
    parts = []
    for (region, zoom), sheet in sheets.items():
        d = pd.read_excel(workbook, sheet_name=sheet)
        d["region"], d["zoom"] = region, zoom
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def load_raw_sheets(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    return _load_raw_cached(str(source_path(cfg["sources"]["workbook"]))).copy()


@functools.lru_cache(maxsize=None)
def _load_vlm_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_vlm_csv(region: str, config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    return _load_vlm_csv_cached(str(source_path(cfg["sources"]["vlm_csvs"][region]))).copy()


def load_vlm_csv_z17(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    return _load_vlm_csv_cached(str(source_path(cfg["sources"]["vlm_csv_z17"]))).copy()


def km_xy(lat, lon, ref_lat: float) -> np.ndarray:
    """Local equirectangular projection in km (adequate at <100 km scales)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return np.c_[lat * 111.32, lon * 111.32 * np.cos(np.radians(ref_lat))]


def ensure_output_dir(sub: str | None = None) -> Path:
    d = OUTPUT_DIR / sub if sub else OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d
