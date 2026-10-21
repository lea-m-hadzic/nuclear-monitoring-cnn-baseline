"""Build the canonical z16 parent-tile table (one row per parent tile) for Saratov, Natanz, Fordow.

Human target (`human_target`, layered-fencing signature present) is derived ONLY from the human label
prefix (TP/FN = actual positive), never from the VLM score. Aggregation to the parent:

  A. Parent not followed up ("Original output"): target = its own label is TP or FN.
  B. Parent replaced by quadrants ("Replaced parent" with quadrant rows):
       target = any quadrant label is TP or FN (quadrants tile the parent exactly).
       The parent's own label (TN or bare FN) is kept as `parent_own_label`; if it disagrees with the
       quadrant union the tile is flagged label_conflict -> excluded from training, needs confirmation.
  C. Parent replaced by zoom-out blocks only: zoom-out blocks are 1.9 km blocks offset from the parent
       (each overlaps only one parent quadrant), so they are NOT aggregated. target = parent's own
       label; flagged needs_author_confirmation.
  Zoom-out block labels are never used for the parent target; a positive block is flagged.

Usage:  python src/build_parent_table.py
"""
from __future__ import annotations

import posixpath

import numpy as np
import pandas as pd

import facility_mapping as fm
from common import (STAGE_ORIGINAL, STAGE_QUADRANT, STAGE_REPLACED, STAGE_ZOOMOUT, ensure_output_dir,
                    load_config, load_processed, load_vlm_csv)

AMBIGUOUS_PATTERNS = {"cut off": "cut_off", "former": "former_structure", "under renovation": "under_renovation"}


def aggregate_parent_targets(rows: pd.DataFrame) -> pd.DataFrame:
    """Pure function: workbook rows for ONE zoom level -> one target row per (region, parent tile).

    Uses only stage, human_label/human_positive, label_detail and record_id. The VLM score is not read.
    """
    recs = []
    for (region, tile), g in rows.groupby(["region", "parent_tile_id"], sort=False):
        par = g[g.stage.isin([STAGE_ORIGINAL, STAGE_REPLACED])]
        if len(par) != 1:
            raise ValueError(f"{region} {tile}: expected exactly one parent row, found {len(par)}")
        p = par.iloc[0]
        quads = g[g.stage == STAGE_QUADRANT]
        zouts = g[g.stage == STAGE_ZOOMOUT]
        flags, reasons = [], []
        conflict = False

        if p.stage == STAGE_ORIGINAL:
            target, source = bool(p.human_positive), "parent_label"
            contributing = par
        elif len(quads):
            target, source = bool(quads.human_positive.any()), "quadrant_union"
            contributing = quads
            if bool(p.human_positive) != target:
                conflict = True
                reasons.append("label_conflict_parent_vs_quadrants")
        else:
            target, source = bool(p.human_positive), "replaced_parent_label_zoomout_only"
            contributing = par
            flags.append("zoomout_only_followup")
            reasons.append("zoomout_only_followup")
        if len(zouts) and zouts.human_positive.any():
            flags.append("zoomout_block_positive_not_aggregated")

        details = [d for d in contributing.label_detail if d]
        ambiguous = sorted({tag for d in details for pat, tag in AMBIGUOUS_PATTERNS.items() if pat in d.lower()})
        recs.append(dict(
            region=region, tile_id=tile, row=int(p.row), col=int(p.col),
            stage=p.stage, n_quadrant_rows=len(quads), n_zoomout_rows=len(zouts),
            human_target=int(target), target_source=source,
            parent_own_label=p.human_label,
            quadrant_labels="; ".join(f"{r.row_tile_id.split(', ')[-1]}={r.human_label}" for r in quads.itertuples()),
            zoomout_labels="; ".join(zouts.human_label),
            positive_label_details="; ".join(sorted({r.human_label for r in contributing.itertuples() if r.human_positive})),
            label_conflict=conflict, ambiguous_label=bool(ambiguous), ambiguous_tags=";".join(ambiguous),
            aux_flags=flags, aux_reasons=reasons,
            parent_record_id=int(p.record_id),
            child_record_ids=";".join(map(str, sorted(pd.concat([quads, zouts]).record_id)))))
    return pd.DataFrame(recs)


def _csv_metadata(region: str, cfg: dict) -> pd.DataFrame:
    d = load_vlm_csv(region, cfg)
    rel = d.image_path.str.split("/IMINT-Images/").str[1]
    return pd.DataFrame({
        "tile_id": d.tile_id, "manifest_row": d.manifest_row, "manifest_col": d.manifest_col,
        "center_lat": d.center_lat, "center_lon": d.center_lon,
        "csv_zoom_triggered": d.zoom_triggered, "csv_zoom_count": d.zoom_count,
        "source_image_relpath": rel,
        "thumbnail_relpath": [posixpath.join(cfg["sources"]["thumbnail_dirs"][region], f"tile_{t}.jpg")
                              for t in d.tile_id],
    })


def build(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    df = load_processed(cfg)
    z = df[(df.zoom == cfg["benchmark_zoom"]) & df.region.isin(cfg["regions"])]

    table = aggregate_parent_targets(z)

    # VLM scores: attached for evaluation only (never used for the target).
    par_rows = z[z.stage.isin([STAGE_ORIGINAL, STAGE_REPLACED])]
    final = z[z.included_final == 1]
    vlm = (par_rows[["region", "parent_tile_id", "score", "lat", "lon"]]
           .rename(columns={"parent_tile_id": "tile_id", "score": "vlm_score_single_pass",
                            "lat": "wb_lat", "lon": "wb_lon"}))
    vlm_max = (final.groupby(["region", "parent_tile_id"]).score.max()
               .rename("vlm_score_adaptive_max").reset_index().rename(columns={"parent_tile_id": "tile_id"}))
    table = table.merge(vlm, on=["region", "tile_id"]).merge(vlm_max, on=["region", "tile_id"])

    meta = pd.concat([_csv_metadata(r, cfg).assign(region=r) for r in cfg["regions"]])
    table = table.merge(meta, on=["region", "tile_id"], how="left", validate="one_to_one")
    table["tile_km"] = cfg["geometry"]["z16_parent_tile_km"]
    table["area_km2"] = cfg["geometry"]["z16_parent_area_km2"]

    # Facilities (provisional reconstruction).
    facilities, assignments = fm.reconstruct(cfg)
    mem = fm.parent_membership(assignments, zoom=cfg["benchmark_zoom"])
    mem = mem.merge(facilities[["facility_id", "facility_type", "known_target", "in_primary_denominator",
                                "needs_author_confirmation", "status"]], on="facility_id")
    fac = (mem.groupby(["region", "tile_id"])
           .agg(facility_ids=("facility_id", ";".join), facility_types=("facility_type", ";".join),
                nuclear_target=("known_target", "any"),
                facility_in_primary_denominator=("in_primary_denominator", "any"),
                facility_needs_confirmation=("needs_author_confirmation", "any"),
                facility_status=("status", lambda s: ";".join(sorted(set(s)))))
           .reset_index())
    table = table.merge(fac, on=["region", "tile_id"], how="left")
    for c in ["facility_ids", "facility_types", "facility_status"]:
        table[c] = table[c].fillna("")
    for c in ["nuclear_target", "facility_in_primary_denominator", "facility_needs_confirmation"]:
        table[c] = table[c].astype("boolean").fillna(False).astype(bool)

    # Flags.
    flags, reasons, confirm, exclude = [], [], [], []
    for r in table.itertuples():
        f, why = list(r.aux_flags), list(r.aux_reasons)
        if r.csv_zoom_triggered and r.stage == STAGE_ORIGINAL:
            f.append("csv_followup_dropped_in_workbook")   # Fordow edge tiles; label is parent-level
        if r.facility_needs_confirmation:
            why.append("facility_mapping_provisional")
        if "candidate_extra_silo_not_in_manuscript_count" in r.facility_status:
            why.append("facility_mapping_discrepancy")
        excl = []
        if r.label_conflict:
            excl.append("label_conflict_parent_vs_quadrants")
        flags.append(";".join(f)); reasons.append(";".join(dict.fromkeys(why)))
        confirm.append(bool(why)); exclude.append(";".join(excl))
    table["data_flags"] = flags
    table["needs_author_confirmation"] = confirm
    table["confirmation_reasons"] = reasons
    table["exclude_from_training"] = [bool(e) for e in exclude]
    table["exclusion_reason"] = exclude
    table["training_target"] = np.where(table.exclude_from_training, np.nan, table.human_target)

    cols = ["region", "tile_id", "row", "col", "manifest_row", "manifest_col", "center_lat", "center_lon",
            "wb_lat", "wb_lon", "tile_km", "area_km2",
            "human_target", "training_target", "target_source", "stage", "n_quadrant_rows", "n_zoomout_rows",
            "parent_own_label", "quadrant_labels", "zoomout_labels", "positive_label_details",
            "label_conflict", "ambiguous_label", "ambiguous_tags",
            "exclude_from_training", "exclusion_reason", "needs_author_confirmation", "confirmation_reasons",
            "data_flags", "facility_ids", "facility_types", "nuclear_target", "facility_in_primary_denominator",
            "facility_status", "vlm_score_single_pass", "vlm_score_adaptive_max",
            "csv_zoom_triggered", "csv_zoom_count", "parent_record_id", "child_record_ids",
            "source_image_relpath", "thumbnail_relpath"]
    order = {r: i for i, r in enumerate(cfg["regions"])}
    table = table.sort_values(["region", "row", "col"], key=lambda s: s.map(order) if s.name == "region" else s)
    return table[cols].reset_index(drop=True)


def summarize(t: pd.DataFrame) -> None:
    print("Parent tiles per region:", t.groupby("region").size().to_dict())
    s = t.groupby("region").agg(
        positive=("human_target", "sum"),
        negative=("human_target", lambda x: int((x == 0).sum())),
        excluded=("exclude_from_training", "sum"),
        ambiguous=("ambiguous_label", "sum"),
        needs_confirmation=("needs_author_confirmation", "sum"),
        target_from_quadrants=("target_source", lambda x: int((x == "quadrant_union").sum())),
        nuclear_target_tiles=("nuclear_target", "sum"))
    print(s.to_string())
    print("\nMissing values in key fields:",
          {c: int(t[c].isna().sum()) for c in ["manifest_row", "center_lat", "source_image_relpath",
                                                 "vlm_score_single_pass", "human_target"] if t[c].isna().any()} or "none")
    print("Row/col vs CSV manifest mismatches:",
          int(((t.row != t.manifest_row) | (t.col != t.manifest_col)).sum()))
    print("Max |workbook - CSV| centre offset (deg):",
          float(np.nanmax(np.abs(np.r_[t.wb_lat - t.center_lat, t.wb_lon - t.center_lon]))))
    print("\nInconsistencies / flags:")
    for _, r in t[t.label_conflict].iterrows():
        print(f"  label conflict  {r.region} {r.tile_id}: parent={r.parent_own_label!r} quadrants: {r.quadrant_labels}")
    for _, r in t[t.data_flags.str.contains("zoomout")].iterrows():
        print(f"  zoom-out        {r.region} {r.tile_id}: {r.data_flags} parent={r.parent_own_label!r} blocks: {r.zoomout_labels}")
    n_drop = t.data_flags.str.contains("csv_followup_dropped").groupby(t.region).sum()
    print("  CSV follow-ups not in workbook (labels are parent-level):", n_drop[n_drop > 0].to_dict())
    print("  Ambiguous tags:", t[t.ambiguous_label].groupby(["region", "ambiguous_tags"]).size().to_dict())


def main() -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    t = build()
    out = ensure_output_dir()
    t.to_csv(out / "parent_tiles.csv", index=False)
    summarize(t)
    print(f"\nWrote {out / 'parent_tiles.csv'} ({len(t)} rows)")


if __name__ == "__main__":
    main()
