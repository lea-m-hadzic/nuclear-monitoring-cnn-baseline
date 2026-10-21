"""Provisional facility reconstruction from the human labels in the master workbook.

The workbook has no facility IDs. Target facilities are identifiable only through the free-text
detail of the human label (e.g. "TP - silo", "FN - warhead storage - cut off", "TP - Natanz").
This module reconstructs facilities as follows (documented in README):

* Natanz / Fordow: every row whose label names the site ("Natanz" / "Fordow") belongs to that
  site's single facility. Membership is therefore name-based, not inferred.
* Saratov: rows naming a target *type* (silo, warhead storage, transfer point, waste repository)
  are grouped per (zoom, type) by single-linkage clustering of row centres with a 2.5 km link.
  This reproduces manuscript Tables 2 and 3 exactly, but it is a reconstruction.
* z17 clusters are matched to z16 clusters (same type, greedy nearest centroid <= 2.5 km) so that
  one facility_id spans both zoom levels.

Every judgement that is not explicit in the source files is marked needs_author_confirmation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (STAGE_QUADRANT, STAGE_REPLACED, STAGE_ZOOMOUT, ensure_output_dir, km_xy,
                    load_config, load_processed)

# (facility_type, regex on lower-cased label detail, id prefix, known nuclear target per manuscript)
TYPE_RULES = [
    ("icbm_silo", r"silo", "SILO", True),
    ("warhead_storage", r"warhead storage", "WHS", True),
    ("warhead_rail_to_road_transfer", r"transfer", "XFER", True),
    ("radioactive_waste_repository", r"radioactive waste|nuclear waste", "RWR", False),
    ("enrichment_plant", r"\bnatanz\b|\bfordow\b", "ENR", True),
]
REGION_CODE = {"Saratov": "SAR", "Natanz": "NAT", "Fordow": "FOR"}
FACILITY_NAMES = {"Natanz": "Natanz Fuel Enrichment Plant", "Fordow": "Fordo Fuel Enrichment Plant"}


def classify_target_type(detail: pd.Series) -> pd.Series:
    d = detail.str.lower()
    out = pd.Series("", index=detail.index, dtype=object)
    for ftype, pattern, _, _ in TYPE_RULES:
        hit = d.str.contains(pattern, regex=True) & (out == "")
        out[hit] = ftype
    return out


def single_linkage(xy: np.ndarray, link_km: float) -> np.ndarray:
    """Connected components of the graph with edges between points closer than link_km."""
    n = len(xy)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    for i, j in zip(*np.where(np.triu(d < link_km, 1))):
        parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def _cluster_rows(rows: pd.DataFrame, link_km: float) -> pd.DataFrame:
    """Assign a provisional cluster key per (region, zoom, type)."""
    rows = rows.copy()
    rows["cluster_key"] = ""
    for (region, zoom, ftype), g in rows.groupby(["region", "zoom", "facility_type"]):
        if region in FACILITY_NAMES:              # named single facility: no clustering needed
            rows.loc[g.index, "cluster_key"] = f"{region}|z{zoom}|{ftype}|named"
            continue
        xy = km_xy(g["lat"], g["lon"], g["lat"].mean())
        lab = single_linkage(xy, link_km)
        rows.loc[g.index, "cluster_key"] = [f"{region}|z{zoom}|{ftype}|{k}" for k in lab]
    return rows


def _centroids(rows: pd.DataFrame) -> pd.DataFrame:
    return (rows.groupby("cluster_key")
            .agg(region=("region", "first"), zoom=("zoom", "first"), facility_type=("facility_type", "first"),
                 lat=("lat", "mean"), lon=("lon", "mean"), n_rows=("unit_id", "size"))
            .reset_index())


def reconstruct(config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (facilities, row_assignments).

    row_assignments: one row per workbook row that names a facility (z16 and z17), with facility_id.
    """
    cfg = config or load_config()
    link = cfg["facility_reconstruction"]["cluster_link_km"]
    match_km = cfg["facility_reconstruction"]["cross_zoom_match_km"]
    df = load_processed(cfg)
    df["facility_type"] = classify_target_type(df["label_detail"])
    rows = _cluster_rows(df[df["facility_type"] != ""], link)
    cents = _centroids(rows)

    # Assign facility IDs from z16 clusters (sorted north->south, west->east within type).
    id_of_cluster: dict[str, str] = {}
    match_dist: dict[str, float] = {}
    for (region, ftype), g in cents[cents.zoom == 16].groupby(["region", "facility_type"]):
        prefix = next(p for t, _, p, _ in TYPE_RULES if t == ftype)
        g = g.sort_values(["lat", "lon"], ascending=[False, True])
        for k, key in enumerate(g["cluster_key"], 1):
            id_of_cluster[key] = f"{REGION_CODE[region]}-{prefix}-{k:02d}"

    # Match z17 clusters to z16 facility IDs (greedy nearest, one-to-one, same type).
    z17_only = []
    for (region, ftype), g17 in cents[cents.zoom == 17].groupby(["region", "facility_type"]):
        g16 = cents[(cents.zoom == 16) & (cents.region == region) & (cents.facility_type == ftype)]
        ref = g17["lat"].mean()
        a, b = km_xy(g16.lat, g16.lon, ref), km_xy(g17.lat, g17.lon, ref)
        d = np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1)) if len(g16) else np.empty((0, len(g17)))
        used16, used17 = set(), set()
        for i, j in sorted(np.ndindex(d.shape), key=lambda ij: d[ij]):
            if d[i, j] > match_km or i in used16 or j in used17:
                continue
            used16.add(i); used17.add(j)
            k17 = g17["cluster_key"].iloc[j]
            id_of_cluster[k17] = id_of_cluster[g16["cluster_key"].iloc[i]]
            match_dist[k17] = float(d[i, j])
        for j in set(range(len(g17))) - used17:
            prefix = next(p for t, _, p, _ in TYPE_RULES if t == ftype)
            fid = f"{REGION_CODE[region]}-{prefix}-Z17ONLY-{len(z17_only) + 1:02d}"
            id_of_cluster[g17["cluster_key"].iloc[j]] = fid
            z17_only.append(fid)

    rows["facility_id"] = rows["cluster_key"].map(id_of_cluster)
    cents["facility_id"] = cents["cluster_key"].map(id_of_cluster)
    cents["z17_match_km"] = cents["cluster_key"].map(match_dist)

    facilities = _facility_table(rows, cents)
    assignments = rows[["unit_id", "region", "zoom", "record_id", "row_tile_id", "parent_tile_id", "stage",
                        "included_final", "score", "human_label", "facility_type", "facility_id"]]
    return facilities, assignments.sort_values(["region", "zoom", "facility_id", "record_id"]).reset_index(drop=True)


def _facility_table(rows: pd.DataFrame, cents: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for fid, g in rows.groupby("facility_id"):
        region, ftype = g["region"].iloc[0], g["facility_type"].iloc[0]
        known = next(k for t, _, _, k in TYPE_RULES if t == ftype)
        g16, g17 = g[g.zoom == 16], g[g.zoom == 17]
        c16 = cents[(cents.facility_id == fid) & (cents.zoom == 16)]
        c17 = cents[(cents.facility_id == fid) & (cents.zoom == 17)]
        labels = "; ".join(sorted(set(g["human_label"])))
        notes, confirm = [], False
        status = "provisional_reconstruction"
        in_denominator = known

        if region in FACILITY_NAMES:
            notes.append(f"Membership: all z16 rows whose human label names '{region}'.")
        else:
            notes.append("Membership: 2.5 km single-linkage cluster of rows naming this target type.")
        if len(g16) and not len(g17) and region == "Saratov" and ftype == "icbm_silo":
            status = "candidate_extra_silo_not_in_manuscript_count"
            in_denominator = False
            confirm = True
            notes.append("z16 workbook yields 21 silo locations vs 20 in the manuscript; this one has no z17 "
                         "silo label within 2.5 km (z17 tiles here are TN). Excluded from the primary "
                         "denominator by default; include via --include-candidate-facilities.")
        if "under renovation" in labels:
            confirm = True
            notes.append("Labelled 'silo under renovation' at z16; counted as one of the 20 silos.")
        z17_km = c17["z17_match_km"].dropna()
        if len(z17_km) and z17_km.iloc[0] > 1.0:
            confirm = True
            notes.append(f"z16/z17 cluster centroids are {z17_km.iloc[0]:.2f} km apart.")
        if "cut off" in labels:
            notes.append("Some z17 rows labelled 'cut off'.")
        if ftype == "warhead_storage":
            confirm = True
            notes.append("Manuscript names three storage sites (Saratov-63, Tatishchevo, Engels); the files do not "
                         "map names to tiles.")
            s7 = g16[(g16.score == 7) & g16.human_label.str.startswith("TP")]
            if len(s7) >= 2:
                notes.append("Two z16 TP rows scored 7 (one via SE quadrant): pattern matches the manuscript's "
                             "Saratov-63 description.")
        if ftype == "radioactive_waste_repository":
            status = "non_target_per_manuscript"
            notes.append("Manuscript explicitly excludes the civilian waste repository from surfaced/missed counts.")
        if region == "Fordow":
            confirm = True
            notes.append("Fordow rows form two groups ~3.4 km apart (r14_c15 vs r15_c15/r16_c16). The manuscript's "
                         "double fence east of the complex (score -7, Fig. 5b) is not labelled 'Fordow' in the "
                         "workbook, so the footprint may be incomplete.")
        if region == "Natanz":
            confirm = True
            notes.append("Parent r15_c17 is labelled TN while its SW quadrant is 'FN - Natanz' (label conflict).")

        z16_parents = sorted(set(g16["parent_tile_id"]), key=_tile_sort_key)
        recs.append(dict(
            facility_id=fid, region=region, facility_type=ftype,
            facility_name=FACILITY_NAMES.get(region, ""),
            known_target=known, in_primary_denominator=in_denominator, status=status,
            needs_author_confirmation=confirm,
            z16_parent_tiles=";".join(z16_parents), n_z16_parent_tiles=len(z16_parents),
            z16_label_rows=len(g16), z17_label_rows=len(g17),
            z17_tiles=";".join(sorted(set(g17["row_tile_id"]))),
            centroid_lat_z16=c16["lat"].iloc[0] if len(c16) else np.nan,
            centroid_lon_z16=c16["lon"].iloc[0] if len(c16) else np.nan,
            centroid_lat_z17=c17["lat"].iloc[0] if len(c17) else np.nan,
            centroid_lon_z17=c17["lon"].iloc[0] if len(c17) else np.nan,
            z16_z17_centroid_km=z17_km.iloc[0] if len(z17_km) else np.nan,
            human_labels=labels, notes=" ".join(notes)))
    return pd.DataFrame(recs).sort_values(["region", "facility_type", "facility_id"]).reset_index(drop=True)


def _tile_sort_key(t: str):
    r, c = t[1:].split("_c")
    return int(r), int(c)


def parent_membership(assignments: pd.DataFrame, zoom: int = 16) -> pd.DataFrame:
    """Facility footprint at parent-tile level: (region, tile_id, facility_id).

    A parent belongs to a facility if the parent row itself or any of its quadrant / zoom-out rows
    names the facility. Zoom-out blocks extend beyond the parent; they are tracked via via_zoomout.
    """
    a = assignments[assignments.zoom == zoom]
    out = (a.groupby(["region", "parent_tile_id", "facility_id"])
           .agg(n_label_rows=("record_id", "size"),
                source_record_ids=("record_id", lambda s: ";".join(map(str, sorted(s)))),
                via_zoomout=("stage", lambda s: bool((s == STAGE_ZOOMOUT).any())))
           .reset_index().rename(columns={"parent_tile_id": "tile_id"}))
    return out


def row_membership(df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Facility membership for VLM output rows (units of the manuscript's rule).

    Rows naming a facility belong to it. A replaced parent (relevant only for single-pass evaluation)
    inherits the facilities named by its quadrant / zoom-out rows, because its footprint contains them.
    """
    direct = assignments[["unit_id", "facility_id"]]
    rp = df[df.stage == STAGE_REPLACED][["unit_id", "region", "zoom", "parent_tile_id"]]
    kids = assignments[assignments.stage.isin([STAGE_QUADRANT, STAGE_ZOOMOUT])]
    inherited = rp.merge(kids[["region", "zoom", "parent_tile_id", "facility_id"]],
                         on=["region", "zoom", "parent_tile_id"])[["unit_id", "facility_id"]]
    return pd.concat([direct, inherited]).drop_duplicates().reset_index(drop=True)


def write_questions(facilities: pd.DataFrame, assignments: pd.DataFrame, path) -> None:
    sar = facilities[facilities.region == "Saratov"]
    extra = sar[sar.status == "candidate_extra_silo_not_in_manuscript_count"]
    reno = sar[sar.human_labels.str.contains("under renovation")]
    whs = sar[sar.facility_type == "warhead_storage"]
    nat = facilities[facilities.region == "Natanz"].iloc[0]
    fordo = facilities[facilities.region == "Fordow"].iloc[0]
    missed = _z17_missed_silos(facilities, assignments)
    no_cut = [r for r in missed.itertuples() if "cut off" not in r.human_labels]
    lines = [
        "# Facility-mapping questions for Yanliang / Qiqi",
        "",
        "Generated by `src/facility_mapping.py` from `VLM_experiment_results_master.xlsx`. Current defaults are "
        "encoded as `needs_author_confirmation` in `facilities_provisional.csv`; nothing changes until confirmed.",
        "",
        "## A. Must decide before final training / evaluation",
        "",
        "These change the recall denominator, the facility footprints used by the shared footprint rule, or "
        "training labels.",
        "",
        "1. **Authoritative facility list.** Please confirm the 24 Saratov targets (20 ICBM silos, 3 warhead-storage "
        "sites, 1 rail-to-road transfer point) plus Natanz and Fordo, ideally with coordinates or the tile IDs you "
        "counted. The current table is reconstructed by clustering label text and reproduces Tables 2-3 exactly.",
        f"2. **21st silo (`{', '.join(extra.facility_id)}`, z16 tile `{', '.join(extra.z16_parent_tiles)}`, label "
        f"`{'; '.join(extra.human_labels)}`).** Full-resolution imagery shows an unmistakable silo enclosure about "
        "40.0 km from the AOI centre, i.e. on the 40 km boundary; the z17 scan does not cover it. In or out of the "
        "denominator? (Default: out, matching the manuscript's 20.)",
        f"3. **Fordo footprint.** 'Fordow' rows fall on parents `{fordo.z16_parent_tiles}`, in two groups ~3.4 km "
        "apart (`r14_c15` vs `r15_c15`/`r16_c16`). Are both part of the facility? Which tile holds the double fence "
        "east of the complex scored -7 (Fig. 5b), and should it be in the footprint?",
        f"4. **Natanz footprint and conflicting parent labels.** 'Natanz' rows fall on parents `{nat.z16_parent_tiles}`. "
        "Parents `r15_c17`, `r3_c9`, `r7_c14`, `r8_c15` are labelled TN while one of their quadrants is TP/FN. "
        "The benchmark uses the quadrant union (positive) and excludes these 4 from training. Agree?",
        "5. **Reasoning-disqualified tiles.** When a tile scored >= 5 contained a target but the VLM reasoning pointed "
        "at the wrong feature, how was it labelled (FP with a confounder detail? FN?), and was the target name "
        "dropped? This decides whether (a) facility footprints are complete and (b) FP tiles are clean negatives.",
        "6. **Partially visible facilities.** Should a footprint include tiles where only part of the perimeter is "
        "visible (tile-edge / 'cut off' cases), and should such tiles be positive training examples?",
        "",
        "## B. Can defer to appendix / cleanup",
        "",
        "7. **Warhead-storage names.** Which reconstructed cluster is Saratov-63, Tatishchevo, and Engels? "
        + "; ".join(f"`{r.facility_id}` = z16 {r.z16_parent_tiles}" for r in whs.itertuples())
        + ". (The Tatishchevo description, two tiles via SE and NE quadrants, matches only one z16 row, `r11_c5, SE`.)",
        "8. **'Silo under renovation'.** Both are counted among the 20 silos (needed to match the manuscript): "
        + "; ".join(f"`{r.facility_id}` (z16 {r.z16_parent_tiles}, z17 counterpart {r.z16_z17_centroid_km:.2f} km "
                    "away)" for r in reno.itertuples()) + ". Please confirm.",
        "9. **Manuscript text, z17 missed silos.** The text says all six were cut off at tile edges; these have no "
        "'cut off' label: " + (", ".join(f"`{r.facility_id}` (z17 {r.z17_tiles})" for r in no_cut) or "none")
        + ". Text or label correction?",
        "10. **Workbook QA note.** 'Notes & QA' says Natanz has 89 final positives vs 86 in the manuscript; the "
        "current data give 90 (63 tiles + 27 quadrants), matching the manuscript's 251.8 km2. Stale note?",
        "11. **Fordow follow-ups.** 13 edge parents were zoomed in the VLM CSV, but their quadrants are absent from the "
        "workbook (419 vs 471 follow-ups). Dropped as out-of-bounds? Labels are parent-level, so no benchmark impact.",
        "",
    ]
    path.write_text("\n".join(lines))


def _z17_missed_silos(facilities: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    z17 = assignments[(assignments.zoom == 17) & (assignments.included_final == 1)]
    hit = set(z17.facility_id[(z17.score >= 5) & z17.human_label.str.startswith("TP")])
    sil = facilities[(facilities.facility_type == "icbm_silo") & facilities.in_primary_denominator]
    return sil[~sil.facility_id.isin(hit)]


def main() -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cfg = load_config()
    facilities, assignments = reconstruct(cfg)
    out = ensure_output_dir()
    facilities.to_csv(out / "facilities_provisional.csv", index=False)
    assignments.to_csv(out / "facility_row_assignments.csv", index=False)
    parent_membership(assignments).to_csv(out / "facility_parent_membership_z16.csv", index=False)
    write_questions(facilities, assignments, out / "facility_mapping_questions.md")

    print("Provisional facilities by region/type (z16 parents):")
    print(facilities.groupby(["region", "facility_type"]).agg(
        n=("facility_id", "size"), in_denominator=("in_primary_denominator", "sum"),
        needs_confirmation=("needs_author_confirmation", "sum"),
        parents_min=("n_z16_parent_tiles", "min"), parents_max=("n_z16_parent_tiles", "max")).to_string())
    print("\nPrimary denominator per region:",
          facilities[facilities.in_primary_denominator].groupby("region").size().to_dict())
    print("Wrote facilities_provisional.csv, facility_row_assignments.csv, facility_parent_membership_z16.csv, "
          "facility_mapping_questions.md")


if __name__ == "__main__":
    main()
