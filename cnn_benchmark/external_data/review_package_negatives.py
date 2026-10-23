"""Targeted negative-review package for Yanliang (Google Earth / KML). Reads metadata only; changes no label.

    python review_package_negatives.py
    -> external_imagery/dataset/review_packages/negative_review_targeted/ (git-ignored; contains coordinates)
       negatives_for_review.kml, negatives_for_review.csv, README.md

Scope: the 382 distinct negatives used by the frozen ablation manifests (A/B/C/C_matched: 225 nearby + 157 confirmed
hard). Selection uses existing metadata / QA heuristics ONLY (no model scores, no temperatures), fixed before looking
at which tiles they pick. A tile is selected if any rule fires; all firing rules are listed:
  P  proximity   nearby negative in the closest 10 % (by distance from its parent positive's fence point) of the
                 225 training nearby negatives: an outlying part of the parent site's perimeter could reach the tile
  X  complex     nearby negative whose parent belongs to a multi-point complex (split group with >= 3 positive
                 points) and lies closer than the median parent-fence distance: large complexes have extended perimeters
  B  built-up    nearby negative in the top 5 % edge density of the 225: industrial / built-up scene where a secured
                 auxiliary compound could be present
  H  pre-flag    confirmed hard negative flagged at pre-extraction screening or pre-flagged for review earlier
Nearby negatives were never manually screened for fence signatures (hard negatives were, under the
reject_target_signature rule), so the rules concentrate on them.
KML: one folder per rule colour (P red, X magenta, B orange, H purple; multiple rules -> first in that order), a pin
at each tile centre and the tile footprint square; parent positive points (yellow) for P / X tiles for context.
"""
from __future__ import annotations

import glob
import math
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parent
D = REPO / "external_imagery" / "dataset"
OUT = D / "review_packages" / "negative_review_targeted"
MAN = PKG / "outputs" / "ablation" / "manifests"
PREFLAG_NAMES = ("Hebei Shenghua", "Angus")   # training hard negatives pre-flagged for review earlier
RULES = {"P": ("proximity", "ff0000ff"), "X": ("complex", "ffff00ff"), "B": ("built-up", "ff0080ff"),
         "H": ("pre-flag", "ffb03c8c")}          # KML colours are aabbggrr
TEXT = {"P": "closest 10% of nearby negatives to the parent site's fence point ({:.2f} km): an outlying part of "
             "the parent perimeter could reach the tile",
        "X": "parent is part of a multi-point complex ({} positive points) and the tile is closer than the median "
             "({:.2f} km): extended perimeter plausible",
        "B": "top 5% edge density among nearby negatives ({:.3f}): built-up / industrial scene where a secured "
             "auxiliary compound could be present",
        "H": "confirmed hard negative flagged at pre-extraction screening / earlier review ({}): re-check for a "
             "layered perimeter"}


def square(lat, lon, km):
    """Approximate tile footprint (axis-aligned square of side `km`) as [lon, lat] corners."""
    dlat = km / 2 / 111.32
    dlon = km / 2 / (111.32 * math.cos(math.radians(lat)))
    return [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat), (lon + dlon, lat + dlat), (lon - dlon, lat + dlat),
            (lon - dlon, lat - dlat)]


def main():
    man = pd.concat([pd.read_csv(MAN / f"{c}.csv").assign(cond=c) for c in ("A", "B", "C", "C_matched")])
    neg = man[man.label == 0].groupby("tile_id").agg(role=("role", "first"), split=("split", "first"),
                                                     conditions=("cond", lambda x: ",".join(sorted(set(x))))).reset_index()
    meta = pd.read_csv(D / "external_tiles_metadata_refined.csv")
    pos = meta[meta.role == "positive"]
    n_in_group = pos.groupby("split_group_id").size()
    nb = neg[neg.role == "nearby"].merge(meta.drop(columns=["role", "split"]), on="tile_id")
    nb["n_pos_in_group"] = nb.split_group_id.map(n_in_group).fillna(0).astype(int)
    p10 = nb.neg_dist_from_parent_fence_km.quantile(0.10)
    med = nb.neg_dist_from_parent_fence_km.median()
    e95 = nb.edge_density.quantile(0.95)
    hn = pd.concat([pd.read_csv(f) for f in glob.glob(str(D / "hard_negative_metadata_private_*.csv"))]).drop_duplicates("tile_id")
    rv = pd.read_csv(D / "hard_negative_review.csv").set_index("tile_id")
    hd = neg[neg.role == "hard"].merge(hn.drop(columns=["role", "split"], errors="ignore"), on="tile_id")
    rows = []
    for r in nb.itertuples():
        rules = []
        if r.neg_dist_from_parent_fence_km <= p10:
            rules.append(("P", TEXT["P"].format(r.neg_dist_from_parent_fence_km)))
        if r.n_pos_in_group >= 3 and r.neg_dist_from_parent_fence_km < med:
            rules.append(("X", TEXT["X"].format(r.n_pos_in_group, r.neg_dist_from_parent_fence_km)))
        if r.edge_density >= e95:
            rules.append(("B", TEXT["B"].format(r.edge_density)))
        if rules:
            parent = pos[pos.tile_id == r.parent_point_id]
            plat, plon = (parent.latitude.iloc[0], parent.longitude.iloc[0]) if len(parent) else ("", "")
            rows.append({"tile_id": r.tile_id, "role": "nearby", "split": r.split, "conditions": r.conditions,
                         "site_or_parent": r.facility_name_original, "center_lat": r.center_lat, "center_lon": r.center_lon,
                         "footprint_km": r.footprint_width_km, "rules": "+".join(k for k, _ in rules),
                         "why_selected": " | ".join(t for _, t in rules),
                         "dist_parent_fence_km": round(r.neg_dist_from_parent_fence_km, 2), "edge_density": round(r.edge_density, 3),
                         "parent_lat": plat, "parent_lon": plon})
    for r in hd.itertuples():
        screen = rv.pre_extraction_screen.get(r.tile_id)
        name = str(r.facility_name_original)
        hit = [n for n in PREFLAG_NAMES if n in name]
        if screen == "flag" or hit:
            why = "pre-extraction screen = flag" if screen == "flag" else f"pre-flagged earlier ({hit[0]})"
            rows.append({"tile_id": r.tile_id, "role": "hard", "split": r.split, "conditions": r.conditions,
                         "site_or_parent": f"{name} [{r.category}]", "center_lat": r.center_lat, "center_lon": r.center_lon,
                         "footprint_km": r.footprint_width_km, "rules": "H", "why_selected": TEXT["H"].format(why),
                         "dist_parent_fence_km": "", "edge_density": round(r.edge_density, 3), "parent_lat": "", "parent_lon": ""})
    out = pd.DataFrame(rows).sort_values(["rules", "tile_id"]).reset_index(drop=True)
    out.insert(0, "row_id", range(1, len(out) + 1))
    for c in ("review_decision", "reviewer_notes"):
        out[c] = ""
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "negatives_for_review.csv", index=False)
    write_kml(out)
    (OUT / "README.md").write_text(README.format(n=len(out), nn=int((out.role == "nearby").sum()),
                                                 nh=int((out.role == "hard").sum()), p10=p10, med=med, e95=e95))
    print(out.rules.value_counts().to_dict(), "| total", len(out), "of", len(neg), "negatives |",
          f"P<= {p10:.2f} km, X<{med:.2f} km & >=3 pts, B>= {e95:.3f}")


def write_kml(t: pd.DataFrame) -> None:
    styles = "".join(f'<Style id="{k}"><IconStyle><color>{c}</color><scale>1.1</scale><Icon><href>'
                     f'http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href></Icon></IconStyle>'
                     f'<LineStyle><color>{c}</color><width>2</width></LineStyle><PolyStyle><color>{"33" + c[2:]}</color>'
                     f'</PolyStyle></Style>' for k, (_, c) in RULES.items())
    styles += ('<Style id="parent"><IconStyle><color>ff00ffff</color><scale>0.8</scale><Icon><href>'
               'http://maps.google.com/mapfiles/kml/shapes/star.png</href></Icon></IconStyle></Style>')
    folders = []
    for k, (label, _) in RULES.items():
        sub = t[t.rules.str.split("+").str[0] == k]
        pms = []
        for r in sub.itertuples():
            ring = " ".join(f"{x:.6f},{y:.6f},0" for x, y in square(r.center_lat, r.center_lon, float(r.footprint_km)))
            desc = escape(f"row {r.row_id} | {r.role} | split {r.split} | used in {r.conditions} | rules {r.rules}\n{r.why_selected}")
            pms.append(f'<Placemark><name>#{r.row_id} {escape(str(r.site_or_parent))[:60]}</name><description>{desc}</description>'
                       f'<styleUrl>#{k}</styleUrl><MultiGeometry><Point><coordinates>{r.center_lon:.6f},{r.center_lat:.6f},0'
                       f'</coordinates></Point><Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates>'
                       f'</LinearRing></outerBoundaryIs></Polygon></MultiGeometry></Placemark>')
            if r.parent_lat != "" and pd.notna(r.parent_lat):
                pms.append(f'<Placemark><name>parent of #{r.row_id}</name><styleUrl>#parent</styleUrl><Point><coordinates>'
                           f'{float(r.parent_lon):.6f},{float(r.parent_lat):.6f},0</coordinates></Point></Placemark>')
        folders.append(f"<Folder><name>{k}: {label} ({len(sub)})</name>{''.join(pms)}</Folder>")
    kml = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f'<name>Negatives for review ({len(t)})</name>{styles}{"".join(folders)}</Document></kml>')
    (OUT / "negatives_for_review.kml").write_text(kml)


README = """# Targeted negative review ({n} tiles)

This is a review package, not a relabelling. **No negative label has been changed.**

It covers {n} of the 382 negatives used in the frozen CNN ablation: {nn} nearby and {nh} confirmed hard negatives. They were chosen with existing metadata and QA heuristics only (no model scores), because each could plausibly contain part of a layered fence.

**Files:**
- `negatives_for_review.kml`: open in Google Earth.
  - One folder per selection rule.
  - Each tile has a pin at its centre and an outline of its approximately 1.9 km footprint.
  - Yellow stars mark the parent positive site, for tiles selected because they are near it.
- `negatives_for_review.csv`: one row per tile.
  - `rules` and `why_selected` explain the selection.
  - `review_decision` and `reviewer_notes` are left blank for you.

**Selection rules** (a tile is listed if any rule applies):

| Rule | Colour | Meaning |
|---|---|---|
| P | red | Nearby tile among the closest 10% to its parent positive's fence point (≤ {p10:.2f} km). Part of the parent site's perimeter could reach into the tile. |
| X | magenta | Nearby tile whose parent site is a multi-point complex (≥ 3 positive points) and that is closer than the median ({med:.2f} km). Large complexes have extended perimeters. |
| B | orange | Nearby tile in the top 5% for edge density (≥ {e95:.3f}). A built-up or industrial scene where a secured compound might be present. |
| H | purple | Confirmed hard negative that was flagged at extraction screening or earlier review. Worth checking for a layered perimeter. |

Nearby negatives were never manually checked for fence signatures. Hard negatives were checked, under a rule that rejected any visible target signature. That's why most rows are nearby tiles.

**Please fill in `review_decision`** with one of:
- `keep_negative`
- `contains_layered_fence` (would need relabelling or exclusion)
- `unclear`

Add anything else in `reviewer_notes`. Labels change only after your answers come back and are approved.
"""


if __name__ == "__main__":
    main()
