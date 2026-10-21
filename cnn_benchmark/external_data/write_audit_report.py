"""Write cnn_benchmark/outputs/external_data/external_dataset_audit.md from the plan + audit summaries.

Aggregate numbers, facility names and IDs only; no coordinates (safe to track in git).
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from dataset_audit import REPORT_DIR
from geometry import REPO_ROOT, load_config, tiered_cost_usd

VALIDATION_REQUESTS = 39
GPPD_NOTE = "WRI Global Power Plant Database v1.3.0 (CC BY 4.0; Coal/Gas/Oil, >= 500 MW)"   # 2026-10-13 reproduction checks (5 + 18 + 5 + 11), same billing month


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def q(s: pd.Series, qs=(0, .25, .5, .75, 1), nd=1) -> str:
    return " / ".join(f"{v:.{nd}f}" for v in s.quantile(list(qs)))


def refinement_section(cfg: dict, ddir) -> list[str]:
    path = ddir / "refinement_summary.json"
    if not path.exists():
        return []
    r = json.loads(path.read_text())
    ar, sel, rv, lk = r["auto_reject"], r["selection"], r["review"], r["leakage"]
    ds, da = r["shortcut_selected_pool"], r["shortcut_all_candidates"]
    hn = pd.read_csv(REPORT_DIR / "external_hard_negatives_plan.csv")
    hcfg = cfg["hard_negatives"]
    feat = pd.DataFrame([{"feature": f, "AUC all candidates": da["per_feature"][f]["auc"],
                          "AUC selected pool": v["auc"], "pos median": v["pos_median"], "neg median": v["neg_median"]}
                         for f, v in ds["per_feature"].items()])
    def clf(d, name):
        c = d["classifier"][name]
        return (f"val AUC **{c['val_auc']}**, accuracy {c['val_accuracy']} (majority-class {d['classifier']['majority_class_val_accuracy']}), "
                f"balanced accuracy {c['val_balanced_accuracy']}; grouped 5-fold CV AUC {c['grouped_cv5_auc_mean']} +/- {c['grouped_cv5_auc_sd']}")
    lr = ds["classifier"]["logistic_regression"]
    tree = ds["classifier"]["tree_depth3"]
    curve = pd.DataFrame(ds["classifier"]["pseudo_hard_negative_curve"])
    need = ds["classifier"]["hard_negative_fraction_needed_per_feature"]
    hn_cat = hn.groupby("category")[["n_facilities", "n_tiles", "expected_requests"]].sum().reset_index()
    n_req = int(hn.expected_requests.sum())
    short = {k: v for k, v in sel["short_positives"].items()}
    one = sorted(k.replace("__pos", "") for k, v in short.items() if v == 1)
    avail = sel["positives_by_n_selected"]
    L = ["", "## 10. Negative-set refinement", "",
         "### Automatic rejection (candidate negatives only; nothing deleted)", "",
         f"Deterministic flags from pixel statistics (`negative_refinement.py`): full placeholder (placeholder colour > "
         f"{r['thresholds']['full_placeholder']:.0%} of pixels), partial placeholder (> {r['thresholds']['partial_placeholder']:.0%}), "
         f"black no-data (> {r['thresholds']['black_nodata']:.0%}), mostly water (calm or rendered water > "
         f"{r['thresholds']['mostly_water']:.0%}), corrupt (decode failure). Columns `auto_reject`, `auto_reject_reason` "
         "(multiple reasons allowed); `manual_review_status` is separate.", "",
         f"- **{ar['rejected']} of {ar['candidates']} candidates auto-rejected** ({ar['by_subset']}); reasons (multi-count): "
         f"{ar['by_reason']}; corrupt files: {ar['corrupt_any_role']}.",
         "- The water rule was calibrated on contact-sheet examples: a first version also caught dark, hazy forest; "
         "tightening the local-smoothness limit fixed it (now no positive is flagged as mostly water). Textured "
         "shallow water and dark cooling ponds may slip through to manual review.",
         f"- Positives carrying any artifact flag: {len(ar['positives_with_artifact_flags'])}.", "",
         "### Nearby-negative selection", "",
         f"Per positive: primary neg1-neg3 in order, auto-rejected removed, reserve neg4 fills gaps, max 3 "
         f"(`nearby_negative_selected`, `selection_rank`). **{sel['selected']} selected** ({sel['from_primary']} primary + "
         f"{sel['from_reserve']} reserve; train {sel['by_split'].get('train', 0)}, val {sel['by_split'].get('val', 0)}).", "",
         "| usable nearby negatives | 0 | 1 | 2 | 3 |", "|---|---|---|---|---|",
         "| positives | " + " | ".join(str(avail.get(str(k), avail.get(k, 0))) for k in range(4)) + " |", "",
         f"Shortages stay explicit: {sum(1 for v in short.values() if v < 3)} positives have < 3 (every candidate was "
         f"placed, so all shortages come from water/placeholder/no-data rejections, i.e. coastal sites); with only 1: "
         f"{', '.join(one)}.", "",
         "### Review artifacts (all statuses `unreviewed`)", "",
         f"- `review/review_sheet_NNN.png`: {rv['rapid_sheets']} sheets x 16 selected negatives, large cell number + tile ID, "
         "facility and split group (no coordinates). `candidate_negative_review.csv`: all 1,148 candidates with "
         "`manual_review_status` = unreviewed | confirmed_negative | reject_visible_layered_fence | reject_artifact | "
         "reject_ambiguous, plus sheet/cell. `review/index.html`: static local page (no JavaScript, no web "
         "dependency) with every sheet; each tile ID links to the full 2560 px PNG for zooming.",
         "- `positive_review_compact.csv`: positives sorted by split group / facility, pointing to the existing "
         f"positive sheets. Positive review status: {rv['positive_review_status']} - **no positive review has been "
         "performed**, so no positive is summarized as clean/ambiguous/mismatched, and no label was changed.",
         f"- Training-eligible tiles: {r['training_eligible']} (requires `fence_visible` for positives and "
         "`confirmed_negative` + not auto-rejected for negatives).", "",
         "### Shortcut diagnostics (nuisance/scene statistics only, no pixels or learned features)", "",
         md_table(feat), "",
         f"Classifier features: {', '.join(ds['features_used_by_classifier'])} (artifact flags excluded: they are "
         "rejection criteria). Train on train split groups, score on val split groups.", "",
         f"- Logistic regression, positives + selected nearby negatives: {clf(ds, 'logistic_regression')}. "
         f"Largest standardized coefficients: {dict(list(lr['std_coefficients'].items())[:4])}.",
         f"- Depth-3 tree: {clf(ds, 'tree_depth3')}; importance {tree['feature_importance']}.",
         f"- Before cleaning (all 1,148 candidates): logistic regression {clf(da, 'logistic_regression')}.",
         "- Reading: removing placeholders/water made the remaining negatives *more* homogeneous rural land, so "
         "nuisance-only separability went **up** (LR val AUC "
         f"{da['classifier']['logistic_regression']['val_auc']} -> {lr['val_auc']}). The dominant cue is scene "
         "contrast/edge density/texture, i.e. built-up vs. not built-up. Watermark phase, grid and zoom carry no "
         "signal (AUC ~0.50). `placeholder_frac` > 0 in clean positives is light-grey concrete matching the "
         "placeholder colour, not an artifact.",
         "- Note for the benchmark: evaluation targets include small fenced sites in open fields (Saratov silos), so a "
         "'built-up = positive' shortcut would hurt exactly there.", "",
         "How many positive-like negatives would be needed (optimistic: hard negatives assumed to match positives on "
         "every nuisance statistic; real ones will match less well):", "",
         md_table(curve.rename(columns={"hard_negative_fraction_of_negatives": "share of negatives positive-like",
                                        "added_per_split_total": "tiles added", "val_auc": "nuisance-only val AUC"})), "",
         "Single-feature analytic requirement (AUC' = (1-w) AUC + w/2): "
         + "; ".join(f"{f}: w = {v['to_auc_0.65']:.0%} for AUC 0.65, {v['to_auc_0.60']:.0%} for 0.60"
                     for f, v in need.items() if v['to_auc_0.65'] > 0) + ".", "",
         "### Proposed hard-negative strategy (not extracted; needs approval)", "",
         "No reliable industrial coordinate source exists in this repo or Qiqi's (`random_coordinates.py` samples "
         "city bounding boxes; `generate_russian_negatives.py` samples around warhead-storage cities; both are "
         "'easy random' negatives).", "",
         md_table(hn_cat), "",
         f"- Sources: {', '.join(sorted({v['source'].split(';')[0].split(' (')[0] for v in hcfg['categories'].values()}))} "
         "(public, re-downloadable, citable). Each selected point is checked by eye on its tile before use.",
         f"- Pilot: **{hn.n_facilities.sum()} facilities x {hcfg['tiles_per_facility']} tile** = {hn.n_tiles.sum()} tiles, "
         f"~{n_req} requests, ~${tiered_cost_usd(n_req, cfg['pricing'], already_used_this_month=21298):.2f} "
         "(free tier already used this month). Region quotas mirror the positives' broad regions; per-category "
         "and per-region allocation in `external_hard_negatives_plan.csv`.",
         f"- Exclusions: >= {hcfg['min_km_from_any_positive']} km from every positive (never shares a split group with "
         "a positive), outside evaluation AOIs + buffer, nuclear-like names dropped, and any candidate within "
         f"{hcfg['exclude_tags_within_km']} km of {', '.join(hcfg['exclude_tags'])} dropped (prisons and military sites "
         "carry the target signature).",
         "- Labels: `candidate_hard_negative` until manual review; tiles showing a recognizable layered fence are "
         "rejected (`reject_visible_layered_fence`), not relabelled.",
         "- Split: one split group per site; train/val assigned per (category, region) stratum with the same seeded "
         "procedure (val 0.2), so every category appears in both splits.",
         "- Shortcut control: region/latitude quotas (same zoom/grid/watermark mix), same jitter rule (fence-free "
         "facility point at 100-350 m from the centre), and the nuisance diagnostics above re-run after the pilot "
         "with category as an extra check (no category may be near-perfectly separable from positives).",
         "- Scale-up decision after the pilot: the optimistic curve says ~30-50% of negatives must be positive-like "
         "to bring nuisance-only AUC to ~0.65-0.70, i.e. roughly 350-800 hard-negative tiles "
         f"(~${tiered_cost_usd(350 * 15, cfg['pricing'], already_used_this_month=21298 + n_req):.0f}-"
         f"${tiered_cost_usd(800 * 15, cfg['pricing'], already_used_this_month=21298 + n_req):.0f} at ~15 requests/tile).", "",
         "### Leakage checks (positives + selected nearby negatives)", "",
         "| check | result |", "|---|---|",
         f"| split groups in both splits | {lk['split_groups_in_both_splits']} |",
         f"| positive/negative footprint overlap | {lk['overlapping_positive_negative_pairs']} (all {lk['overlapping_pairs_total']} "
         f"overlaps are positive-positive within one split group; across splits {lk['overlapping_pairs_across_splits']}) |",
         f"| negative footprint containing a listed fence point | {lk['negative_footprint_contains_a_fence_point']} "
         f"(closest negative centre {lk['min_negative_centre_to_any_fence_point_km']} km from a fence point) |",
         f"| near-duplicate images in the selected pool / across splits | {lk['near_duplicate_pairs_selected']} / "
         f"{lk['near_duplicate_pairs_across_splits']} |",
         f"| tiles with conflicting labels | {lk['tiles_with_conflicting_labels']} (negatives carry their parent "
         "positive's facility_id as provenance only; labels are per tile) |",
         "| hard-negative plan points inside AOIs / sharing a split group with a positive | none exist yet; enforced by "
         "`hard_negatives.py --validate` (>= 20 km from positives, AOI + buffer) |", "",
         "### Manifests", "",
         "Coordinate-free (in `outputs/external_data/`): `external_nearby_candidates.csv`, `external_nearby_selected.csv`, "
         "`external_hard_negatives_plan.csv`, `external_training_manifest_draft.csv` (positives + selected nearby "
         "negatives, `training_eligible = false` for all until review). Coordinate-bearing `*_private.csv` copies are "
         "in the git-ignored dataset directory.", "",
         "### Decisions requiring Lea's approval", "",
         "1. Auto-rejection thresholds and the water rule (154 candidates removed from selection; imagery kept).",
         "2. Accept the shortages (9 positives with 1 nearby negative, 44 with 2) or fill them (more candidates per "
         "positive in a wider annulus, or rely on hard negatives).",
         "3. Run the manual review: 50 negative sheets (799 tiles) and 24 positive sheets (287 tiles).",
         "4. Hard-negative pilot: approve sources (WRI GPPD download + OpenStreetMap/Overpass queries), categories, "
         f"80 facilities x 1 tile (~${tiered_cost_usd(n_req, cfg['pricing'], already_used_this_month=21298):.2f}), "
         "and whether layered-fenced non-nuclear sites are rejected or treated as positives.",
         "5. Scale-up target for hard negatives after the pilot diagnostics (see curve).", ""]
    return L


def pilot_section(cfg: dict, ddir) -> list[str]:
    path = ddir / "hard_negative_pilot_summary.json"
    if not path.exists():
        return []
    s = json.loads(path.read_text())
    chk = json.loads((ddir / "hard_negative_plan_checks.json").read_text())
    pts = pd.read_csv(ddir / "hard_negative_points.csv")
    log = pd.read_csv(ddir / "hard_negative_sourcing_log.csv")
    res = log.result.str.replace(r"excluded_near:.*", "excluded_near:military/prison/nuclear", regex=True) \
        .str.replace(r"shortfall .*", "shortfall", regex=True).value_counts()
    cat = pts.groupby("category").agg(sites=("hn_id", "size"), regions=("broad_region", "nunique")).reset_index()
    cat = cat.merge(pd.DataFrame(chk["category_x_split"]).fillna(0).astype(int).reset_index()
                    .rename(columns={"index": "category"}), on="category", how="left")
    diag = pd.DataFrame(s["diagnostics"])
    show = [c for c in ("comparison", "status", "n_pos", "n_neg", "lr_val_auc", "lr_grouped_cv_auc",
                        "lr_val_balanced_accuracy", "dominant_features") if c in diag]
    med = pd.DataFrame(s["median_by_group"]).reset_index().rename(columns={"index": "feature"})
    n_req = s["requests_metered"]
    L = ["", "## 11. Hard-negative pilot (80 sites)", "",
         "### Sourcing", "",
         f"- Power plants: {GPPD_NOTE}. Other categories: OpenStreetMap via Overpass (main instance, with "
         "public mirrors as fallback; the OSM data timestamp and endpoint are recorded per site). Raw responses are "
         "cached under the git-ignored `external_imagery/sources/`.",
         f"- Exclusions applied: >= {cfg['hard_negatives']['min_km_from_any_positive']} km from every external positive, "
         "outside evaluation AOIs + buffer, no nuclear-like name/tag, no military area / military=* / prison / nuclear "
         f"plant within {cfg['hard_negatives']['exclude_tags_within_km']} km (distance to the feature's bounding box), "
         ">= 2 km from every other accepted site.",
         f"- Candidates tried and outcome: {res.to_dict()}.", "",
         md_table(cat), "",
         f"- Plan checks: min distance to a positive {chk['min_km_to_positive']:.1f} km; tiles overlapping existing tiles or "
         f"each other {chk['tiles_overlapping_existing_or_each_other']}; split groups shared with positives "
         f"{chk['split_groups_shared_with_positives']}; every category in both splits: "
         f"{all(v.get('val', 0) > 0 and v.get('train', 0) > 0 for v in pd.DataFrame(chk['category_x_split']).fillna(0).T.to_dict().values())}.",
         "",
         "### Extraction", "",
         f"- Same pipeline as the positives (auto_mosaic, 1.9 km width, 2560 px, 100-350 m deterministic jitter "
         f"seeded per site, metering, redaction, geometry checks). Extraction {s['extraction']}; validation "
         f"{s['validation']}; **{n_req} metered requests** (~${tiered_cost_usd(n_req, cfg['pricing'], already_used_this_month=21298):.2f}).",
         f"- Automatic artifact flags: {s['artifact_flags']}.", "",
         "### Review (label rule: target signature, not facility type)", "",
         "- `review/hn_review_sheet_NNN.png` (tile + ~475 m zoom on the site point, cell numbers) and "
         "`hard_negative_review.csv` with `manual_review_status` = unreviewed | confirmed_hard_negative | "
         "reject_target_signature | reject_artifact | reject_ambiguous. A site with a recognizable layered fence is "
         "`reject_target_signature` and kept for audit; it is never relabelled.",
         f"- Current status: {s['review_status']}; training-eligible hard negatives: {s['training_eligible']}.", "",
         "### Diagnostics (confirmed labels only)", "",
         md_table(diag[show]), "",
         "C uses equal numbers of nearby and confirmed hard negatives per split (nearby sampled by seeded hash); A' is "
         "the size-matched nearby-only control, so sample count alone cannot explain a difference between A' and C.", "",
         "Medians (descriptive; hard-negative column = unreviewed, artifact-free candidates):", "",
         md_table(med), "", "Figure: `hard_negative_pilot_distributions.png`.", ""]
    return L


def main():
    cfg = load_config()
    ddir = REPO_ROOT / cfg["dataset"]["dataset_dir"]
    plan = json.loads((ddir / "plan_summary.json").read_text())
    aud = json.loads((ddir / "audit_summary.json").read_text())
    pos = pd.read_csv(ddir / "positive_audit.csv")
    meta = pd.read_csv(ddir / "external_tiles_metadata.csv")
    groups = pd.read_csv(ddir / "split_groups.csv")
    d = cfg["dataset"]
    j, n = d["jitter"], d["negatives"]
    flags = pos["flags"].fillna("")
    reqs = aud["requests_metered_total"]
    cost = tiered_cost_usd(reqs, cfg["pricing"], already_used_this_month=VALIDATION_REQUESTS)
    cost_alone = tiered_cost_usd(reqs + VALIDATION_REQUESTS, cfg["pricing"])
    p = meta[meta.role == "positive"]
    ok = meta.validation_status == "ok"

    # split x label (x subset) by split group
    t = meta.assign(cls=meta.candidate_subset.map({"positive": "positive", "primary": "neg_primary",
                                                   "reserve": "neg_reserve"}))
    split_tbl = (t.groupby("split").agg(split_groups=("split_group_id", "nunique"), facilities=("facility_id", "nunique"))
                 .join(pd.crosstab(t.split, t.cls)).reset_index())
    split_tbl["primary ratio"] = (split_tbl["neg_primary"] / split_tbl["positive"]).round(2)
    region = pd.crosstab(t.broad_region, [t.split, t.cls])
    region.columns = [f"{a} {b}" for a, b in region.columns]
    region = region.reset_index()
    gsize = pos.groupby("split_group_id").size()
    by_size = gsize.value_counts().sort_index()
    geo_tbl = (meta.groupby(["source_zoom", "mosaic_grid", "role"]).size().unstack(fill_value=0).reset_index())
    dup = aud["near_duplicates"]
    wm = aud["watermark_by_grid"]
    qa_by_role = pd.DataFrame(aud["qa_flags_by_role"]).fillna(0).astype(int).T.reset_index().rename(columns={"index": "qa_flags"})
    jp = plan["jitter_per_point"]
    ni = aud["no_imagery"]
    ph = meta.qa_flags.fillna("").str.contains("no_imagery_placeholder")
    usable = meta[(meta.role == "candidate_negative") & ~ph]
    n_short = int((usable.groupby("parent_point_id").size().clip(upper=3) < 3).sum()
                  + (len(p) - usable.parent_point_id.nunique()))

    L = [f"# External training set: data-preparation audit ({dt.date.today().isoformat()})", "",
         "**Status: data preparation only.** No model was trained; benchmark labels, splits and evaluation are "
         "unchanged. Candidate negatives and positives are **unreviewed**: only `confirmed_negative` tiles (and "
         "positives that pass review) may be used for training. Coordinate-bearing files and imagery are in the "
         f"git-ignored `{cfg['dataset']['dataset_dir']}/` and `{cfg['output_root']}/`.", "",

         "## 1. Source file, facility identity and split groups", "",
         f"- `Reviewed fence list.csv` (repo root, git-ignored): UTF-8 with BOM, columns exactly `Name, Latitude, Longitude`, "
         f"**{plan['source_rows']} rows**, no blank cells, all numeric and within WGS84 ranges (lat "
         f"{pos.latitude.min():.1f}..{pos.latitude.max():.1f}, lon {pos.longitude.min():.1f}..{pos.longitude.max():.1f}; "
         "southern/western signs present and plausible). Original strings are kept verbatim in `facility_name_original`, "
         "`latitude_original`, `longitude_original`.",
         f"- **Facility identity:** `facility_id = ext_<slug(facility_name_original)>`, one per source row: "
         f"**{plan['facilities']} unique facility IDs** for {plan['source_rows']} rows, no collisions, no names merged "
         f"(one-to-one: {plan['facility_id_one_to_one_with_rows']}). `point_id = <facility_id>__pos`.",
         f"- **Split groups (separate field):** `split_group_id` links rows closer than **{plan['split_group_link_km']} km** "
         "(single linkage, haversine between Yanliang's points) so that co-located facilities/sub-sites and their "
         "negatives (annulus up to 6 km) cannot fall on both sides of the train/val split. "
         f"**{plan['split_groups']} split groups**; {plan['multi_row_split_groups']} contain more than one facility "
         f"(sizes: {', '.join(f'{k} facilities x {v}' for k, v in by_size.items())}).",
         "- Not supplied, therefore derived or left blank: country (blank), coordinate type (`fence_segment`, as stated "
         "by Yanliang), label source (file + date). `broad_region` is a coarse lon/lat box, not a country.",
         f"- Exact duplicate coordinates: {int(pos.exact_duplicate_coordinate.sum())}; near duplicates < 100 m: "
         f"{int(pos.near_duplicate_lt_100m.sum())}; closest pair {pos.nearest_other_km.min():.2f} km. "
         f"{int(flags.str.contains('another_row_within_one_tile').sum())} rows have another row within one tile width "
         f"({plan['checks']['overlapping_tile_pairs']} overlapping positive-tile pairs, all inside one split group).",
         f"- Low precision: {', '.join(pos.loc[flags.str.contains('low_precision'), 'facility_name_original'])} (4 decimals, ~10 m).",
         "- Evaluation AOIs: no row within an AOI + "
         f"{cfg['aoi_exclusion_buffer_km']} km. Nearest: "
         + ", ".join(f"{k} {pos[f'dist_{k.lower()}_aoi_centre_km'].min():.0f} km from its centre "
                     f"({pos.loc[pos[f'dist_{k.lower()}_aoi_centre_km'].idxmin(), 'facility_name_original']})"
                     for k in cfg["evaluation_aois"]) + ". Balakovo is in the same oblast as the Saratov AOI (informational).",
         "- Facility mix (names only): power plants, research reactors, fuel cycle, enrichment, reprocessing, "
         "waste/spent-fuel storage and several defence-complex sites (e.g. Khushab, VNIIEF, VNIITF, Mayak, Plants 404/814).",
         "", f"Multi-facility split groups (all {len(groups)}; linking pairs are the < {plan['split_group_link_km']} km edges):", "",
         md_table(groups.sort_values("n_facilities", ascending=False)[
             ["split_group_id", "n_facilities", "facility_names_original", "max_pairwise_km", "linking_pairs_lt_threshold"]]
                  .assign(linking_pairs_lt_threshold=lambda x: x.linking_pairs_lt_threshold.str.replace(";", "; "))), "",

         "## 2. Positive tiles and jitter", "",
         f"Rule: tile centre = Yanliang's point moved by bearing ~ U[0, 360) deg and distance area-uniform in "
         f"[{j['min_m']}, {j['max_m']}] m, both from SHA-256(`{d['seed']}` | facility_id | point_id | `jitter`). The seed is "
         f"per source row/point, never per split group: the {jp['points_in_multi_row_groups']} points in multi-facility "
         f"split groups have {jp['distinct_jitter_vectors_among_them']} distinct offsets; {jp['distinct_seeds']} distinct "
         "seeds overall. Source coordinates are not modified; `--no-jitter` gives `tile_plan_nojitter.csv` (ablation).", "",
         f"- Verified for all {len(pos)} positives with each tile's exact auto_mosaic zoom/crop in Web-Mercator pixels: "
         f"fence point >= **{plan['jitter']['edge_margin_m_min']:.0f} m** from the nearest tile edge (threshold "
         f"{j['min_edge_margin_m']} m), failures {plan['jitter']['n_fail']}.",
         f"- Offset distance m (min/25/50/75/max): {q(p.jitter_m)}; in px: {q(p.fence_offset_px)}. Bearing quartiles "
         f"(deg): {q(p.jitter_bearing_deg, nd=0)}. Mean fence position ({p.fence_px_col.mean():.1f}, "
         f"{p.fence_px_row.mean():.1f}) px vs centre (1280, 1280). Plot: `jitter_and_centre_bias.png`.",
         "- Residual bias (by design, conservative): fences are always 137-471 px from the centre; benchmark fences can "
         "be anywhere, including partly outside a tile.", "",

         "## 3. Candidate negatives and the 3:1 subset", "",
         f"- Geometry: centres in the {n['annulus_inner_km']}-{n['annulus_outer_km']} km annulus around the parent fence point; "
         f">= {n['exclusion_km']} km from **every** listed fence point; axis-aligned separation >= {n['min_center_sep_km']} km "
         "from every other tile (no footprint overlap); outside all evaluation AOIs + buffer.",
         f"- **Why 4 were generated:** an oversampling margin so that ~3 per positive survive manual review "
         "(candidates near large complexes, on open water, or under cloud are expected to be rejected), extracted "
         "in one pass to avoid a second extraction round. It is not a methodological requirement.",
         f"- **Primary subset (3:1) = neg1, neg2, neg3** of every positive (deterministic by candidate index; on the "
         f"first attempt the four candidates sit ~90 deg apart, so the primary three cover three sides); **reserve = neg4**. Primary {plan['negatives']['primary']}, reserve "
         f"{plan['negatives']['reserve']}; all downloaded imagery is kept. Reserve tiles are used only to replace "
         "rejected primaries (same positive) or for ablations, never added on top by default.",
         "- Trade-off: a closer annulus matches terrain/imaging conditions better but raises the chance that other fences "
         "of the same complex are visible; 3 km from every listed point keeps the known point >= 1.66 km outside any "
         "negative tile, and manual review covers the rest. All negatives remain `candidate_negative`.", "",

         "## 4. Facility-level split (external data only)", "",
         md_table(split_tbl), "",
         f"Split unit = split group, assigned per broad region by a seeded hash order (val fraction {d['split']['val_fraction']}). "
         f"Facilities in both splits: {aud['facilities_in_both_splits']}; split groups in both splits: "
         f"{plan['checks']['groups_in_both_splits']}. Per-group counts: `split_group_counts.csv`. Saratov/Natanz/Fordow "
         "splits are untouched.", "", "By broad region:", "", md_table(region), "",

         "## 5. Extraction, geometry and cost", "",
         f"- `auto_mosaic`, tile width 1.9 km, 2560 px; Qiqi's script at the pinned commit, unmodified. "
         f"Tiles: **{int(ok.sum())} validated OK / {len(meta)}** ({int((ok & (meta.role == 'positive')).sum())}/{len(p)} "
         f"positives, {int((ok & (meta.role != 'positive')).sum())} candidate negatives); extraction status "
         f"{aud['extraction_status']}.",
         f"- **Footprint width** {aud['footprint_width_km_range'][0]:.4f}-{aud['footprint_width_km_range'][1]:.4f} km "
         f"(area {aud['footprint_area_km2_range'][0]:.3f}-{aud['footprint_area_km2_range'][1]:.3f} km^2; nominal 3.61); "
         f"**effective GSD** {aud['effective_mpp_range'][0]:.5f}-{aud['effective_mpp_range'][1]:.5f} m/px. Native m/px "
         f"{meta.native_mpp.min():.3f}-{meta.native_mpp.max():.3f}, resampled to 0.742.", "",
         md_table(geo_tbl.rename(columns={"source_zoom": "zoom", "mosaic_grid": "grid"})), "",
         f"- **Static Maps requests, metered (actual): {reqs}** for this dataset (every tile's count equals its planned "
         f"2 + grid^2). Plus {VALIDATION_REQUESTS} validation requests on 2026-10-13. Cost at the configured pricing "
         f"(checked {cfg['pricing']['checked_on']}, 10,000 free/month then $2/1000): **~${cost_alone:.2f}** for the "
         f"{reqs + VALIDATION_REQUESTS:,} requests made this month, assuming no other Static Maps use on the account "
         "(otherwise up to $2.00 per 1,000 more). Exact billing is in the Google Cloud console.",
         f"- Files containing an API key: {len(aud['files_containing_key'])}.", "",

         "## 6. Leakage and shortcut checks", "",
         "| check | result |", "|---|---|",
         f"| evaluation AOIs | every tile centre >= {min(plan['checks']['min_km_outside_evaluation_aoi_edge'].values())} km "
         "outside the nearest AOI edge |",
         f"| footprint overlap across splits / across split groups | {plan['checks']['overlaps_across_splits']} / "
         f"{plan['checks']['overlaps_across_split_groups']} |",
         f"| negative overlapping any tile | {plan['checks']['negative_overlapping_any_tile']} |",
         f"| same facility / split group in train and val | {aud['facilities_in_both_splits']} / {plan['checks']['groups_in_both_splits']} |",
         f"| near-duplicate images (dHash <= 12/256) | {dup['pairs']} pairs, of which {dup['placeholder_pairs']} are two "
         f"'no imagery' placeholders (identical grey tiles from different places) and **{dup['genuine_pairs']} genuine** "
         f"(cross-split {dup['genuine_cross_split']}) |",
         f"| watermark layout vs label | Cramer's V(grid, role) = {aud['cramers_v_grid_vs_role']} |",
         "| watermark contrast pos vs neg (artifact_audit frequency-map masks, per grid) | "
         + "; ".join(f"{k}x{k}: " + (f"AUC {v['auc_wm_contrast_pos_vs_neg']} (mask {v['mask_area_pct']}%, n={v['n_tiles']})"
                                      if v['mask_area_pct'] > 0 else f"no fixed-position mask exists (n={v['n_tiles']})")
                     for k, v in wm.items()) + " |",
         "| watermark row position (crop-offset phase, from manifest geometry) pos vs neg | "
         + "; ".join(f"{k}x{k}: AUC {v['auc_phase_pos_vs_neg']} (phase {v['phase_px_range'][0]:.0f}-{v['phase_px_range'][1]:.0f} px)"
                     for k, v in aud['watermark_phase_by_grid'].items()) + " |",
         f"| region vs label | Cramer's V = {aud['cramers_v_region_vs_role']} |",
         "| generation settings vs label | one preset for all; zoom/grid table above |",
         f"| centre bias of positives | fence 137-471 px from centre; mean offset ({p.fence_px_col.mean() - 1280:+.1f}, "
         f"{p.fence_px_row.mean() - 1280:+.1f}) px |",
         f"| simple image statistics pos vs neg (AUC), all tiles | {aud['simple_stat_auc_pos_vs_neg']} |",
         f"| ... excluding 'no imagery' placeholders | {aud['simple_stat_auc_pos_vs_neg_excl_placeholders']} "
         "(remaining gap = genuine content: facilities are built-up and textured; see decision 4) |",
         f"| 'no imagery' placeholder tiles | full: {ni['placeholder_tiles_by_role']} (by subset {ni['placeholder_by_subset']}); "
         f"partial: {ni['partial_tiles_by_role']} |",
         f"| blank / cloud / corrupt (heuristic flags, for review) | {aud['qa_flags']} |", "",
         "QA flags by role:", "", md_table(qa_by_role), "",

         "## 7. Manual-review artifacts (all decisions start empty)", "",
         "- `positive_review.csv` + `contact_sheets/positives/pos_sheet_NNN.png`: per positive, the full tile with "
         "Yanliang's point circled and a ~475 m zoom around it. Fields `positive_review_status` (`unreviewed`) and "
         "`positive_review_notes`. Labels are not changed; this only checks the signature is visible in current imagery.",
         "- `contact_sheets/negatives/sheet_NNN.png`: context view, one row per facility (positive with point circled, "
         "then neg1-neg4 marked PRIMARY/RESERVE). Decisions go in `candidate_negative_review.csv` (written by "
         "`negative_refinement.py`, see section 10).", "",

         "## 8. Finding: 'no imagery' placeholders pass Qiqi's script", "",
         "Where Google has no imagery at the requested zoom (mostly open water, e.g. Lake Ontario next to Pickering/"
         "Darlington, Sea of Japan coast), the Static Maps API returns a normal PNG with a grey (229, 227, 223) "
         "background and the text \"Sorry, we have no imagery here\". The script's `google_returned_no_imagery()` "
         "searches the response *bytes* for that text, so it never triggers; the tile is saved as `downloaded`. "
         f"Here this affects {sum(ni['placeholder_tiles_by_role'].values())} candidate negatives fully and "
         f"{sum(ni['partial_tiles_by_role'].values())} tiles partly (by role {ni['partial_tiles_by_role']}). They are flagged "
         "`no_imagery_placeholder` / `partial_no_imagery` and should be rejected as unusable (replace from reserve). "
         f"No positive is affected. Even using the reserve, {n_short} coastal positives would have fewer than 3 usable "
         "candidates after rejecting full placeholders (before any manual rejections). "
         "Worth telling Qiqi; the inland benchmark AOIs are unlikely to be affected, but it is cheap to scan them.", "",
         "Related, seen on the contact sheets and only partly auto-flagged (all in candidate negatives, mostly offshore): "
         "solid black no-data blocks inside sub-images (`black_nodata`), Google's rendered blue bathymetry instead of "
         "photographs at sea, and hard seams between sub-images of different dates. Reviewers should reject these as "
         "`reject_unusable`; positives show none of them.", "",
         "## 9. Decisions needed before training", "",
         "1. Review the primary candidate negatives; replace rejected ones (incl. all `no_imagery_placeholder`) from the "
         "reserve of the same positive.",
         "2. Review positives (e.g. Almaraz: the point sits inside the plant perimeter while the most obvious "
         "layered-fence compound is elsewhere in the tile); confirm with Yanliang where unclear.",
         f"3. Overlapping positives within a complex ({plan['checks']['overlapping_tile_pairs']} pairs): keep all (same split "
         "group, current plan) or de-duplicate for training.",
         "4. Negatives are easier than positives on simple statistics (texture/brightness AUC ~0.7 after removing "
         "placeholders): many candidates are farmland/forest/water. Consider adding non-nuclear industrial "
         "'hard negatives' so the model cannot learn 'built-up = positive'.",
         "5. Watermark handling (shared union mask) for training and evaluation alike; not applied here.",
         "6. Whether the defence-complex sites stay in; whether real country codes are needed.", ""]
    L += refinement_section(cfg, ddir)
    L += pilot_section(cfg, ddir)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "external_dataset_audit.md").write_text("\n".join(L))
    print(f"-> {REPORT_DIR / 'external_dataset_audit.md'}")


if __name__ == "__main__":
    main()
