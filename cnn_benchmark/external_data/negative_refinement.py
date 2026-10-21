"""Negative-set refinement for the external training set. Makes NO API calls and trains NO CNN.

Steps (all deterministic):
  A. nuisance/scene statistics for every extracted tile -> automatic rejection flags for candidate negatives
     (`auto_reject`, `auto_reject_reason`; several reasons possible). Imagery is never deleted.
  B. nearby-negative selection per positive: primary neg1..neg3 first, auto-rejected removed, reserve neg4 fills
     gaps, at most 3 (`nearby_negative_selected`, `selection_rank`). Shortages stay explicit.
  C. rapid review: 16-tile contact sheets of the selected negatives, a companion review CSV, a static local HTML
     index (no JS, no web dependency), and a compact positive review CSV.
  E. shortcut diagnostics on positives + selected negatives: per-feature AUCs and a nuisance-only classifier
     (logistic regression and a depth-3 tree) trained on train split groups and scored on val split groups.
  G. manifests (coordinate-free copies in outputs/external_data/, coordinate-bearing ones in the git-ignored
     dataset dir), with `training_eligible = false` until manual review is complete.
  H. leakage checks.

Manual review status is kept separate from automatic rejection: `manual_review_status` starts `unreviewed`.

    python negative_refinement.py            # (re)uses cached nuisance_features.csv if present
    python negative_refinement.py --recompute-features
"""
from __future__ import annotations

import argparse
import html
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from dataset_audit import PLACEHOLDER_RGB, REPORT_DIR
from geo import chebyshev_m, haversine_m
from geometry import REPO_ROOT, load_config

NEAR_TARGET = 3
WATER_LOCAL_STD_MAX = 4.0   # calm water is locally smooth (median 0.3-3.3 on examples); dark hazy forest is ~5-6
MANUAL_STATUSES = ["unreviewed", "confirmed_negative", "reject_visible_layered_fence", "reject_artifact",
                   "reject_ambiguous"]
POSITIVE_STATUSES = ["unreviewed", "fence_visible", "fence_not_visible", "ambiguous"]
# automatic rejection thresholds (fractions of tile pixels)
THRESH = {"full_placeholder": 0.5, "partial_placeholder": 0.02, "black_nodata": 0.01, "mostly_water": 0.6}
NUISANCE = ["brightness", "contrast", "edge_density", "texture_hp", "water_frac", "rendered_ocean_frac",
            "placeholder_frac", "black_frac", "wm_phase_px", "mosaic_grid", "source_zoom"]
# features available to the nuisance-only classifier (artifact flags are excluded: they are rejection criteria)
CLASSIFIER_FEATURES = ["brightness", "contrast", "edge_density", "texture_hp", "water_frac", "wm_phase_px",
                       "mosaic_grid", "source_zoom"]


# ------------------------------------------------------------------ A. nuisance statistics

def tile_features(png: str) -> dict:
    """Scene/nuisance statistics at 640 px (no learned features). Water = calm (locally smooth) blue-dominant pixels
    OR Google's rendered (bathymetry-style) ocean; thresholds calibrated on contact-sheet examples (a looser
    local-std limit of 8 classified dark hazy forest as water)."""
    out = {"png": png, "decode_error": ""}
    try:
        with Image.open(png) as im:
            im.load()
            rgb = np.asarray(im.convert("RGB").resize((640, 640), Image.BILINEAR), dtype=np.float32)
            exact = np.asarray(im.convert("RGB").reduce(4), dtype=np.int16)
    except Exception as e:  # corrupt / truncated
        out["decode_error"] = f"{type(e).__name__}: {e}"[:200]
        return out
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    g = rgb.mean(-1)
    loc = np.sqrt(np.maximum(ndimage.uniform_filter(g ** 2, 5) - ndimage.uniform_filter(g, 5) ** 2, 0))
    sob = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1)) / 8.0
    placeholder = (np.abs(exact - np.array(PLACEHOLDER_RGB)).max(-1) <= 1)
    black = exact.max(-1) <= 3
    calm_water = (B > R + 20) & (B >= G - 5) & (loc < WATER_LOCAL_STD_MAX)
    rendered = (B - R > 50) & (B > 120) & (R < 110)
    out.update(brightness=float(g.mean()), contrast=float(g.std()),
               edge_density=float((sob > 20).mean()), texture_hp=float(np.abs(g - ndimage.gaussian_filter(g, 2)).mean()),
               water_frac=float((calm_water | rendered).mean()), rendered_ocean_frac=float(rendered.mean()),
               placeholder_frac=float(placeholder.mean()), black_frac=float(black.mean()))
    return out


def compute_features(meta: pd.DataFrame, workers: int) -> pd.DataFrame:
    paths = [str(REPO_ROOT / p) for p in meta.png_path]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(tile_features, paths, chunksize=8))
    f = pd.DataFrame(rows)
    f.insert(0, "tile_id", meta.tile_id.values)
    return f.drop(columns=["png"])


def auto_reject(f: pd.Series) -> list[str]:
    r = []
    if f.get("decode_error"):
        return ["corrupt"]
    if f.placeholder_frac > THRESH["full_placeholder"]:
        r.append("full_placeholder")
    elif f.placeholder_frac > THRESH["partial_placeholder"]:
        r.append("partial_placeholder")
    if f.black_frac > THRESH["black_nodata"]:
        r.append("black_nodata")
    if f.water_frac > THRESH["mostly_water"] and f.placeholder_frac <= THRESH["full_placeholder"]:
        r.append("mostly_water")
    return r


# ------------------------------------------------------------------ B. selection

def select_nearby(neg: pd.DataFrame) -> pd.DataFrame:
    """Per parent positive: primary (neg1..3) in index order, then reserve; drop auto-rejected; keep first 3."""
    neg = neg.copy()
    neg["cand_index"] = neg.tile_id.str.extract(r"__neg(\d+)$")[0].astype(int)
    neg["subset_order"] = (neg.candidate_subset != "primary").astype(int)
    neg = neg.sort_values(["parent_point_id", "subset_order", "cand_index"])
    neg["selection_rank"] = np.nan
    usable = neg[~neg.auto_reject]
    rank = usable.groupby("parent_point_id").cumcount() + 1
    neg.loc[rank.index, "selection_rank"] = rank
    neg["nearby_negative_selected"] = neg.selection_rank.le(NEAR_TARGET).fillna(False)
    neg.loc[~neg.nearby_negative_selected, "selection_rank"] = np.nan
    return neg.drop(columns=["subset_order"])


# ------------------------------------------------------------------ C. review artifacts

def rapid_sheets(sel: pd.DataFrame, out_dir: Path, per_sheet: int = 16, ncol: int = 4, thumb: int = 460) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("review_sheet_*.png"):
        old.unlink()                                   # regenerated deterministically (our own derived files)
    try:
        big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except OSError:
        big = small = ImageFont.load_default()
    sel = sel.sort_values(["split", "split_group_id", "parent_point_id", "selection_rank"]).reset_index(drop=True)
    rows = []
    for s, start in enumerate(range(0, len(sel), per_sheet)):
        chunk = sel.iloc[start:start + per_sheet]
        nrow = int(np.ceil(len(chunk) / ncol))
        cell_h = thumb + 58
        sheet = Image.new("RGB", (ncol * (thumb + 8) + 8, nrow * cell_h + 8), "white")
        dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk.itertuples()):
            x, y = 8 + (k % ncol) * (thumb + 8), 8 + (k // ncol) * cell_h
            sheet.paste(Image.open(REPO_ROOT / r.png_path).convert("RGB").resize((thumb, thumb)), (x, y))
            dr.rectangle([x, y, x + 52, y + 34], fill="black")
            dr.text((x + 6, y + 3), f"{k + 1:02d}", fill="yellow", font=big)
            dr.text((x, y + thumb + 4), r.tile_id[:48], fill="black", font=small)
            dr.text((x, y + thumb + 24), f"{r.facility_name_original[:34]} | {r.split_group_id[:22]} | {r.split}",
                    fill="gray", font=small)
            rows.append({"tile_id": r.tile_id, "review_sheet": f"review_sheet_{s + 1:03d}.png", "sheet_cell": k + 1})
        sheet.save(out_dir / f"review_sheet_{s + 1:03d}.png")
    return pd.DataFrame(rows)


def write_html_index(review: pd.DataFrame, pos_compact: pd.DataFrame, out_dir: Path) -> Path:
    """Static, local, no-JS index: sheets as images; each tile ID links to its full-resolution PNG (zoom in browser)."""
    def rel(p):
        return os.path.relpath(REPO_ROOT / p, out_dir)
    h = ["<!doctype html><html><head><meta charset='utf-8'><title>External negative review</title>",
         "<style>body{font-family:sans-serif;margin:16px;max-width:1900px}img{max-width:100%;border:1px solid #ccc}"
         "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:2px 6px}</style></head><body>",
         "<h1>External training set: manual review</h1>",
         "<p>Enter decisions in <code>candidate_negative_review.csv</code> (column <code>manual_review_status</code>): "
         + ", ".join(f"<code>{s}</code>" for s in MANUAL_STATUSES) + ". Positives: <code>positive_review_compact.csv</code> ("
         + ", ".join(f"<code>{s}</code>" for s in POSITIVE_STATUSES) + "). Click a tile ID for the full 2560 px tile.</p>",
         "<h2>Selected nearby negatives</h2>"]
    for sheet, sub in review[review.review_sheet.notna()].groupby("review_sheet"):
        h.append(f"<h3 id='{sheet}'>{sheet}</h3><a href='{sheet}'><img src='{sheet}' loading='lazy'></a><table>"
                 "<tr><th>cell</th><th>tile</th><th>facility</th><th>split group</th><th>split</th></tr>")
        for r in sub.sort_values("sheet_cell").itertuples():
            h.append(f"<tr><td>{int(r.sheet_cell):02d}</td><td><a href='{html.escape(rel(r.png_path))}'>{r.tile_id}</a></td>"
                     f"<td>{html.escape(r.facility_name_original)}</td><td>{r.split_group_id}</td><td>{r.split}</td></tr>")
        h.append("</table>")
    h.append("<h2>Positive review sheets</h2>")
    for sheet in sorted(pos_compact.positive_sheet.dropna().unique()):
        h.append(f"<h3>{sheet}</h3><a href='../contact_sheets/positives/{sheet}'>"
                 f"<img src='../contact_sheets/positives/{sheet}' loading='lazy'></a>")
    h.append("</body></html>")
    path = out_dir / "index.html"
    path.write_text("\n".join(h))
    return path


# ------------------------------------------------------------------ E. shortcut diagnostics

def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def shortcut_diagnostics(df: pd.DataFrame, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    y = (df.label_role == "positive").astype(int).to_numpy()
    per_feature = {}
    for c in NUISANCE:
        v = df[c].astype(float).to_numpy()
        a = auc(y, v) if np.nanstd(v) > 0 else 0.5
        per_feature[c] = {"auc": round(a, 3), "separation": round(abs(a - 0.5) * 2, 3),
                          "pos_median": round(float(np.median(v[y == 1])), 4), "neg_median": round(float(np.median(v[y == 0])), 4)}
    X = df[CLASSIFIER_FEATURES].astype(float).to_numpy()
    tr, va = (df.split == "train").to_numpy(), (df.split == "val").to_numpy()
    models = {"logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
              "tree_depth3": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed)}
    res = {}
    for name, m in models.items():
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        # grouped 5-fold CV over split groups (stability check, same features)
        cv = []
        for a_idx, b_idx in GroupKFold(5).split(X, y, df.split_group_id):
            mm = type(m)(**m.get_params()) if name == "tree_depth3" else make_pipeline(
                StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
            mm.fit(X[a_idx], y[a_idx])
            cv.append(roc_auc_score(y[b_idx], mm.predict_proba(X[b_idx])[:, 1]))
        if name == "logistic_regression":
            coef = m[-1].coef_[0]
            imp = {f: round(float(c), 3) for f, c in sorted(zip(CLASSIFIER_FEATURES, coef), key=lambda t: -abs(t[1]))}
        else:
            imp = {f: round(float(c), 3) for f, c in sorted(zip(CLASSIFIER_FEATURES, m.feature_importances_), key=lambda t: -t[1]) if c > 0}
        res[name] = {"val_auc": round(roc_auc_score(y[va], p), 3), "val_accuracy": round(accuracy_score(y[va], p > 0.5), 3),
                     "val_balanced_accuracy": round(balanced_accuracy_score(y[va], p > 0.5), 3),
                     "val_n": int(va.sum()), "val_pos": int(y[va].sum()),
                     "grouped_cv5_auc_mean": round(float(np.mean(cv)), 3), "grouped_cv5_auc_sd": round(float(np.std(cv)), 3),
                     ("std_coefficients" if name == "logistic_regression" else "feature_importance"): imp}
    res["majority_class_val_accuracy"] = round(float(1 - y[va].mean()), 3)

    # How many positive-like ("hard") negatives would be needed? (a) analytic, per feature: mixing a fraction w of
    # negatives distributed like positives gives AUC' = (1-w)*AUC + w*0.5 (exact for a single-feature AUC).
    def w_needed(a, target):
        a = max(a, 1 - a)
        return 0.0 if a <= target else round((a - target) / (a - 0.5), 3)
    res["hard_negative_fraction_needed_per_feature"] = {
        c: {"to_auc_0.65": w_needed(v["auc"], 0.65), "to_auc_0.60": w_needed(v["auc"], 0.60)}
        for c, v in per_feature.items() if c in CLASSIFIER_FEATURES}
    # (b) empirical, for the classifier: add pseudo-hard negatives = bootstrap copies of positives' nuisance vectors
    # (the optimistic case where hard negatives match positives on every nuisance statistic) to train AND val.
    rng = np.random.default_rng(seed)
    curve = []
    neg_n = int((y == 0).sum())
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        k = int(round(w / (1 - w) * neg_n)) if w < 1 else 0
        Xa, ya, sa = [X], [y], [df.split.to_numpy()]
        if k:
            for sp in ("train", "val"):
                pos_idx = np.where((y == 1) & (df.split == sp).to_numpy())[0]
                kk = int(round(k * (df.split == sp).mean()))
                pick = rng.choice(pos_idx, kk, replace=True)
                Xa.append(X[pick] + rng.normal(0, 1e-6, X[pick].shape)); ya.append(np.zeros(kk, int)); sa.append(np.full(kk, sp))
        Xa, ya, sa = np.vstack(Xa), np.concatenate(ya), np.concatenate(sa)
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        m.fit(Xa[sa == "train"], ya[sa == "train"])
        curve.append({"hard_negative_fraction_of_negatives": w, "added_per_split_total": k,
                      "val_auc": round(roc_auc_score(ya[sa == "val"], m.predict_proba(Xa[sa == "val"])[:, 1]), 3)})
    res["pseudo_hard_negative_curve"] = curve
    return {"per_feature": per_feature, "classifier": res, "n": int(len(df)), "n_pos": int(y.sum()),
            "features_used_by_classifier": CLASSIFIER_FEATURES}


# ------------------------------------------------------------------ H. leakage

def leakage_checks(man: pd.DataFrame, plan: pd.DataFrame, meta: pd.DataFrame, cfg: dict) -> dict:
    sel = man.merge(plan[["tile_id", "center_lat", "center_lon"]], on="tile_id", how="left")
    lat, lon = sel.center_lat.to_numpy(), sel.center_lon.to_numpy()
    cheb = chebyshev_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(cheb, np.inf)
    ov = np.triu(cheb < cfg["tile_km"] * 1000, 1)
    i, j = np.where(ov)
    roles = sel.label_role.to_numpy()
    pos = plan[plan.role == "positive"]
    neg = sel[sel.label_role == "nearby_negative"]
    dmin_fence = haversine_m(neg.center_lat.to_numpy()[:, None], neg.center_lon.to_numpy()[:, None],
                             pos.latitude.to_numpy()[None, :], pos.longitude.to_numpy()[None, :]).min(1)
    # near-duplicate imagery among the selected pool (dHash from the audit)
    h = meta.set_index("tile_id").loc[sel.tile_id, "dhash"].tolist()
    b = np.unpackbits(np.array([np.frombuffer(bytes.fromhex(x), np.uint8) for x in h]), axis=1)
    ham = (b[:, None, :] != b[None, :, :]).sum(-1)
    np.fill_diagonal(ham, 999)
    di, dj = np.where(np.triu(ham <= 12, 1))
    sp = sel.split.to_numpy()
    return {
        "split_groups_in_both_splits": int((sel.groupby("split_group_id").split.nunique() > 1).sum()),
        "overlapping_pairs_total": int(len(i)),
        "overlapping_positive_negative_pairs": int(sum(1 for a, c in zip(i, j) if {roles[a], roles[c]} == {"positive", "nearby_negative"})),
        "overlapping_pairs_across_splits": int(sum(sp[a] != sp[c] for a, c in zip(i, j))),
        "min_negative_centre_to_any_fence_point_km": round(float(dmin_fence.min()) / 1000, 3),
        "negative_footprint_contains_a_fence_point": int((dmin_fence < cfg["tile_km"] * 1000 / 2 * np.sqrt(2)).sum()),
        "near_duplicate_pairs_selected": int(len(di)),
        "near_duplicate_pairs_across_splits": int(sum(sp[a] != sp[c] for a, c in zip(di, dj))),
        "tiles_with_conflicting_labels": int((man.groupby("tile_id").label_role.nunique() > 1).sum()),
        "facility_ids_as_both_positive_and_hard_negative": 0,   # no hard negatives exist yet (see hard_negatives.py)
    }


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recompute-features", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config()
    ddir = REPO_ROOT / cfg["dataset"]["dataset_dir"]
    meta = pd.read_csv(ddir / "external_tiles_metadata.csv")
    plan = pd.read_csv(ddir / "tile_plan.csv")

    fpath = ddir / "nuisance_features.csv"
    if args.recompute_features or not fpath.exists():
        compute_features(meta, args.workers).to_csv(fpath, index=False)
    feats = pd.read_csv(fpath, keep_default_na=False, na_values=[""])
    feats["decode_error"] = feats["decode_error"].fillna("")
    m = meta.drop(columns=[c for c in feats.columns if c != "tile_id" and c in meta.columns]).merge(feats, on="tile_id")
    m["mosaic_grid"] = m["mosaic_grid"].astype(int)
    m["source_zoom"] = m["source_zoom"].astype(int)

    # A. automatic rejection (computed for every tile; applied to candidate negatives only)
    reasons = [auto_reject(r) for _, r in m.iterrows()]
    m["artifact_flags"] = [";".join(r) for r in reasons]
    is_neg = m.role == "candidate_negative"
    m["auto_reject"] = is_neg & m.artifact_flags.ne("")
    m["auto_reject_reason"] = np.where(m.auto_reject, m.artifact_flags, "")

    # B. selection
    neg = select_nearby(m[is_neg])
    m = m.merge(neg[["tile_id", "selection_rank", "nearby_negative_selected"]], on="tile_id", how="left")
    m["nearby_negative_selected"] = m["nearby_negative_selected"].fillna(False).astype(bool)
    n_sel = m[m.nearby_negative_selected].groupby("parent_point_id").size().reindex(
        m.loc[m.role == "positive", "tile_id"], fill_value=0)
    availability = n_sel.value_counts().reindex(range(NEAR_TARGET + 1), fill_value=0)

    # C. review artifacts
    rdir = ddir / "review"
    sel = m[m.nearby_negative_selected]
    where = rapid_sheets(sel, rdir)
    review_path = ddir / "candidate_negative_review.csv"
    old = pd.read_csv(review_path, dtype=str, keep_default_na=False) if review_path.exists() else pd.DataFrame()
    review = m[is_neg][["tile_id", "facility_id", "facility_name_original", "split_group_id", "split", "candidate_subset",
                        "auto_reject", "auto_reject_reason", "nearby_negative_selected", "selection_rank",
                        "png_path"]].merge(where, on="tile_id", how="left")
    review["manual_review_status"] = "unreviewed"
    review["reviewer"] = ""
    review["review_notes"] = ""
    if "manual_review_status" in old:
        keep = old[old.manual_review_status.isin(MANUAL_STATUSES[1:])][["tile_id", "manual_review_status", "reviewer", "review_notes"]]
        review = review.drop(columns=["manual_review_status", "reviewer", "review_notes"]).merge(keep, on="tile_id", how="left")
        review["manual_review_status"] = review.manual_review_status.fillna("unreviewed")
        review = review.fillna({"reviewer": "", "review_notes": ""})
    elif "review_decision" in old and (old.review_decision != "").any():
        raise SystemExit("candidate_negative_review.csv holds decisions in the old vocabulary; map them before rerunning")
    review = review.sort_values(["review_sheet", "sheet_cell", "tile_id"], na_position="last")
    review.to_csv(review_path, index=False)

    pos_rev = pd.read_csv(ddir / "positive_review.csv", dtype=str, keep_default_na=False)
    pos_compact = pos_rev[["split_group_id", "facility_id", "facility_name_original", "tile_id", "split", "positive_sheet",
                           "positive_sheet_index", "positive_review_status", "positive_review_notes"]] \
        .sort_values(["split_group_id", "facility_id"])
    pos_compact.to_csv(ddir / "positive_review_compact.csv", index=False)
    html_path = write_html_index(review, pos_compact, rdir)

    # status columns for manifests
    m = m.merge(review[["tile_id", "manual_review_status"]], on="tile_id", how="left")
    pstat = pos_rev.set_index("tile_id")["positive_review_status"]
    m.loc[m.role == "positive", "manual_review_status"] = m.loc[m.role == "positive", "tile_id"].map(pstat)
    m["label_role"] = np.where(m.role == "positive", "positive", "nearby_negative")
    m["training_eligible"] = (((m.role == "positive") & (m.manual_review_status == "fence_visible")) |
                              ((m.role == "candidate_negative") & m.nearby_negative_selected & ~m.auto_reject &
                               (m.manual_review_status == "confirmed_negative")))

    # G. manifests
    geom = ["preset", "source_zoom", "mosaic_grid", "crop_px", "native_mpp", "footprint_width_km", "footprint_area_km2",
            "effective_mpp", "wm_phase_px"]
    base = ["tile_id", "facility_id", "facility_name_original", "parent_point_id", "split_group_id", "split", "broad_region",
            "label_role", "candidate_subset", "auto_reject", "auto_reject_reason", "artifact_flags",
            "nearby_negative_selected", "selection_rank", "manual_review_status", "training_eligible",
            "validation_status"] + geom
    private_extra = ["latitude_original", "longitude_original", "center_lat", "center_lon", "jitter_m",
                     "jitter_bearing_deg", "neg_dist_from_parent_fence_km", "png_path"]
    cand = m[is_neg]
    draft = m[(m.role == "positive") | m.nearby_negative_selected]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in (("external_nearby_candidates", cand), ("external_nearby_selected", m[m.nearby_negative_selected]),
                     ("external_training_manifest_draft", draft)):
        df[base].to_csv(REPORT_DIR / f"{name}.csv", index=False)                       # coordinate-free (trackable)
        df[base + private_extra].to_csv(ddir / f"{name}_private.csv", index=False)     # coordinates (git-ignored)
    m.to_csv(ddir / "external_tiles_metadata_refined.csv", index=False)

    # E. shortcut diagnostics (positives + selected nearby negatives); before = positives + all candidates
    diag_sel = shortcut_diagnostics(draft)
    diag_all = shortcut_diagnostics(m.assign(label_role=np.where(m.role == "positive", "positive", "nearby_negative")))

    # H. leakage
    leak = leakage_checks(draft[base], plan, meta, cfg)

    summary = {
        "thresholds": THRESH,
        "auto_reject": {"candidates": int(is_neg.sum()), "rejected": int(m.auto_reject.sum()),
                        "by_reason": m.loc[m.auto_reject, "auto_reject_reason"].str.split(";").explode().value_counts().to_dict(),
                        "by_subset": m[m.auto_reject].candidate_subset.value_counts().to_dict(),
                        "corrupt_any_role": int((m.decode_error != "").sum()),
                        "positives_with_artifact_flags": m[(m.role == "positive") & m.artifact_flags.ne("")]
                        [["tile_id", "artifact_flags"]].to_dict("records")},
        "selection": {"selected": int(m.nearby_negative_selected.sum()),
                      "from_primary": int((m.nearby_negative_selected & (m.candidate_subset == "primary")).sum()),
                      "from_reserve": int((m.nearby_negative_selected & (m.candidate_subset == "reserve")).sum()),
                      "positives_by_n_selected": {int(k): int(v) for k, v in availability.items()},
                      "short_positives": n_sel[n_sel < NEAR_TARGET].sort_values().to_dict(),
                      "by_split": m[m.nearby_negative_selected].split.value_counts().to_dict()},
        "review": {"rapid_sheets": int(where.review_sheet.nunique()), "tiles_on_sheets": int(len(where)),
                   "html_index": str(html_path.relative_to(REPO_ROOT)),
                   "negative_manual_status": review.manual_review_status.value_counts().to_dict(),
                   "positive_review_status": pos_rev.positive_review_status.value_counts().to_dict()},
        "training_eligible": int(m.training_eligible.sum()),
        "shortcut_selected_pool": diag_sel, "shortcut_all_candidates": diag_all,
        "leakage": leak,
    }
    (ddir / "refinement_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("shortcut")}, indent=1, default=str))
    print(json.dumps({"per_feature": diag_sel["per_feature"], "classifier": {k: v for k, v in diag_sel["classifier"].items()
                                                                              if k != "hard_negative_fraction_needed_per_feature"}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
