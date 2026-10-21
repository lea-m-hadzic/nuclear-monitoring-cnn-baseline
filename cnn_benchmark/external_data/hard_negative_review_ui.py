"""Manual-review workflow for the 68-site hard-negative pilot. Never sets a review status.

    python hard_negative_review_ui.py build      # review images + order table (the reviewer UI is hard_negative_review_app.py)
    python hard_negative_review_ui.py validate   # check hard_negative_review.csv, report counts by status
    python hard_negative_review_ui.py diagnose   # A/B/C nuisance-only diagnostics (only if confirmed labels exist)
    python hard_negative_review_ui.py tree       # depth-3 tree on C (and A' for reference), same split/CV
    python hard_negative_review_ui.py matched    # C-random vs C-matched (contrast + water matched nearby half)

`hard_negative_review.csv` (git-ignored dataset dir) is the ONLY place decisions are entered: edit the
`manual_review_status` column (and optionally `reviewer`, `review_notes`), save, then run `validate`.
Labels can be set by clicking in the local app (hard_negative_review_app.py), which edits this same CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from geometry import REPO_ROOT, load_config
from hard_negative_pilot import HN_STATUSES, diag_row, h, mixed_set
from negative_refinement import NUISANCE

GROUPS = [  # review order requested by Lea (2026-10-15)
    ("a. Likely unusable / off-site / placeholder",
     ["hn_thermal_power_plant_25__hn", "hn_thermal_power_plant_24__hn", "hn_hv_substation_01__hn",
      "hn_industrial_facility_16__hn", "hn_rail_yard_port_06__hn", "hn_hv_substation_03__hn",
      "hn_hv_substation_04__hn", "hn_rail_yard_port_09__hn"]),
    ("b. Weak or ambiguous scenes",
     ["hn_thermal_power_plant_09__hn", "hn_logistics_warehouse_02__hn", "hn_rail_yard_port_01__hn",
      "hn_rail_yard_port_04__hn", "hn_logistics_warehouse_07__hn", "hn_thermal_power_plant_07__hn"]),
    ("c. Fence inspection", ["hn_thermal_power_plant_21__hn", "hn_thermal_power_plant_14__hn"]),
]
GUIDE = [
    "<b>confirmed_hard_negative</b>: only if the tile is a valid built/industrial scene AND shows no recognizable layered fencing.",
    "<b>reject_artifact</b>: the source point is clearly off-site, or the scene is unusable (placeholder, no-data, mostly water/haze).",
    "<b>reject_target_signature</b>: recognizable layered fencing is visible anywhere relevant in the tile (kept for audit; never relabelled).",
    "Tank bund walls, roads, shadows, walls and ordinary single fencing are <i>not</i> the layered-fence target.",
    "<b>reject_ambiguous</b>: when unsure.",
]


def paths(cfg):
    d = REPO_ROOT / cfg["dataset"]["dataset_dir"]
    return d, d / "hard_negative_review.csv", d / "review"


def load_review(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


PLAN_FILES = ("hard_negative_tile_plan_final.csv", "hard_negative_tile_plan_scaleup.csv")    # pilot 68 + scale-up 130
METADATA_FILES = ("hard_negative_metadata_private_thermal_power_plant.csv", "hard_negative_metadata_private_replacement.csv",
                  "hard_negative_metadata_private_scaleup.csv")
PRIORITY_GROUPS = ("a. ", "b. ")


def all_plans(d: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(d / f, dtype=str, keep_default_na=False) for f in PLAN_FILES if (d / f).exists()],
                     ignore_index=True)


def notes_map(d: Path, rv: pd.DataFrame) -> dict[str, str]:
    """Pre-screen notes (pilot), scale-up pre-flags, and a plain-language note for automatic artifact flags."""
    notes = {}
    for f in ("hard_negative_power_prescreen.csv", "hard_negative_replacement_prescreen.csv"):
        if (d / f).exists():
            x = pd.read_csv(d / f, dtype=str, keep_default_na=False)
            notes.update(zip(x.tile_id, x.prescreen_observation))
    if (d / "hard_negative_scaleup_flags.csv").exists():
        x = pd.read_csv(d / "hard_negative_scaleup_flags.csv", dtype=str, keep_default_na=False)
        notes.update({t: f"PRE-FLAGGED: {n}" for t, fl, n in zip(x.tile_id, x.flag, x.note) if fl == "suspicious_pre_review"})
    for t, a in zip(rv.tile_id, rv.artifact_flags):
        if a and t not in notes:
            notes[t] = f"AUTOMATIC ARTIFACT FLAG: {a}"
    return notes


def ordered(rv: pd.DataFrame, d: Path | None = None) -> pd.DataFrame:
    """Scale-up first (a: automatic artifact flags, b: pre-flagged suspicious, c: rest), then the reviewed pilot
    (in its original review order)."""
    d = d or REPO_ROOT / load_config()["dataset"]["dataset_dir"]
    scale = set(pd.read_csv(d / PLAN_FILES[1]).tile_id) if (d / PLAN_FILES[1]).exists() else set()
    flags = pd.read_csv(d / "hard_negative_scaleup_flags.csv", dtype=str) if (d / "hard_negative_scaleup_flags.csv").exists() else pd.DataFrame(columns=["tile_id", "flag"])
    susp = list(flags[flags.flag == "suspicious_pre_review"].tile_id)
    new = rv[rv.tile_id.isin(scale)]
    a = list(new[new.artifact_flags != ""].sort_values("tile_id").tile_id)
    b = [t for t in susp if t not in a]
    c = list(new[~new.tile_id.isin(a + b)].sort_values(["category", "tile_id"]).tile_id)
    seq = [("a. Scale-up: automatic artifact flags", a), ("b. Scale-up: pre-flagged suspicious sites", b),
           ("c. Scale-up: remaining new candidates", c)]
    pilot_ids = [t for _, ids in GROUPS for t in ids] + \
        list(rv[~rv.tile_id.isin(scale) & ~rv.tile_id.isin([t for _, ids in GROUPS for t in ids])].sort_values(["category", "tile_id"]).tile_id)
    seq.append(("d. Pilot (68; reviewed 2026-10-15)", [t for t in pilot_ids if t in set(rv.tile_id)]))
    order, group = {}, {}
    for g, ids in seq:
        for t in ids:
            order[t], group[t] = len(order), g
    missing = set(rv.tile_id) - set(order)
    if missing:
        raise SystemExit(f"tiles without a review group: {sorted(missing)[:5]}")
    return rv.assign(review_order=rv.tile_id.map(order) + 1, review_group=rv.tile_id.map(group)).sort_values("review_order")


def ensure_images(cfg, rv: pd.DataFrame) -> Path:
    """Full-tile (840 px) and ~475 m zoom JPEGs per tile, derived from the PNGs (which are only read)."""
    d, _, rdir = paths(cfg)
    plan = all_plans(d).set_index("tile_id")
    img = rdir / "hn_images"
    img.mkdir(parents=True, exist_ok=True)
    for r in rv.itertuples():
        full, zoom = img / f"{r.tile_id}_full.jpg", img / f"{r.tile_id}_zoom.jpg"
        if full.exists() and zoom.exists():
            continue
        p = plan.loc[r.tile_id]
        im = Image.open(REPO_ROOT / r.png_path).convert("RGB")
        im.resize((840, 840)).save(full, quality=88)
        cx, cy, half = float(p.facility_px_col), float(p.facility_px_row), 320
        im.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((840, 840)).save(zoom, quality=90)
    return img


def build(cfg):
    """Prepare review images + the review-order table (the interactive app is hard_negative_review_app.py)."""
    d, path, rdir = paths(cfg)
    rv = ordered(load_review(path), d)
    ensure_images(cfg, rv)
    rv[["review_order", "review_group", "tile_id", "facility_name_original", "category", "manual_review_status"]] \
        .to_csv(rdir / "hard_negative_review_order.csv", index=False)
    old = rdir / "hard_negative_review.html"
    if old.exists():
        old.unlink()                     # superseded by the local review app
    print(f"images + order ready ({len(rv)} candidates). Launch the reviewer: python hard_negative_review_app.py")


def validate(cfg) -> pd.DataFrame:
    d, path, _ = paths(cfg)
    rv = load_review(path)
    plan = all_plans(d)
    problems = []
    bad = sorted(set(rv.manual_review_status) - set(HN_STATUSES))
    if bad:
        problems.append(f"invalid statuses {bad} (allowed {HN_STATUSES}); check spelling/whitespace")
    if rv.tile_id.duplicated().any():
        problems.append(f"duplicate tile_ids {rv.tile_id[rv.tile_id.duplicated()].tolist()}")
    if set(rv.tile_id) != set(plan.tile_id):
        problems.append(f"tile set differs from the hard-negative plans: missing {sorted(set(plan.tile_id) - set(rv.tile_id))}, "
                        f"extra {sorted(set(rv.tile_id) - set(plan.tile_id))}")
    art = rv[(rv.manual_review_status == "confirmed_hard_negative") & (rv.artifact_flags != "")]
    if len(art):
        problems.append(f"confirmed tiles carrying automatic artifact flags (check): {art.tile_id.tolist()}")
    print("counts by status:", rv.manual_review_status.value_counts().to_dict())
    print(pd.crosstab(rv.category, rv.manual_review_status).to_string())
    print("PROBLEMS:\n  " + "\n  ".join(problems) if problems else "CSV valid.")
    if problems:
        raise SystemExit(1)
    return rv


def datasets(cfg) -> dict[str, tuple[pd.DataFrame, str]]:
    """A, A', B, C exactly as used by `diagnose` (confirmed hard negatives only; seeded nearby sampling)."""
    d, _, _ = paths(cfg)
    rv = validate(cfg)
    conf = rv[rv.manual_review_status == "confirmed_hard_negative"].tile_id
    if len(conf) == 0:
        raise SystemExit("No confirmed_hard_negative labels yet: B/C not run (by design).")
    feats = pd.concat([pd.read_csv(d / f) for f in METADATA_FILES if (d / f).exists()])
    keep = ["tile_id", "split", "split_group_id", "label_role"] + NUISANCE
    hard = feats[feats.tile_id.isin(conf)].assign(label_role="hard_negative")[keep]
    m = pd.read_csv(d / "external_tiles_metadata_refined.csv")
    A = m[(m.role == "positive") | m.nearby_negative_selected][keep]
    pos, near = A[A.label_role == "positive"], A[A.label_role != "positive"]
    seed = cfg["dataset"]["seed"]
    mix = mixed_set(len(pos), near, hard, seed)
    ctl = pd.concat([near[near.split == s].assign(o=lambda x: [h(seed, "ctl", i) for i in x.tile_id]).sort_values("o")
                     .head(int((mix.split == s).sum())).drop(columns="o") for s in ("train", "val")])
    return {"A": (A, f"{len(near)} nearby (auto-cleaned)"),
            "A'": (pd.concat([pos, ctl]), "control for C"),
            "B": (pd.concat([pos, hard]), f"{len(hard)} confirmed (train {int((hard.split == 'train').sum())}, "
                                          f"val {int((hard.split == 'val').sum())})"),
            "C": (pd.concat([pos, mix]), "per split: all confirmed hard + equal number of nearby (seeded)")}


def diagnose(cfg):
    d, _, _ = paths(cfg)
    ds = datasets(cfg)
    names = {"A": "A: positives vs cleaned nearby negatives", "A'": "A': positives vs nearby, size-matched to C",
             "B": "B: positives vs confirmed hard negatives", "C": "C: positives vs 1:1 nearby + confirmed hard"}
    rows = [diag_row(names[k], df, note) for k, (df, note) in ds.items()]
    (d / "hard_negative_diagnostics.json").write_text(json.dumps(rows, indent=1, default=str))
    for r in rows:
        print(json.dumps(r, default=str))


def tree(cfg):
    """Depth-3 decision tree on the nuisance features for C (and A' for reference): same split/CV as the LR."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    from negative_refinement import CLASSIFIER_FEATURES, shortcut_diagnostics
    d, _, _ = paths(cfg)
    ds = datasets(cfg)
    out = {}
    for k in ("C", "A'"):
        df = ds[k][0]
        res = shortcut_diagnostics(df)["classifier"]
        t = res["tree_depth3"]
        y = (df.label_role == "positive").astype(int).to_numpy()
        tr = (df.split == "train").to_numpy()
        m = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
        m.fit(df[CLASSIFIER_FEATURES].astype(float).to_numpy()[tr], y[tr])      # identical to the scored tree
        out[k] = {"tree_grouped_cv_auc": f"{t['grouped_cv5_auc_mean']} +/- {t['grouped_cv5_auc_sd']}",
                  "tree_val_auc": t["val_auc"], "tree_val_balanced_accuracy": t["val_balanced_accuracy"],
                  "tree_feature_importance": t["feature_importance"],
                  "lr_grouped_cv_auc": f"{res['logistic_regression']['grouped_cv5_auc_mean']} +/- {res['logistic_regression']['grouped_cv5_auc_sd']}",
                  "splits": export_text(m, feature_names=CLASSIFIER_FEATURES, show_weights=True, decimals=3)}
    (d / "hard_negative_tree_diagnostic.json").write_text(json.dumps(out, indent=1, default=str))
    for k, v in out.items():
        print(f"== {k}")
        print(json.dumps({x: y for x, y in v.items() if x != "splits"}, indent=1))
        print(v["splits"])


MATCH_FEATURES = ("contrast", "water_frac")


def matched_nearby(pos: pd.DataFrame, near: pd.DataFrame, n_by_split: dict[str, int], seed: str) -> pd.DataFrame:
    """C-matched nearby half (sampling ablation; the nearby pool itself is only read).

    Per split separately (train / val; tiles keep their split groups):
      1. anchors = n_by_split[split] positives of that split, taken in SHA-256(seed | 'match_anchor' | tile_id) order;
      2. features = MATCH_FEATURES only, each mapped to its empirical-CDF rank (average ranks / n) over the split's
         positives + nearby pool, so the zero-heavy water fraction and contrast share one scale;
      3. for each anchor in that order, the nearest still-unused nearby negative in rank space (Euclidean;
         ties broken by SHA-256(seed | 'match_tie' | tile_id)) is selected, without replacement.
    Only the positive/negative distinction is used; watermark, grid and zoom are not used.
    """
    out = []
    for sp, k in n_by_split.items():
        P = pos[pos.split == sp].copy()
        N = near[near.split == sp].copy()
        if k > len(N) or k > len(P):
            raise SystemExit(f"cannot match {k} nearby negatives in split {sp} (positives {len(P)}, nearby {len(N)})")
        both = pd.concat([P[list(MATCH_FEATURES)], N[list(MATCH_FEATURES)]])
        ranks = both.rank(method="average") / len(both)
        rp, rn = ranks.iloc[:len(P)].to_numpy(), ranks.iloc[len(P):].to_numpy()
        porder = np.argsort([h(seed, "match_anchor", t) for t in P.tile_id])[:k]
        tie = np.array([int(h(seed, "match_tie", t)[:12], 16) for t in N.tile_id], dtype=float) / 16 ** 12 * 1e-9
        used = np.zeros(len(N), bool)
        pick = []
        for i in porder:
            dist = np.sqrt(((rn - rp[i]) ** 2).sum(1)) + tie
            dist[used] = np.inf
            j = int(np.argmin(dist))
            used[j] = True
            pick.append(j)
        out.append(N.iloc[pick])
    sel = pd.concat(out)
    assert not sel.tile_id.duplicated().any()
    return sel


def matched(cfg):
    """C-random (unchanged) vs C-matched: logistic regression + depth-3 tree, distributions, counts."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    from negative_refinement import CLASSIFIER_FEATURES, shortcut_diagnostics
    d, _, _ = paths(cfg)
    ds = datasets(cfg)
    c_random = ds["C"][0]
    A = ds["A"][0]
    pos, near = A[A.label_role == "positive"], A[A.label_role != "positive"]
    hard = c_random[c_random.label_role == "hard_negative"]
    near_random = c_random[(c_random.label_role != "positive") & (c_random.label_role != "hard_negative")]
    n_by_split = hard.split.value_counts().to_dict()
    near_matched = matched_nearby(pos, near, n_by_split, cfg["dataset"]["seed"])
    c_matched = pd.concat([pos, hard, near_matched])
    pd.DataFrame({"tile_id": near_matched.tile_id, "split": near_matched.split}).to_csv(d / "c_matched_nearby_selection.csv", index=False)
    res = {}
    for name, df in (("C-random", c_random), ("C-matched", c_matched)):
        r = shortcut_diagnostics(df)
        lr, tr = r["classifier"]["logistic_regression"], r["classifier"]["tree_depth3"]
        neg = df[df.label_role != "positive"]
        res[name] = {"counts": {"train_pos": int(((df.label_role == "positive") & (df.split == "train")).sum()),
                                "val_pos": int(((df.label_role == "positive") & (df.split == "val")).sum()),
                                "train_neg": int((neg.split == "train").sum()), "val_neg": int((neg.split == "val").sum())},
                     "lr": {"grouped_cv_auc": f"{lr['grouped_cv5_auc_mean']} +/- {lr['grouped_cv5_auc_sd']}",
                            "val_auc": lr["val_auc"], "val_balanced_accuracy": lr["val_balanced_accuracy"],
                            "top_coefficients": dict(list(lr["std_coefficients"].items())[:4])},
                     "tree": {"grouped_cv_auc": f"{tr['grouped_cv5_auc_mean']} +/- {tr['grouped_cv5_auc_sd']}",
                              "val_auc": tr["val_auc"], "val_balanced_accuracy": tr["val_balanced_accuracy"],
                              "feature_importance": tr["feature_importance"]},
                     "feature_aucs": {f: v["auc"] for f, v in r["per_feature"].items() if f in CLASSIFIER_FEATURES}}
        y = (df.label_role == "positive").astype(int).to_numpy()
        trn = (df.split == "train").to_numpy()
        m = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0).fit(df[CLASSIFIER_FEATURES].astype(float).to_numpy()[trn], y[trn])
        res[name]["tree_splits"] = export_text(m, feature_names=CLASSIFIER_FEATURES, show_weights=True, decimals=3)

    def summ(x):
        return {"contrast_median": round(float(x.contrast.median()), 2),
                "contrast_iqr": [round(float(x.contrast.quantile(.25)), 2), round(float(x.contrast.quantile(.75)), 2)],
                "water_median": round(float(x.water_frac.median()), 4),
                "water_p75": round(float(x.water_frac.quantile(.75)), 4),
                "share_water_gt_5pct": round(float((x.water_frac > 0.05).mean()), 3)}
    dist = {"positives": summ(pos), "nearby pool (799)": summ(near), "nearby random (C-random)": summ(near_random),
            "nearby matched (C-matched)": summ(near_matched), "confirmed hard": summ(hard),
            "all negatives C-random": summ(pd.concat([hard, near_random])),
            "all negatives C-matched": summ(pd.concat([hard, near_matched]))}
    res["distributions"] = dist
    res["method"] = matched_nearby.__doc__
    (d / "c_matched_diagnostic.json").write_text(json.dumps(res, indent=1, default=str))
    for name in ("C-random", "C-matched"):
        print(f"== {name}", json.dumps({k: v for k, v in res[name].items() if k != "tree_splits"}, indent=1))
    print(pd.DataFrame(dist).T.to_string())
    print("== C-matched tree splits\n" + res["C-matched"]["tree_splits"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "validate", "diagnose", "tree", "matched"])
    cfg = load_config()
    {"build": build, "validate": validate, "diagnose": diagnose, "tree": tree, "matched": matched}[ap.parse_args().cmd](cfg)


if __name__ == "__main__":
    main()
