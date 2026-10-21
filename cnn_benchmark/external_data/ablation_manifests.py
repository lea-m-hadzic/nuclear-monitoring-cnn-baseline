"""Tile-ID manifests for the ResNet-50 negative-composition ablation (A nearby-only, B hard-only, C mixed,
plus the C-matched sampling sensitivity condition). No training, no API calls.

    python ablation_manifests.py         # -> outputs/ablation/manifests/{A,B,C,C_matched}.csv + checks.json (tile IDs only)
                                         #    + external_imagery/dataset/ablation_manifests/*.csv (with png_path, private)

Design (all conditions): the same 287 external positives (their existing split groups) and the same number of
negatives per split, fixed by the confirmed hard negatives' split: train 126, val 31 (157 total).
  A          nearby-only: 126 + 31 nearby negatives, seeded per split from the cleaned nearby pool
             (`nearby_negative_selected` & not auto-rejected), order SHA-256(seed | 'ablation_nearby' | tile_id).
  B          hard-only: all 157 confirmed hard negatives.
  C          mixed: hard half = first ceil(n/2) confirmed hard negatives per split in SHA-256(seed | 'ablation_hard' |
             tile_id) order (train 63, val 16); nearby half = the first floor(n/2) of A's per-split order (train 63,
             val 15), i.e. nested inside A. 79 hard + 78 nearby.
  C_matched  SAMPLING SENSITIVITY ANALYSIS (not the main dataset): C's hard half + a nearby half of the same size per
             split chosen by hard_negative_review_ui.matched_nearby (contrast + water-fraction matching to positives).
Sampling never crosses train/val, never replaces, and never modifies the nearby pool or the review CSV.
  shared_val CHECKPOINT-SELECTION set used by every condition: all val positives + A's val nearby negatives + all
             confirmed hard val negatives (union, so C's val negatives are a subset). Checked: val split only, and no
             tile / split group / facility shared with any condition's training split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geometry import REPO_ROOT, load_config
from hard_negative_review_ui import METADATA_FILES, matched_nearby, validate
from negative_refinement import NUISANCE

OUT = Path(__file__).resolve().parents[1] / "outputs" / "ablation" / "manifests"
COLS = ["tile_id", "label", "role", "split", "split_group_id", "facility_id", "category", "png_path"]
PUBLIC_COLS = [c for c in COLS if c != "png_path"]      # png filenames embed tile-centre coordinates: private only
PRIVATE = REPO_ROOT / "external_imagery" / "dataset" / "ablation_manifests"


def h(*p) -> str:
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()


def load_pools(cfg):
    d = REPO_ROOT / cfg["dataset"]["dataset_dir"]
    m = pd.read_csv(d / "external_tiles_metadata_refined.csv")
    pos = m[m.role == "positive"].assign(label=1, role="positive", category="positive")
    near = m[m.nearby_negative_selected & ~m.auto_reject].assign(label=0, role="nearby", category="nearby")
    rv = validate(cfg)
    conf = set(rv[rv.manual_review_status == "confirmed_hard_negative"].tile_id)
    hn = pd.concat([pd.read_csv(d / f) for f in METADATA_FILES if (d / f).exists()], ignore_index=True)
    hard = hn[hn.tile_id.isin(conf)].assign(label=0, role="hard")
    assert len(hard) == len(conf), "confirmed hard negatives missing metadata"
    return pos, near, hard


def build(cfg) -> dict:
    seed = cfg["dataset"]["seed"]
    pos, near, hard = load_pools(cfg)
    n_split = hard.split.value_counts().to_dict()                       # e.g. {'train': 126, 'val': 31}
    near_order = near.assign(o=[h(seed, "ablation_nearby", t) for t in near.tile_id]).sort_values(["split", "o"])
    hard_order = hard.assign(o=[h(seed, "ablation_hard", t) for t in hard.tile_id]).sort_values(["split", "o"])
    A_near, C_near, C_hard = [], [], []
    for sp, n in n_split.items():
        ns = near_order[near_order.split == sp]
        hs = hard_order[hard_order.split == sp]
        if len(ns) < n:
            raise SystemExit(f"nearby pool too small in {sp}: {len(ns)} < {n}")
        A_near.append(ns.head(n))
        C_near.append(ns.head(n // 2))
        C_hard.append(hs.head(n - n // 2))
    A_near, C_near, C_hard = (pd.concat(x) for x in (A_near, C_near, C_hard))
    cm_n = C_near.split.value_counts().to_dict()
    feat_cols = ["tile_id", "split"] + NUISANCE
    cm_near_ids = matched_nearby(pos[feat_cols], near[feat_cols], cm_n, seed).tile_id
    CM_near = near[near.tile_id.isin(cm_near_ids)]
    conds = {"A": pd.concat([pos, A_near]), "B": pd.concat([pos, hard]), "C": pd.concat([pos, C_hard, C_near]),
             "C_matched": pd.concat([pos, C_hard, CM_near])}
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    checks = {"sampling_seed": seed, "negatives_per_split": n_split, "conditions": {}}
    for name, df in conds.items():
        df = df[COLS].sort_values(["split", "label", "tile_id"], ascending=[True, False, True]).reset_index(drop=True)
        df.to_csv(PRIVATE / f"{name}.csv", index=False)                  # with png_path (git-ignored)
        df[PUBLIC_COLS].to_csv(OUT / f"{name}.csv", index=False)         # tile IDs only (trackable)
        conds[name] = df
        checks["conditions"][name] = {
            "counts": df.groupby(["split", "role"]).size().unstack(fill_value=0).to_dict("index"),
            "n_pos": int(df.label.sum()), "n_neg": int((df.label == 0).sum()),
            "duplicate_tiles": int(df.tile_id.duplicated().sum()),
            "split_groups_in_both_splits": int((df.groupby("split_group_id").split.nunique() > 1).sum()),
            "facilities_in_both_splits": int((df.groupby("facility_id").split.nunique() > 1).sum()),
            "sha256": hashlib.sha256((OUT / f"{name}.csv").read_bytes()).hexdigest()}
    sv = pd.concat([conds["A"][conds["A"].split == "val"], conds["B"][conds["B"].split == "val"]]).drop_duplicates("tile_id")
    sv = sv.sort_values(["label", "role", "tile_id"], ascending=[False, True, True]).reset_index(drop=True)
    sv.to_csv(PRIVATE / "shared_val.csv", index=False)
    sv[PUBLIC_COLS].to_csv(OUT / "shared_val.csv", index=False)
    tr_all = pd.concat([c[c.split == "train"] for c in conds.values()])
    checks["shared_val"] = {
        "counts": sv.groupby("role").size().to_dict(), "only_val_split": bool((sv.split == "val").all()),
        "tiles_in_any_train": int(sv.tile_id.isin(tr_all.tile_id).sum()),
        "split_groups_in_any_train": int(sv.split_group_id.isin(tr_all.split_group_id).sum()),
        "facilities_in_any_train": int(sv.facility_id.isin(tr_all.facility_id).sum()),
        "contains_every_main_condition_val": {n: bool(set(conds[n][conds[n].split == "val"].tile_id) <= set(sv.tile_id)) for n in ("A", "B", "C")},
        "sha256": hashlib.sha256((OUT / "shared_val.csv").read_bytes()).hexdigest()}
    checks["identical_positives"] = all(set(c[c.label == 1].tile_id) == set(conds["A"][conds["A"].label == 1].tile_id) for c in conds.values())
    checks["C_nearby_subset_of_A"] = set(conds["C"][conds["C"].role == "nearby"].tile_id) <= set(conds["A"][conds["A"].role == "nearby"].tile_id)
    checks["C_matched_hard_equals_C_hard"] = set(conds["C_matched"][conds["C_matched"].role == "hard"].tile_id) == set(conds["C"][conds["C"].role == "hard"].tile_id)
    checks["negatives_equal_per_split"] = len({json.dumps(c[c.label == 0].split.value_counts().sort_index().to_dict()) for c in conds.values()}) == 1
    (OUT / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    return checks


if __name__ == "__main__":
    print(json.dumps(build(load_config()), indent=1, default=str))
