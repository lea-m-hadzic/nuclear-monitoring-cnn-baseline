"""CANDIDATE leakage-aware splits (pending discussion with the authors; not final).

A. saratov_spatial_cv: Saratov z16 parents are grouped into ~5 compact geographic blocks (k-means on
   grid row/col, fixed seed). All tiles of any reconstructed facility (targets, candidates, non-targets)
   are forced into one block. Fold k: test = block k, val = block (k+1) mod K, buffer = tiles within
   `buffer_tiles` (Chebyshev, grid units) of a test or val tile that are not themselves test/val; these
   are excluded from training. Everything else is train.
B. loao (leave-one-AOI-out): test = one AOI (all tiles); train = the other two AOIs, each with one
   spatial block held out as val (plus buffer). The val block is chosen to lose the fewest positive
   tiles from training (val + its buffer), among blocks with >= 1 positive in val (ties -> lowest id),
   because positives are scarce (Natanz has 8).

Usage: python src/splits.py      -> outputs/splits/*.csv and printed leakage checks
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import facility_mapping as fm
from build_parent_table import build as build_parent_table
from common import ensure_output_dir, load_config

STATUS = "CANDIDATE_pending_discussion"


def kmeans_blocks(xy: np.ndarray, k: int, seed: int, n_init: int = 20, iters: int = 100) -> np.ndarray:
    """Plain k-means (k-means++ init, best of n_init); deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        c = [xy[rng.integers(len(xy))]]
        for _ in range(k - 1):
            d2 = np.min(((xy[:, None] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(xy[rng.choice(len(xy), p=d2 / d2.sum())])
        c = np.array(c, float)
        for _ in range(iters):
            lab = ((xy[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.array([xy[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((xy - c[lab]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    # Relabel blocks north-west -> south-east for readability.
    order = np.argsort([xy[best == j].mean(0) @ np.array([1.0, 0.5]) for j in range(k)])
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[best]


def assign_blocks(tiles: pd.DataFrame, membership: pd.DataFrame, k: int, seed: int) -> pd.Series:
    """Block id per tile; facility footprints are moved wholesale to their majority block."""
    xy = tiles[["row", "col"]].to_numpy(float)
    block = pd.Series(kmeans_blocks(xy, k, seed), index=tiles.tile_id.values)
    for fid, g in membership.groupby("facility_id"):
        ids = [t for t in g.tile_id if t in block.index]
        if ids and block[ids].nunique() > 1:
            block[ids] = block[ids].value_counts().sort_index().idxmax()
    return block


def _buffer(tiles: pd.DataFrame, core: set, width: int) -> set:
    rc = tiles.set_index("tile_id")[["row", "col"]]
    core_rc = rc.loc[list(core)].to_numpy()
    out = set()
    for t, (r, c) in rc.iterrows():
        if t in core:
            continue
        if (np.maximum(np.abs(core_rc[:, 0] - r), np.abs(core_rc[:, 1] - c)) <= width).any():
            out.add(t)
    return out


def _roles(tiles, test: set, val: set, width: int) -> pd.Series:
    buf = _buffer(tiles, test | val, width)
    role = pd.Series("train", index=tiles.tile_id.values)
    role[list(buf)] = "buffer"
    role[list(val)] = "val"
    role[list(test)] = "test"
    return role


def saratov_spatial_cv(pt: pd.DataFrame, mem: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    k, seed, width = cfg["splits"]["saratov_n_blocks"], cfg["splits"]["seed"], cfg["splits"]["buffer_tiles"]
    tiles = pt[pt.region == "Saratov"]
    block = assign_blocks(tiles, mem[mem.region == "Saratov"], k, seed)
    parts = []
    for fold in range(k):
        test = set(block.index[block == fold]); val = set(block.index[block == (fold + 1) % k])
        role = _roles(tiles, test, val, width)
        parts.append(pd.DataFrame({"split": "saratov_spatial_cv", "fold": fold, "region": "Saratov",
                                   "tile_id": role.index, "role": role.values, "block": block[role.index].values}))
    return pd.concat(parts, ignore_index=True)


def _choose_val_block(tiles: pd.DataFrame, block: pd.Series, k: int, width: int) -> set:
    pos = set(tiles.tile_id[tiles.human_target == 1])
    best = None
    for b in range(k):
        val = set(block.index[block == b])
        if not val & pos:
            continue
        lost = len((val | _buffer(tiles, val, width)) & pos)
        if best is None or lost < best[0]:
            best = (lost, val)
    return best[1]


def loao(pt: pd.DataFrame, mem: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    k, seed, width = cfg["splits"]["saratov_n_blocks"], cfg["splits"]["seed"], cfg["splits"]["buffer_tiles"]
    regions = cfg["regions"]
    blocks = {r: assign_blocks(pt[pt.region == r], mem[mem.region == r], k, seed) for r in regions}
    parts = []
    for fold, test_region in enumerate(["Natanz", "Fordow", "Saratov"]):
        for r in regions:
            tiles = pt[pt.region == r]
            if r == test_region:
                role = pd.Series("test", index=tiles.tile_id.values)
            else:
                val = _choose_val_block(tiles, blocks[r], k, width)
                role = _roles(tiles, set(), val, width)
            parts.append(pd.DataFrame({"split": "loao", "fold": fold, "test_region": test_region, "region": r,
                                       "tile_id": role.index, "role": role.values,
                                       "block": blocks[r][role.index].values}))
    return pd.concat(parts, ignore_index=True)


def check_manifest(man: pd.DataFrame, pt: pd.DataFrame, mem: pd.DataFrame, width: int) -> list[str]:
    """Return a list of problems (empty = all checks passed)."""
    problems = []
    if man.duplicated(["split", "fold", "region", "tile_id"]).any():
        problems.append("duplicate tile assignment within a fold")
    z16 = set(zip(pt.region, pt.tile_id))
    if not set(zip(man.region, man.tile_id)) <= z16:
        problems.append("manifest contains tiles that are not z16 parents (possible z17 / quadrant leakage)")
        return problems
    if pt.source_image_relpath.duplicated().any():
        problems.append("duplicate source images among parent tiles")
    rc = pt.set_index(["region", "tile_id"])[["row", "col"]]
    for (split, fold), g in man.groupby(["split", "fold"]):
        regions_in = set(g.region)
        expected = set(zip(pt.region[pt.region.isin(regions_in)], pt.tile_id[pt.region.isin(regions_in)]))
        if set(zip(g.region, g.tile_id)) != expected:
            problems.append(f"{split}/{fold}: not every parent tile has exactly one role")
        for region, h in g.groupby("region"):
            role = h.set_index("tile_id").role
            m = mem[mem.region == region]
            for fid, f in m.groupby("facility_id"):
                roles = set(role.reindex(f.tile_id).dropna())
                if len(roles & {"train", "val", "test"}) > 1:
                    problems.append(f"{split}/{fold}: facility {fid} split across roles {sorted(roles)}")
            held = h[h.role.isin(["test", "val"])].tile_id
            train = h[h.role == "train"].tile_id
            if len(held) and len(train):
                a = rc.loc[[(region, t) for t in held]].to_numpy()
                b = rc.loc[[(region, t) for t in train]].to_numpy()
                cheb = np.maximum(np.abs(a[:, None, 0] - b[None, :, 0]), np.abs(a[:, None, 1] - b[None, :, 1]))
                if (cheb <= width).any():
                    problems.append(f"{split}/{fold}/{region}: train tile within {width} tile(s) of a held-out tile")
    if "saratov_spatial_cv" in set(man.split):
        s = man[(man.split == "saratov_spatial_cv") & (man.role == "test")]
        if not (s.groupby("tile_id").size() == 1).all() or s.tile_id.nunique() != (pt.region == "Saratov").sum():
            problems.append("saratov_spatial_cv: each tile must be a test tile exactly once")
        blk = man[man.split == "saratov_spatial_cv"].drop_duplicates("tile_id").set_index("tile_id").block
        for fid, f in mem[mem.region == "Saratov"].groupby("facility_id"):
            if blk.reindex(f.tile_id).nunique() > 1:
                problems.append(f"facility {fid} spans multiple spatial blocks")
    return problems


def summarize(man: pd.DataFrame, pt: pd.DataFrame) -> pd.DataFrame:
    x = man.merge(pt[["region", "tile_id", "human_target", "exclude_from_training", "nuclear_target"]],
                  on=["region", "tile_id"])
    keys = ["split", "fold"] + (["test_region"] if "test_region" in x else []) + ["role"]
    return (x.groupby(keys, dropna=False)
            .agg(tiles=("tile_id", "size"), positives=("human_target", "sum"),
                 nuclear_target_tiles=("nuclear_target", "sum"), excluded=("exclude_from_training", "sum"))
            .reset_index())


def main() -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cfg = load_config()
    pt = build_parent_table(cfg)
    _, assignments = fm.reconstruct(cfg)
    mem = fm.parent_membership(assignments)
    out = ensure_output_dir("splits")
    width = cfg["splits"]["buffer_tiles"]
    for name, man in [("saratov_spatial_cv", saratov_spatial_cv(pt, mem, cfg)), ("loao", loao(pt, mem, cfg))]:
        man.insert(0, "split_status", STATUS)
        man.to_csv(out / f"{name}.csv", index=False)
        summ = summarize(man, pt)
        summ.to_csv(out / f"{name}_summary.csv", index=False)
        problems = check_manifest(man, pt, mem, width)
        print(f"== {name} ({STATUS}): {'all leakage checks passed' if not problems else problems}")
        print(summ.to_string(index=False))
    print(f"Wrote manifests to {out}")


if __name__ == "__main__":
    main()
