"""Saratov auto_mosaic rescoring sensitivity (no retraining). Frozen outputs are only read.

Saratov's frozen benchmark tiles are Qiqi's legacy --mosaic2x2 z16 tiles; training / Natanz / Fordow are auto_mosaic.
This re-extracts the 1,387 Saratov parent tiles with `--auto_mosaic --output_px 2560` (preset auto_mosaic_1p9km, all other
flags unchanged) at the exact frozen parent-tile centres, builds a separate 1280 px cache with the frozen decode_reduce,
applies the frozen geometric watermark mask rule using each new tile's own mosaic geometry, and scores it with the three
frozen condition-C checkpoints. Nothing is fine-tuned, calibrated or re-thresholded.

    python saratov_auto_mosaic_rescore.py extract [--workers 4]   # Static Maps (key from env; never printed)
    python saratov_auto_mosaic_rescore.py cache
    python saratov_auto_mosaic_rescore.py score [--device mps]     # also re-scores the legacy cache as a reproduction check
    python saratov_auto_mosaic_rescore.py report

Fresh extraction root: external_imagery/saratov_auto_mosaic_2026-10-22/ (git-ignored; refused if it exists without this
script's marker, so no tile from another extraction mode can be reused). Cache: cache/saratov_auto_mosaic_1280px/.
Outputs: results/robustness/saratov_auto_mosaic/ (tables) and figures/fig_saratov_legacy_vs_auto_mosaic.png.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parent
sys.path[:0] = [str(HERE), str(PKG / "src")]
FRESH = REPO / "external_imagery" / "saratov_auto_mosaic_2026-10-22"
MARKER = FRESH / ".saratov_auto_mosaic_rescore_root"
PRESET = "auto_mosaic_1p9km"
CACHE = PKG / "cache" / "saratov_auto_mosaic_1280px"
LEGACY_CACHE = PKG / "cache" / "decoded_1280px"
OUT = PKG.parent / "results" / "robustness" / "saratov_auto_mosaic"   # extraction_summary.csv + legacy re-scores git-ignored
RUNS = PKG / "outputs" / "ablation" / "runs" / "C"
SEEDS = (0, 1, 2)
BUDGETS = (10, 20, 30)
TARGETS = (0.25, 0.5, 0.75, 0.9)
WHS = ("SAR-WHS-01", "SAR-WHS-02", "SAR-WHS-03")
TOPS = (1, 5, 10, 20)


def tiles() -> pd.DataFrame:
    pt = pd.read_csv(PKG / "outputs" / "parent_tiles.csv")
    return pt[pt.region == "Saratov"].reset_index(drop=True)


def point_dir(tile_id: str) -> Path:
    return FRESH / PRESET / "saratov" / tile_id


# ---------------------------------------------------------------- extract
def extract(workers: int):
    import extract_external_tiles as X
    from geometry import load_config
    if FRESH.exists() and not MARKER.exists():
        raise SystemExit(f"{FRESH} exists but was not created by this script: refusing")
    cfg = load_config()
    key = X.api_key_or_exit(cfg)
    vend = X.check_vendor(cfg, allow_mismatch=False)
    FRESH.mkdir(parents=True, exist_ok=True)
    MARKER.touch()
    t = tiles()
    todo = [r for r in t.itertuples() if not X.is_complete(point_dir(r.tile_id), 1)]
    print(f"{len(t)} tiles, {len(t) - len(todo)} already complete in this fresh root, {len(todo)} to extract", flush=True)

    def one(r):
        p = {"point_id": r.tile_id, "facility_id": "saratov", "latitude": float(r.center_lat), "longitude": float(r.center_lon)}
        rec = X.extract_point(p, PRESET, 0.0, point_dir(r.tile_id), key, vend)
        return r.tile_id, rec["status"], rec["requests_metered"]

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for tid, st, n in ex.map(one, todo):
            done += 1
            if st != "complete" or done % 50 == 0:
                print(f"{done}/{len(todo)} {tid} {st} requests={n}", flush=True)
    summary()


def summary() -> dict:
    rows = []
    for r in tiles().itertuples():
        d = point_dir(r.tile_id)
        rec = json.loads((d / "extraction_record.json").read_text()) if (d / "extraction_record.json").exists() else {}
        m = pd.read_csv(d / "manifest.csv").iloc[0] if (d / "manifest.csv").exists() else None
        png = next((d / "tiles").glob("*.png"), None) if (d / "tiles").exists() else None
        rows.append({"tile_id": r.tile_id, "center_lat": r.center_lat, "center_lon": r.center_lon,
                     "record_status": rec.get("status", "missing"), "requests": rec.get("requests_metered"),
                     "vendor_status": None if m is None else m.status, "vendor_message": None if m is None else m.message,
                     "zoom": None if m is None else m.zoom, "grid": None if m is None else m.mosaic_grid_n,
                     "crop_px": None if m is None else m.crop_px, "native_mpp": None if m is None else m.native_mpp,
                     "manifest_lat": None if m is None else m.center_lat, "manifest_lon": None if m is None else m.center_lon,
                     "png": None if png is None else str(png.relative_to(REPO)), "png_bytes": None if png is None else png.stat().st_size})
    s = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "extraction_summary.csv", index=False)
    print(s.record_status.value_counts().to_dict(), s.vendor_status.value_counts().to_dict(), "requests", s.requests.sum())
    return s


# ---------------------------------------------------------------- cache
def cache():
    import decoded_cache
    from PIL import Image
    import io
    s = pd.read_csv(OUT / "extraction_summary.csv")
    bad = s[(s.record_status != "complete") | (s.vendor_status != "downloaded")]
    if len(bad):
        raise SystemExit(f"{len(bad)} tiles not cleanly extracted, e.g. {bad.tile_id.head().tolist()}: stopping")
    if ((s.center_lat - s.manifest_lat).abs().max() > 5e-8) or ((s.center_lon - s.manifest_lon).abs().max() > 5e-8):
        raise SystemExit("vendor tile centre differs from the frozen parent-tile centre")
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in s.itertuples():
        raw = (REPO / r.png).read_bytes()
        with Image.open(io.BytesIO(raw)) as im:
            size = im.size
        arr = decoded_cache.decode_reduce(raw)          # requires 2560x2560, RGB, exact 2x box reduce -> 1280
        out = CACHE / f"Saratov__{r.tile_id}.npy"
        tmp = out.with_suffix(".npy.tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, out)
        rows.append({"region": "Saratov", "tile_id": r.tile_id, "cache_file": out.name, "source_png": r.png,
                     "source_px": f"{size[0]}x{size[1]}", "source_sha256": hashlib.sha256(raw).hexdigest(),
                     "cache_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "zoom": int(r.zoom), "grid": int(r.grid),
                     "crop_px": int(r.crop_px), "native_mpp": float(r.native_mpp)})
    pd.DataFrame(rows).to_csv(CACHE / "cache_manifest.csv", index=False)
    print(len(rows), "cached")


# ---------------------------------------------------------------- score
def AutoMosaicMasks(man: pd.DataFrame):
    """Frozen geometric-mask rule (watermark_geometry.geometry_mask) with each fresh tile's own grid / crop. A MaskBank
    subclass so NpyTileDataset applies it through its unchanged per-tile path (fill_rgb on .get(tile_id))."""
    from watermark_geometry import MaskBank

    class _M(MaskBank):
        def __init__(self, geo):
            self.geo = geo

        def get(self, key):
            from watermark_geometry import geometry_mask
            return geometry_mask(*self.geo[key])

    return _M({f"Saratov__{r.tile_id}": (int(r.grid), float(r.crop_px)) for r in man.itertuples()})


def score_cache(run: Path, cache_dir: Path, mask, device_name: str) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader
    from common import load_config
    from model import build_model, pick_device
    from train_ablation import NpyTileDataset
    from watermark_geometry import MaskBank
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    meta = ck["meta"]
    assert meta["condition"] == "C" and meta["watermark_mask"] == "geometric"
    if MaskBank().template_sha256 != meta["watermark_template_sha256"]:
        raise SystemExit("watermark template differs from the one used in training")
    if mask is None:                                     # legacy reproduction: the frozen geometry table itself
        mb = MaskBank()
        if mb.table_sha256 != meta["watermark_geometry_sha256"]:
            raise SystemExit("frozen geometry table changed")
        mask = mb
    t = tiles()
    ids = "Saratov__" + t.tile_id
    device = pick_device(device_name)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    dl = DataLoader(NpyTileDataset(ids, t.human_target.fillna(0), cache_dir, mask, train=False),
                    batch_size=load_config()["training"]["batch_size"])
    sc = []
    with torch.no_grad():
        for x, _, _ in dl:
            sc += torch.sigmoid(model(x.to(device)).squeeze(1).float().cpu()).tolist()
    return t[["region", "tile_id", "human_target"]].assign(score=sc, seed=meta["seed"])


def score(device: str):
    man = pd.read_csv(CACHE / "cache_manifest.csv")
    assert len(man) == len(tiles())
    masks = AutoMosaicMasks(man)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        run = RUNS / f"seed{s}"
        a = score_cache(run, CACHE, masks, device)
        a.to_csv(OUT / f"scores_auto_mosaic_seed{s}.csv", index=False)
        leg = score_cache(run, LEGACY_CACHE, None, device)
        frozen = pd.read_csv(run / "benchmark" / "scores.csv").query("region == 'Saratov'")
        j = leg.merge(frozen, on="tile_id", suffixes=("", "_frozen"))
        dev = float((j.score - j.score_frozen).abs().max())
        print(f"seed{s}: legacy re-score vs frozen scores.csv max |diff| = {dev:.2e} over {len(j)} tiles", flush=True)
        leg.to_csv(OUT / f"scores_legacy_rescored_seed{s}.csv", index=False)
        (OUT / f"legacy_reproduction_seed{s}.json").write_text(json.dumps({"max_abs_diff": dev, "n": len(j)}))


# ---------------------------------------------------------------- report
def metrics(scores: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    import facility_mapping as fm
    from common import load_config
    from evaluation import evaluate_parent_scores
    from train_ablation import auroc, pr_auc
    cfg = load_config()
    pt = pd.read_csv(PKG / "outputs" / "parent_tiles.csv")
    fac, asg = fm.reconstruct(cfg)
    fr = evaluate_parent_scores(scores.assign(region="Saratov"), pt, fac, fm.parent_membership(asg), "Saratov",
                                aoi_area_km2=cfg["geometry"]["aoi_area_km2_nominal"]["Saratov"],
                                recall_targets=cfg["evaluation"]["recall_targets"])
    c = fr.curve
    y, sc = scores.human_target.fillna(0).astype(int).to_numpy(), scores.score.to_numpy()
    m = {"tile_auroc": auroc(y, sc), "tile_pr_auc": pr_auc(y, sc)}
    for b in BUDGETS:
        m[f"recall_at_{b}pct"] = float(c[c.pct_aoi_reviewed <= b + 1e-9].iloc[-1].facility_recall)
    for t in TARGETS:
        hit = c[c.facility_recall >= t - 1e-12]
        m[f"area_pct_for_{int(t * 100)}pct_recall"] = float(hit.pct_aoi_reviewed.iloc[0]) if len(hit) else np.nan
    fh = fr.first_hits.set_index("facility_id")
    for w in WHS:
        m[f"first_hit_pct_{w}"] = float(fh.loc[w, "first_hit_pct_aoi_expected"])
    # cross-check against the evaluator's own recall-target table
    rt = fr.recall_targets.set_index("recall_target").pct_aoi_reviewed
    for t in (0.5, 0.75, 0.9):
        assert abs(rt[t] - m[f"area_pct_for_{int(t * 100)}pct_recall"]) < 1e-9
    return m, c.assign()


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, curves, ranks = [], {}, []
    for s in SEEDS:
        frozen = pd.read_csv(RUNS / f"seed{s}" / "benchmark" / "scores.csv").query("region == 'Saratov'")[["tile_id", "human_target", "score"]]
        auto = pd.read_csv(OUT / f"scores_auto_mosaic_seed{s}.csv")[["tile_id", "human_target", "score"]]
        for name, sc in (("legacy (frozen)", frozen), ("auto_mosaic", auto)):
            m, c = metrics(sc)
            rows.append({"seed": s, "imagery": name, **m})
            curves[(s, name)] = c
        j = frozen.merge(auto, on="tile_id", suffixes=("_legacy", "_auto"))
        r = {"seed": s, "n_tiles": len(j), "spearman_scores": float(j.score_legacy.corr(j.score_auto, method="spearman"))}
        for k in TOPS:
            n = int(np.ceil(len(j) * k / 100))
            a = set(j.nlargest(n, "score_legacy").tile_id)
            b = set(j.nlargest(n, "score_auto").tile_id)
            r[f"top{k}pct_n"] = n
            r[f"top{k}pct_overlap"] = len(a & b)
            r[f"top{k}pct_overlap_frac"] = len(a & b) / n
            r[f"top{k}pct_jaccard"] = len(a & b) / len(a | b)
        ranks.append(r)
    seed = pd.DataFrame(rows)
    seed.to_csv(OUT / "saratov_auto_mosaic_rescore_seed_level.csv", index=False)
    rk = pd.DataFrame(ranks)
    rk.to_csv(OUT / "saratov_auto_mosaic_rank_comparison.csv", index=False)
    met = [c for c in seed.columns if c not in ("seed", "imagery")]
    leg = seed[seed.imagery == "legacy (frozen)"].set_index("seed")[met]
    aut = seed[seed.imagery == "auto_mosaic"].set_index("seed")[met]
    d = aut - leg
    summ = pd.DataFrame({"legacy_mean": leg.mean(), "legacy_sd": leg.std(ddof=1), "auto_mean": aut.mean(),
                         "auto_sd": aut.std(ddof=1), "delta_mean": d.mean(), "delta_sd": d.std(ddof=1),
                         "delta_seed0": d.loc[0], "delta_seed1": d.loc[1], "delta_seed2": d.loc[2]})
    summ.index.name = "metric"
    summ.to_csv(OUT / "saratov_auto_mosaic_rescore_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, s in zip(axes, SEEDS):
        for name, col in (("legacy (frozen)", "#4C72B0"), ("auto_mosaic", "#DD8452")):
            c = curves[(s, name)]
            ax.step(c.pct_aoi_reviewed, c.facility_recall, where="post", color=col, label=name)
        for b in BUDGETS:
            ax.axvline(b, color="0.85", lw=0.8, zorder=0)
        ax.set_title(f"Condition C, seed {s}")
        ax.set_xlabel("% of Saratov AOI reviewed")
        ax.set_xlim(0, 100)
    axes[0].set_ylabel("Saratov facility recall")
    axes[0].legend(loc="lower right")
    fig.suptitle("Saratov facility recall vs review area: legacy z16 2x2 tiles vs auto_mosaic re-extraction (same frozen C checkpoints)")
    fig.tight_layout()
    fig.savefig(PKG.parent / "figures" / "fig_saratov_legacy_vs_auto_mosaic.png", dpi=150)
    print(summ.round(3).to_string())
    print(rk.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "summary", "cache", "score", "report"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    {"extract": lambda: extract(a.workers), "summary": summary, "cache": cache,
     "score": lambda: score(a.device), "report": report}[a.step]()
