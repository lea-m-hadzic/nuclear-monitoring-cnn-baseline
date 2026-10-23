"""Aggregate the robustness runs as two backend-matched comparisons (never mixing backends in one table).

    python aggregate_robustness.py     # -> results/robustness/{mps_mask_sensitivity,cuda_matching_control}/

  mps_mask_sensitivity    Apple MPS: frozen A, C (mask on; outputs/ablation/runs/, read only) vs
                          A_unmasked, C_unmasked (mask off; outputs/ablation/runs_robustness_mps/)
  cuda_matching_control   CUDA, A100 80GB (bend GPU 1): C vs C_matched (both mask on; outputs/ablation/runs_robustness_cuda/)

The frozen A/B/C main ablation keeps its own tables (results/main/abc_tables/, aggregate_ablation.py). Each comparison
is produced by aggregate_ablation.main() unchanged, over a view directory of symlinks, so every metric uses exactly
the same code as the main ablation. Condition labels carry the backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import aggregate_ablation as aa

PKG = Path(__file__).resolve().parents[1]
ABL = PKG / "outputs" / "ablation"
OUT = PKG.parent / "results" / "robustness"
COMPARISONS = {
    "mps_mask_sensitivity": {
        "A [MPS, mask on]": ABL / "runs" / "A", "C [MPS, mask on]": ABL / "runs" / "C",
        "A [MPS, mask off]": ABL / "runs_robustness_mps" / "A_unmasked",
        "C [MPS, mask off]": ABL / "runs_robustness_mps" / "C_unmasked"},
    "cuda_matching_control": {
        "C [CUDA]": ABL / "runs_robustness_cuda" / "C",
        "C_matched [CUDA]": ABL / "runs_robustness_cuda" / "C_matched"},
}
COLORS = ["#1f77b4", "#2ca02c", "#17becf", "#bcbd22"]


def run_one(name: str, conds: dict[str, Path]) -> None:
    out = OUT / name
    view = out / "_runs_view"
    view.mkdir(parents=True, exist_ok=True)
    backends = {}
    for label, src in conds.items():
        link = view / label
        if link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve(), target_is_directory=True)
        for s in aa.SEEDS:
            rc = src / f"seed{s}" / "run_config.json"
            if rc.exists():
                m = json.loads(rc.read_text())
                backends[f"{label}/seed{s}"] = {"device": m.get("device"), "backend": m.get("backend"),
                                               "watermark_mask": m.get("watermark_mask"), "git": (m.get("git_at_launch") or {}).get("commit")}
    (out / "backends.json").write_text(json.dumps(backends, indent=1))
    aa.RUNS, aa.OUT, aa.CONDS = view, out, list(conds)
    aa.COLORS.update(dict(zip(conds, COLORS)))
    aa.LABEL.update({c: c for c in conds})
    aa.main()


def main():
    for name, conds in COMPARISONS.items():
        print(f"\n######## {name}")
        run_one(name, conds)


if __name__ == "__main__":
    main()
