"""Freeze record for the main ablation matrix: checksums of every input that defines the experiment, plus git state.

    python freeze_ablation.py           # writes outputs/ablation/freeze.json (refuses to overwrite)
    python freeze_ablation.py --check   # recompute and compare with the frozen record

train_ablation.py (non-dry runs) refuses to start if the current inputs differ from freeze.json, and copies the
freeze record plus the git state at launch into each run's run_config.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
FREEZE = PKG / "outputs" / "ablation" / "freeze.json"
FREEZE_ROB = PKG / "outputs" / "ablation" / "freeze_robustness.json"   # robustness matrix; freeze.json is never rewritten
CONFIG = HERE / "ablation_config.yaml"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=PKG, capture_output=True, text=True, check=True).stdout
    diff = run("diff", "HEAD")
    return {"commit": run("rev-parse", "HEAD").strip(), "branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "tracked_changes": run("status", "--porcelain", "--untracked-files=no").splitlines(),
            "diff_vs_head_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def checksums() -> dict:
    from watermark_geometry import template
    ac = yaml.safe_load(CONFIG.read_text())
    man = PKG / ac["data"]["manifests_dir"]
    return {"ablation_config.yaml": _sha(CONFIG),
            **{f"manifest_{c}": _sha(man / f"{c}.csv") for c in ac["conditions"]},
            "shared_val.csv": _sha(PKG / ac["data"]["shared_val"]),
            "watermark_template": template()["sha256"],
            "watermark_geometry.csv": _sha(PKG / ac["data"]["mask_geometry"])}


def check() -> dict:
    """Frozen record if the current inputs match it; SystemExit otherwise."""
    if not FREEZE.exists():
        raise SystemExit(f"{FREEZE} missing: run freeze_ablation.py before the main runs")
    frozen = json.loads(FREEZE.read_text())
    now = checksums()
    bad = {k: (frozen["checksums"].get(k), v) for k, v in now.items() if frozen["checksums"].get(k) != v}
    if bad:
        raise SystemExit(f"experiment inputs changed since the freeze: {sorted(bad)}")
    return frozen


def checksums_robustness() -> dict:
    ac = yaml.safe_load(CONFIG.read_text())
    man = PKG / ac["data"]["manifests_dir"]
    c = checksums()
    c["manifest_C_matched"] = _sha(man / "C_matched.csv")
    c["matrix"] = "C_matched seeds 0-2 (masked); A_unmasked, C_unmasked seeds 0-2 (--no-mask only); recipe otherwise frozen"
    return c


def check_robustness() -> dict:
    if not FREEZE_ROB.exists():
        raise SystemExit(f"{FREEZE_ROB} missing: run freeze_ablation.py --robustness before the robustness runs")
    frozen = json.loads(FREEZE_ROB.read_text())
    now = checksums_robustness()
    bad = sorted(k for k, v in now.items() if frozen["checksums"].get(k) != v)
    if bad:
        raise SystemExit(f"robustness inputs changed since the freeze: {bad}")
    return frozen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--robustness", action="store_true", help="write / check freeze_robustness.json")
    a = ap.parse_args(argv)
    if a.robustness:
        if a.check:
            print(json.dumps(check_robustness(), indent=1))
            return
        if FREEZE_ROB.exists():
            raise SystemExit(f"{FREEZE_ROB} already exists; the freeze is not overwritten")
        rec = {"checksums": checksums_robustness(), "git_at_freeze": git_state()}
        FREEZE_ROB.write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1))
        return
    if a.check:
        print(json.dumps(check(), indent=1))
        return
    if FREEZE.exists():
        raise SystemExit(f"{FREEZE} already exists; the freeze is not overwritten")
    rec = {"checksums": checksums(), "git_at_freeze": git_state()}
    FREEZE.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
