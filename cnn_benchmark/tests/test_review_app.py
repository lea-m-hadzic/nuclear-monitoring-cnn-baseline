"""Tests for the local hard-negative review app: single-row atomic CSV edits, backups, validation, HTTP API."""
from __future__ import annotations

import csv
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parents[1] / "external_data"
sys.path.insert(0, str(EXT))

import hard_negative_review_app as app  # noqa: E402
from geometry import REPO_ROOT, load_config  # noqa: E402

HEADER = ["tile_id", "facility_id", "category", "facility_name_original", "source_id", "split", "artifact_flags",
          "review_sheet", "sheet_cell", "png_path", "manual_review_status", "reviewer", "review_notes", "extra"]


def make_csv(tmp_path) -> Path:
    p = tmp_path / "hard_negative_review.csv"
    rows = [HEADER] + [[f"t{i}", f"f{i}", "cat", f'Name, "quoted" {i}', f"src{i}", "train", "", "s.png", str(i),
                        f"img{i}.png", "unreviewed", "", "", "keep 1.50"] for i in range(5)]
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    return p


def test_update_changes_only_one_row_and_backs_up(tmp_path):
    p = make_csv(tmp_path)
    before = p.read_bytes()
    s = app.ReviewStore(p, tmp_path / "bk", reviewer="Lea")
    s.update("t2", "reject_artifact", "point off-site")
    after = p.read_text().splitlines()
    old = before.decode().splitlines()
    assert [i for i, (a, b) in enumerate(zip(old, after)) if a != b] == [3]          # header + row t2 only
    assert after[3].split(",")[-4:] == ["reject_artifact", "Lea", "point off-site", "keep 1.50"]
    backups = list((tmp_path / "bk").glob("hard_negative_review.*.csv"))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    s.update("t3", "confirmed_hard_negative")
    assert len(list((tmp_path / "bk").glob("*.csv"))) == 1                          # one backup per session
    s.update("t2", "unreviewed")                                                     # reset keeps notes
    assert p.read_text().splitlines()[3].split(",")[-4:] == ["unreviewed", "Lea", "point off-site", "keep 1.50"]
    assert not list(tmp_path.glob(".hard_negative_review.*.tmp"))                    # no temp files left


def test_rejects_bad_status_unknown_tile_and_corrupt_file(tmp_path):
    p = make_csv(tmp_path)
    before = p.read_bytes()
    s = app.ReviewStore(p, tmp_path / "bk")
    for args in (("t1", "confirmed"), ("nope", "reject_artifact")):
        with pytest.raises(ValueError):
            s.update(*args)
    assert p.read_bytes() == before and not (tmp_path / "bk").exists()              # nothing written, no backup
    p.write_text(before.decode().replace("unreviewed", "maybe", 1))                  # someone typed a bad status
    with pytest.raises(ValueError, match="invalid statuses"):
        s.update("t3", "reject_ambiguous")


cfg = load_config()
DATA = REPO_ROOT / cfg["dataset"]["dataset_dir"] / "hard_negative_review.csv"


@pytest.mark.skipif(not DATA.exists(), reason="pilot data not present")
def test_http_api_on_a_copy(tmp_path):
    copy = tmp_path / "hard_negative_review.csv"
    copy.write_bytes(DATA.read_bytes())
    srv, store = app.build_server(cfg, 0, "tester", csv_path=copy)
    store.backup_dir = tmp_path / "bk"
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        items = json.loads(urllib.request.urlopen(base + "/api/items").read())
        assert len(items) == len(set(i["tile_id"] for i in items)) >= 68 and items[0]["order"] == 1 and items[0]["priority"]
        assert urllib.request.urlopen(base + f"/img/{items[0]['tile_id']}/zoom").headers["Content-Type"] == "image/jpeg"
        req = urllib.request.Request(base + "/api/label", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"tile_id": items[0]["tile_id"], "status": "reject_artifact"}).encode())
        assert json.loads(urllib.request.urlopen(req).read())["row"]["manual_review_status"] == "reject_artifact"
        bad = urllib.request.Request(base + "/api/label", method="POST", data=json.dumps({"tile_id": "x", "status": "oops"}).encode())
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/img/../../etc/passwd/full")
        assert srv.server_address[0] == "127.0.0.1"
    finally:
        srv.shutdown()
    changed = [a != b for a, b in zip(DATA.read_text().splitlines(), copy.read_text().splitlines())]
    assert sum(changed) == 1                                                         # only that row differs


def test_matched_nearby_is_deterministic_split_preserving_and_unique():
    import numpy as np
    import pandas as pd
    from hard_negative_review_ui import matched_nearby
    rng = np.random.default_rng(1)
    def frame(prefix, n, split, c_mu, w_mu):
        return pd.DataFrame({"tile_id": [f"{prefix}{split}{i}" for i in range(n)], "split": split,
                             "contrast": rng.normal(c_mu, 5, n), "water_frac": np.clip(rng.normal(w_mu, .05, n), 0, 1)})
    pos = pd.concat([frame("p", 60, "train", 38, .10), frame("p", 15, "val", 38, .10)])
    near = pd.concat([frame("n", 200, "train", 28, .02), frame("n", 50, "val", 28, .02)])
    before = near.copy()
    a = matched_nearby(pos, near, {"train": 30, "val": 8}, "s")
    assert a.equals(matched_nearby(pos, near, {"train": 30, "val": 8}, "s"))          # deterministic
    assert not a.tile_id.duplicated().any() and a.split.value_counts().to_dict() == {"train": 30, "val": 8}
    assert set(a[a.split == "val"].tile_id) <= set(near[near.split == "val"].tile_id)  # stays within its split
    assert near.equals(before)                                                         # pool untouched
    assert a.contrast.median() > near.contrast.median()                                # moved toward positives
