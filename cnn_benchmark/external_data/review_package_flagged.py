"""Review package of the 14 QA-flagged external positives for Yanliang. Reads QA outputs only; changes no label.

    python review_package_flagged.py   # -> external_imagery/dataset/review_packages/flagged_training_positives_14/ (git-ignored)

Contents: flagged_training_positives_14.csv (Yanliang's original point coordinates; reviewer columns blank),
contact_sheet.html (self-contained; per row the 475 m QA crop centred on the point, plus the full 1.9 km tile with
the crop outlined), README.md. The CSV row order, row_id and the images are generated from one table, and verify()
re-renders every image from the CSV row and compares it with the image embedded in the sheet.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geometry import REPO_ROOT  # noqa: E402

PKG = HERE.parent
QA = PKG.parent / "results" / "qa" / "positive_review" / "positive_proposed_corrections.csv"
META = REPO_ROOT / "external_imagery" / "dataset" / "external_tiles_metadata_refined.csv"
CACHE = PKG / "cache" / "external_decoded_1280px"
OUT = REPO_ROOT / "external_imagery" / "dataset" / "review_packages" / "flagged_training_positives_14"
FLAG = {"point_likely_offsite": "likely_off_site", "no_secure_perimeter_visible": "no_visible_secure_perimeter"}
COLS = ["row_id", "site_name", "latitude", "longitude", "qa_flag", "qa_note",
        "review_decision", "corrected_latitude", "corrected_longitude", "reviewer_notes"]
CROP_M = 475.0


def table() -> pd.DataFrame:
    qa = pd.read_csv(QA)
    m = pd.read_csv(META, dtype={"latitude_original": str, "longitude_original": str}).set_index("tile_id")
    t = qa.assign(site_name=[m.loc[k, "facility_name_original"] for k in qa.tile_id],
                  latitude=[m.loc[k, "latitude_original"] for k in qa.tile_id],
                  longitude=[m.loc[k, "longitude_original"] for k in qa.tile_id],
                  qa_flag=qa.visual_status.map(FLAG), qa_note=qa.visual_notes)
    t = t.sort_values(["qa_flag", "site_name"], ascending=[True, True]).reset_index(drop=True)   # likely_off_site first
    t["row_id"] = np.arange(1, len(t) + 1)
    for c in ("review_decision", "corrected_latitude", "corrected_longitude", "reviewer_notes"):
        t[c] = ""
    return t


def render(tile_id: str, row_id: int, site: str) -> tuple[bytes, bytes]:
    """(QA crop JPEG, full-tile JPEG) for one row; deterministic so it can be re-rendered for verification."""
    m = pd.read_csv(META).set_index("tile_id").loc[tile_id]
    arr = np.load(CACHE / f"{tile_id}.npy")                      # 1280 px decoded tile (same source as the QA review)
    mpp = m.footprint_width_km * 1000 / 1280
    half = int(round(CROP_M / 2 / mpp))
    cx, cy = int(m.fence_px_col / 2), int(m.fence_px_row / 2)    # Yanliang's point in the 1280 px frame
    pad = np.pad(arr, ((half, half), (half, half), (0, 0)), constant_values=255)
    crop = Image.fromarray(pad[cy:cy + 2 * half, cx:cx + 2 * half]).resize((420, 420), Image.LANCZOS)
    full = Image.fromarray(arr).resize((420, 420), Image.LANCZOS)
    s = 420 / 1280
    out = []
    for im, (px, py), box in ((crop, (210, 210), None),
                              (full, (cx * s, cy * s), ((cx - half) * s, (cy - half) * s, (cx + half) * s, (cy + half) * s))):
        canvas = Image.new("RGB", (420, 446), "white")
        canvas.paste(im, (0, 26))
        d = ImageDraw.Draw(canvas)
        d.text((4, 6), f"#{row_id}  {site[:58]}", fill="black")
        d.ellipse([px - 9, py + 26 - 9, px + 9, py + 26 + 9], outline=(255, 0, 255), width=3)
        if box:
            d.rectangle([box[0], box[1] + 26, box[2], box[3] + 26], outline=(255, 255, 0), width=2)
        b = io.BytesIO(); canvas.save(b, "JPEG", quality=88); out.append(b.getvalue())
    return out[0], out[1]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    t = table()
    t[COLS].to_csv(OUT / "flagged_training_positives_14.csv", index=False)
    rows = []
    for r in t.itertuples():
        crop, full = render(r.tile_id, r.row_id, r.site_name)
        b64 = lambda x: base64.b64encode(x).decode()
        rows.append(f"""
<section class="row" id="row{r.row_id}" data-row-id="{r.row_id}" data-site="{html.escape(r.site_name)}">
  <h2>#{r.row_id} · {html.escape(r.site_name)}</h2>
  <p><span class="flag {r.qa_flag}">{r.qa_flag}</span> &nbsp; {html.escape(r.latitude)}, {html.escape(r.longitude)}</p>
  <p class="note">QA note: {html.escape(r.qa_note)}</p>
  <div class="imgs">
    <figure><img data-kind="crop" alt="#{r.row_id} QA crop" src="data:image/jpeg;base64,{b64(crop)}"><figcaption>QA crop: {CROP_M:.0f} m × {CROP_M:.0f} m centred on the point (magenta)</figcaption></figure>
    <figure><img data-kind="full" alt="#{r.row_id} full tile" src="data:image/jpeg;base64,{b64(full)}"><figcaption>Full tile, about 1.9 km wide; yellow box = QA crop</figcaption></figure>
  </div>
</section>""")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Flagged training positives: review sheet</title>
<style>
 body {{ font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1d2329; margin: 0 auto; max-width: 900px; padding: 24px 16px; }}
 h1 {{ font-size: 20px; margin: 0 0 6px; }} h2 {{ font-size: 16px; margin: 0 0 4px; }}
 .row {{ border-top: 1px solid #d9dee3; padding: 18px 0; break-inside: avoid; }}
 .imgs {{ display: flex; gap: 12px; flex-wrap: wrap; }} figure {{ margin: 0; }} img {{ width: 420px; max-width: 100%; display: block; }}
 figcaption {{ font-size: 12px; color: #66707a; }} .note {{ color: #3b444d; margin: 2px 0 8px; }}
 .flag {{ font: 12px ui-monospace, monospace; padding: 2px 6px; border-radius: 4px; }}
 .likely_off_site {{ background: #fde7e7; color: #8a1c1c; }} .no_visible_secure_perimeter {{ background: #fff4d6; color: #7a5a00; }}
</style></head><body>
<h1>Flagged training positives (14): review sheet</h1>
<p>Each numbered row matches <code>row_id</code> in <code>flagged_training_positives_14.csv</code>. The magenta circle marks the
submitted coordinate. Please record a decision for each row in the CSV: <code>keep</code>, <code>adjust_coordinate</code> or <code>exclude</code>.</p>
<p>Imagery: Google Maps Static API satellite tiles (© Google and its imagery providers), as used in training.</p>
{''.join(rows)}
</body></html>"""
    (OUT / "contact_sheet.html").write_text(page)
    (OUT / "README.md").write_text(README)
    return t


README = """# Review: 14 flagged training positives

This package covers only the **14 training positives flagged during label QA**. It contains no other training data,
negatives or benchmark material. **No labels have been changed**: the flags are questions for you, not decisions.

## Files

- `flagged_training_positives_14.csv`: one row per flagged point, with the coordinate as originally submitted.
- `contact_sheet.html`: open it in any browser. Each numbered row (matching `row_id`) shows the site name, the flag,
  the coordinate, the 475 m crop used in QA (point marked in magenta) and a wider view of the whole 1.9 km tile.

## Flags

- `likely_off_site` (4): the facility is visible in the tile, but the point seems to lie outside it (forest, road, shoreline).
- `no_visible_secure_perimeter` (10): the site is probably correct, but no clear layered or secure fence is visible near
  the point at this resolution (urban, campus or business-park settings). The model is trained to recognise that fence
  signature, so these may not suit this task even if the site identity is right.

## What to fill in (CSV)

- `review_decision`: one of `keep`, `adjust_coordinate` or `exclude`.
- `corrected_latitude`, `corrected_longitude`: only for `adjust_coordinate`. Use decimal degrees (WGS84), ideally a point
  on the secure perimeter.
- `reviewer_notes`: anything else, optional.

Leave the other columns unchanged. Please return the CSV; nothing is changed until your answers are in.
"""


def verify() -> None:
    t = pd.read_csv(OUT / "flagged_training_positives_14.csv", dtype=str, keep_default_na=False)
    assert len(t) == 14, len(t)
    assert list(t.columns) == COLS
    assert t.qa_flag.value_counts().to_dict() == {"no_visible_secure_perimeter": 10, "likely_off_site": 4}
    assert (t[["review_decision", "corrected_latitude", "corrected_longitude", "reviewer_notes"]] == "").all().all()
    assert t.row_id.tolist() == [str(i) for i in range(1, 15)]
    page = (OUT / "contact_sheet.html").read_text()
    secs = re.findall(r'<section class="row" id="row(\d+)" data-row-id="(\d+)" data-site="([^"]*)">(.*?)</section>', page, re.S)
    assert len(secs) == 14
    full = table().set_index("row_id")
    for (rid, rid2, site, body), r in zip(secs, t.itertuples()):
        assert rid == rid2 == r.row_id and html.unescape(site) == r.site_name, (rid, site, r.site_name)
        assert r.qa_flag in body and html.escape(r.latitude) in body and html.escape(r.longitude) in body
        imgs = dict(re.findall(r'data-kind="(crop|full)"[^>]*src="data:image/jpeg;base64,([^"]+)"', body))
        crop, fullimg = render(full.loc[int(rid), "tile_id"], int(rid), r.site_name)
        assert base64.b64decode(imgs["crop"]) == crop and base64.b64decode(imgs["full"]) == fullimg, f"image mismatch row {rid}"
    print("verified: 14 rows, columns and flags as specified, reviewer columns blank, row_id 1-14;")
    print("          every sheet row matches its CSV row (row_id, site, flag, coordinate) and both images re-render identically")
    print("package sha256:", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in sorted(OUT.iterdir())})


if __name__ == "__main__":
    build()
    verify()
