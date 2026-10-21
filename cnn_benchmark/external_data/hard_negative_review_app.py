"""Tiny local review app for the hard-negative review pool (pilot + scale-up) (Python stdlib only; binds 127.0.0.1).

    python hard_negative_review_app.py [--port 8765] [--reviewer Lea]
    -> open http://127.0.0.1:8765/

`hard_negative_review.csv` stays the single source of truth; no other store is used.
Each click rewrites exactly one row's `manual_review_status` (and `reviewer`, plus `review_notes` when notes are
saved) with the csv module (byte-identical round trip verified), via temp file + fsync + os.replace, keeping row
order and every other cell. Before the first write of a session a timestamped backup is made. Allowed statuses
are validated before every write. Imagery and all other files are only read. `validate` / `diagnose` live in
hard_negative_review_ui.py and are never run from here.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from geometry import REPO_ROOT, load_config
from hard_negative_pilot import HN_STATUSES
from hard_negative_review_ui import GUIDE, PRIORITY_GROUPS, all_plans, ensure_images, notes_map, ordered, paths

STATUS_COL, REVIEWER_COL, NOTES_COL = "manual_review_status", "reviewer", "review_notes"


class ReviewStore:
    """All CSV access. Reads fresh on every request so external edits are never overwritten blindly."""

    def __init__(self, csv_path: Path, backup_dir: Path, reviewer: str = ""):
        self.path, self.backup_dir, self.reviewer = csv_path, backup_dir, reviewer
        self.backup_made: Path | None = None

    def read(self) -> tuple[list[str], list[list[str]]]:
        with open(self.path, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        return rows[0], rows[1:]

    def _validate(self, header: list[str], rows: list[list[str]]) -> None:
        si = header.index(STATUS_COL)
        bad = {r[si] for r in rows} - set(HN_STATUSES)
        if bad:
            raise ValueError(f"refusing to write: invalid statuses {sorted(bad)}")
        ids = [r[0] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("refusing to write: duplicate tile_id rows")

    def update(self, tile_id: str, status: str | None = None, notes: str | None = None) -> dict:
        if status is not None and status not in HN_STATUSES:
            raise ValueError(f"invalid status {status!r}; allowed {HN_STATUSES}")
        header, rows = self.read()
        for c in (STATUS_COL, REVIEWER_COL, NOTES_COL):
            if c not in header:
                raise ValueError(f"column {c} missing from {self.path.name}")
        hits = [i for i, r in enumerate(rows) if r[0] == tile_id]
        if len(hits) != 1:
            raise ValueError(f"tile_id {tile_id!r} matched {len(hits)} rows")
        row = rows[hits[0]]
        if status is not None:
            row[header.index(STATUS_COL)] = status
            if self.reviewer:
                row[header.index(REVIEWER_COL)] = self.reviewer
        if notes is not None:
            row[header.index(NOTES_COL)] = notes.replace("\r", " ").replace("\n", " ").strip()
        self._validate(header, rows)
        if self.backup_made is None:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self.backup_made = self.backup_dir / f"hard_negative_review.{time.strftime('%Y%m%dT%H%M%S')}.csv"
            shutil.copy2(self.path, self.backup_made)
        buf = io.StringIO(newline="")
        csv.writer(buf, lineterminator="\n").writerows([header] + rows)
        fd, tmp = tempfile.mkstemp(prefix=".hard_negative_review.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(buf.getvalue())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return dict(zip(header, row))


def items(store: ReviewStore, cfg) -> list[dict]:
    d, _, _ = paths(cfg)
    raw = pd.read_csv(store.path, dtype=str, keep_default_na=False)
    rv = ordered(raw, d)
    plan = all_plans(d).set_index("tile_id")
    notes = notes_map(d, raw)
    out = []
    for r in rv.itertuples():
        p = plan.loc[r.tile_id]
        out.append({"tile_id": r.tile_id, "order": int(r.review_order), "group": r.review_group,
                    "priority": r.review_group.startswith(PRIORITY_GROUPS), "name": r.facility_name_original, "category": r.category,
                    "category_detail": p.get("category_detail", ""), "source": p.get("source", "").split(",")[0],
                    "source_id": r.source_id, "split": r.split, "artifact_flags": r.artifact_flags,
                    "prescreen": notes.get(r.tile_id, ""), "status": r.manual_review_status, "notes": r.review_notes,
                    "reviewer": r.reviewer})
    return out


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Hard-negative review</title><style>
body{font-family:-apple-system,sans-serif;margin:0;background:#fafafa;color:#222}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 14px;z-index:2}
#bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}#prog{font-weight:600}
.cnt span{margin-right:10px;font-size:13px}main{padding:12px 14px}
.imgs{display:flex;gap:10px}.imgs img{width:min(46vw,640px);height:min(46vw,640px);border:1px solid #aaa;background:#eee}
.meta{margin:8px 0;font-size:15px;line-height:1.55}.st{font-weight:700;padding:1px 6px;border-radius:4px}
.unreviewed{background:#fff3cd}.confirmed_hard_negative{background:#d1f2dc}.reject_target_signature,.reject_artifact,.reject_ambiguous{background:#f8d7da}
.btns button{font-size:15px;margin:4px 6px 4px 0;padding:7px 11px;cursor:pointer}textarea{width:min(92vw,1290px);height:44px}
#msg{font-size:13px;margin-left:8px}.err{color:#c62828}.ok{color:#1a7f37}details{font-size:13px}
</style></head><body><header><div id="bar"><span id="prog"></span>
<select id="filt"><option value="all">all</option><option value="unreviewed">unreviewed</option><option value="priority">priority (a-b)</option>
__STATUS_OPTIONS__</select>
<button onclick="go(-1)">&larr; Prev</button><button onclick="go(1)">Next &rarr;</button><button onclick="nextUnrev()">Next unreviewed (n)</button>
<span id="pos"></span><span id="msg"></span></div><div class="cnt" id="cnt"></div>
<details><summary>Reviewer guide &amp; keys</summary><ul>__GUIDE__</ul>
Keys: 1 confirm &middot; 2 target signature &middot; 3 artifact &middot; 4 ambiguous &middot; 0 reset &middot; &larr;/&rarr; or j/k prev/next &middot; n next unreviewed. Notes are saved with the button (or on any label click).</details></header>
<main><div class="imgs"><a id="fulllink" target="_blank"><img id="full" title="full tile (click: 2560 px)"></a><img id="zoom" title="~475 m zoom on the source point"></div>
<div class="meta" id="meta"></div><div class="btns">
<button onclick="label('confirmed_hard_negative')">1 Confirm hard negative</button>
<button onclick="label('reject_target_signature')">2 Reject: target signature</button>
<button onclick="label('reject_artifact')">3 Reject: artifact</button>
<button onclick="label('reject_ambiguous')">4 Reject: ambiguous</button>
<button onclick="label('unreviewed')">0 Reset to unreviewed</button></div>
<textarea id="notes" placeholder="optional reviewer notes"></textarea><br><button onclick="saveNotes()">Save note</button></main>
<script>
let all=[],view=[],i=0;const S=__STATUSES__;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function load(){const r=await fetch('/api/items');all=await r.json();applyFilter(true)}
function applyFilter(keep){const f=document.getElementById('filt').value,cur=view[i]&&view[i].tile_id;
 view=all.filter(x=>f==='all'||(f==='priority'?x.priority:x.status===f));
 i=keep&&cur?Math.max(0,view.findIndex(x=>x.tile_id===cur)):0;render()}
function render(){const n=all.length,rev=all.filter(x=>x.status!=='unreviewed').length;
 document.getElementById('prog').textContent=`${rev} / ${n} reviewed`;
 document.getElementById('cnt').innerHTML=S.map(s=>`<span>${s}: ${all.filter(x=>x.status===s).length}</span>`).join('');
 if(!view.length){document.getElementById('meta').innerHTML='<i>no items in this filter</i>';document.getElementById('pos').textContent='';return}
 i=Math.min(Math.max(i,0),view.length-1);const x=view[i];document.getElementById('pos').textContent=`item ${i+1} of ${view.length} (review #${x.order})`;
 document.getElementById('full').src=`/img/${x.tile_id}/full`;document.getElementById('zoom').src=`/img/${x.tile_id}/zoom`;
 document.getElementById('fulllink').href=`/img/${x.tile_id}/png`;
 document.getElementById('meta').innerHTML=`<b>#${x.order} ${esc(x.name)}</b> &middot; ${esc(x.group)}<br>
 category: ${esc(x.category)}${x.category_detail?' ('+esc(x.category_detail)+')':''} &middot; source: ${esc(x.source)} &middot; ${esc(x.source_id)}<br>
 tile: <code>${esc(x.tile_id)}</code> &middot; split ${esc(x.split)} &middot; artifact flags: ${esc(x.artifact_flags||'none')}<br>
 pre-screen: <i>${esc(x.prescreen)}</i><br>status: <span class="st ${esc(x.status)}">${esc(x.status)}</span>${x.reviewer?' &middot; reviewer '+esc(x.reviewer):''}`;
 document.getElementById('notes').value=x.notes||''}
function go(d){i+=d;render()}
function nextUnrev(){const start=all.findIndex(x=>x.tile_id===(view[i]||{}).tile_id);
 for(let k=1;k<=all.length;k++){const x=all[(start+k)%all.length];if(x.status==='unreviewed'){document.getElementById('filt').value='all';view=all;i=all.indexOf(x);render();return}}
 msg('all reviewed',true)}
function msg(t,ok){const m=document.getElementById('msg');m.textContent=t;m.className=ok?'ok':'err'}
async function post(body){const x=view[i];body.tile_id=x.tile_id;
 const r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json();if(!r.ok){msg('NOT saved: '+j.error,false);return null}
 const t=all.find(y=>y.tile_id===x.tile_id);t.status=j.row.manual_review_status;t.notes=j.row.review_notes;t.reviewer=j.row.reviewer;msg('saved '+x.tile_id,true);return t}
async function label(s){const t=await post({status:s,notes:document.getElementById('notes').value});if(t){const f=document.getElementById('filt').value;
 if(f!=='all'){applyFilter(true)}else render()}}
async function saveNotes(){if(await post({notes:document.getElementById('notes').value}))render()}
document.addEventListener('change',e=>{if(e.target.id==='filt')applyFilter(false)});
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||e.metaKey||e.ctrlKey)return;
 const k={'1':'confirmed_hard_negative','2':'reject_target_signature','3':'reject_artifact','4':'reject_ambiguous','0':'unreviewed'}[e.key];
 if(k){label(k);e.preventDefault()}else if(e.key==='ArrowRight'||e.key==='k')go(1);else if(e.key==='ArrowLeft'||e.key==='j')go(-1);else if(e.key==='n')nextUnrev()});
load();
</script></body></html>"""


def make_handler(store: ReviewStore, cfg, img_dir: Path, tiles: dict[str, Path]):
    page = PAGE.replace("__STATUSES__", json.dumps(HN_STATUSES)) \
        .replace("__STATUS_OPTIONS__", "".join(f'<option value="{s}">{s}</option>' for s in HN_STATUSES)) \
        .replace("__GUIDE__", "".join(f"<li>{g}</li>" for g in GUIDE))

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def log_message(self, fmt, *args):        # quiet; only label writes are printed
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if u.path == "/api/items":
                return self._json(200, items(store, cfg))
            parts = u.path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "img" and parts[1] in tiles:    # only known tiles: no path traversal
                if parts[2] in ("full", "zoom"):
                    f = img_dir / f"{parts[1]}_{parts[2]}.jpg"
                    return self._send(200, f.read_bytes(), "image/jpeg")
                if parts[2] == "png":
                    return self._send(200, tiles[parts[1]].read_bytes(), "image/png")
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/api/label":
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                row = store.update(str(body.get("tile_id", "")), body.get("status"), body.get("notes"))
                print(f"{time.strftime('%H:%M:%S')} {row['tile_id']} -> {row[STATUS_COL]}"
                      + (f" (backup {store.backup_made.name})" if store.backup_made else ""), flush=True)
                return self._json(200, {"row": row})
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
    return H


def build_server(cfg, port: int, reviewer: str, csv_path: Path | None = None) -> tuple[HTTPServer, ReviewStore]:
    d, path, rdir = paths(cfg)
    path = csv_path or path
    store = ReviewStore(path, d / "review_backups", reviewer)
    rv = pd.read_csv(path, dtype=str, keep_default_na=False)
    img_dir = ensure_images(cfg, rv)
    tiles = {t: REPO_ROOT / p for t, p in zip(rv.tile_id, rv.png_path)}
    return HTTPServer(("127.0.0.1", port), make_handler(store, cfg, img_dir, tiles)), store


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reviewer", default="", help="written to the `reviewer` column on each label")
    ap.add_argument("--csv", type=Path, default=None, help="review a COPY instead (testing); default: the real review CSV")
    a = ap.parse_args()
    srv, store = build_server(load_config(), a.port, a.reviewer, csv_path=a.csv)
    if a.csv:
        store.backup_dir = a.csv.parent / "review_backups"
    print(f"Hard-negative review app: http://127.0.0.1:{a.port}/   (CSV: {store.path}; Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
