"""Resolve z16 parent tiles to full-resolution images and check cache integrity.

The cache shared by Qiqi arrives as Google-Drive download parts: several extracted folders named
"<scan_dir>", "<scan_dir> 2", ... each with a tiles/ subfolder, plus the original zip parts (in any
subfolder, e.g. zips/). Files are matched on (scan_dir, exact original filename) taken from the VLM CSV
`image_path`, i.e. row, col AND centre coordinates must agree; row/col alone is never trusted because
the shared folders also contain tiles from other grids. Extracted files are preferred; tiles present only
inside a zip are read directly from the zip (nothing is copied or renamed).

  python src/image_index.py                 # resolve + header check (dimensions) for every parent tile
  python src/image_index.py --deep          # + full PNG decode of every resolved image (slow, ~50 GB read)

Writes outputs/image_index.csv (local, git-ignored) and outputs/image_integrity_report.md.
Exit code 1 if any benchmark parent tile is unresolved, ambiguous, or unreadable.
"""
from __future__ import annotations

import argparse
import functools
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from common import OUTPUT_DIR, REPO_ROOT, load_config

SUFFIX_RE = re.compile(r"^(.*?)(?: \d+)?$")             # "scan_dir 3" -> "scan_dir"
NAME_RE = re.compile(r"^region_r(\d+)_c(\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_tile([\d.]+)km(\(\d+\))?\.png$")
ZIP_SEP = "::"
EXPECTED_PX = 2560


def image_root_from(arg: str | None = None) -> Path:
    """CLI arg > $CNN_BENCH_IMAGE_ROOT > config image_root; relative paths are relative to the repo root."""
    val = arg or os.environ.get("CNN_BENCH_IMAGE_ROOT") or load_config().get("image_root")
    if not val:
        raise FileNotFoundError("image_root not configured (configs/default.yaml, $CNN_BENCH_IMAGE_ROOT, or --image-root)")
    p = Path(val).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


@functools.lru_cache(maxsize=4)
def scan_cache(root: str) -> pd.DataFrame:
    """Every PNG under root: extracted (<dir>/tiles/*.png) and inside zip files. Lightweight (no decoding)."""
    root = Path(root)
    rows = []
    for f in root.glob("*/tiles/*.png"):
        rows.append(dict(scan_dir=SUFFIX_RE.match(f.parent.parent.name).group(1), filename=f.name, kind="file",
                         locator=str(f.relative_to(root)), bytes=f.stat().st_size))
    for z in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for i in zf.infolist():
                parts = i.filename.split("/")
                if i.filename.endswith(".png") and len(parts) >= 3 and parts[-2] == "tiles":
                    rows.append(dict(scan_dir=parts[0], filename=parts[-1], kind="zip",
                                     locator=f"{z.relative_to(root)}{ZIP_SEP}{i.filename}", bytes=i.file_size))
    return pd.DataFrame(rows, columns=["scan_dir", "filename", "kind", "locator", "bytes"])


def image_sources() -> dict:
    """Per-AOI canonical sources from config `image_sources` (root relative to the repo root, or absolute)."""
    out = {}
    for region, spec in (load_config().get("image_sources") or {}).items():
        p = Path(spec["root"]).expanduser()
        out[region] = dict(spec, root=p if p.is_absolute() else REPO_ROOT / p)
    return out


def effective_cache(parent_table: pd.DataFrame, root: Path) -> pd.DataFrame:
    """All candidate images, with canonical per-AOI sources swapped in.

    For an AOI in `image_sources`, images under its canonical root are matched under the benchmark scan_dir
    (the source's own folder name, e.g. "<scan>_reupload", is an alias); with `exclusive: true` the copies of
    that AOI's scan_dir under image_root are ignored. Locators stay relative to image_root (canonical sources
    outside it get a "../" relative path), so every consumer keeps using `image_root / locator`.
    """
    cache = scan_cache(str(root)).assign(source="image_root")
    scans = parent_table.assign(scan_dir=parent_table.source_image_relpath.str.split("/").str[0]) \
        .groupby("region").scan_dir.first().to_dict()
    parts = [cache]
    for region, spec in image_sources().items():
        bench_scan = scans.get(region)
        if bench_scan is None:
            continue
        if not Path(spec["root"]).is_dir():
            raise FileNotFoundError(f"canonical {region} image source not found: {spec['root']}")
        if spec.get("exclusive", True):
            parts[0] = parts[0][parts[0].scan_dir != bench_scan]
        c = scan_cache(str(spec["root"]))
        c = c[c.scan_dir == spec.get("scan_dir", bench_scan)].copy()
        c["scan_dir"] = bench_scan

        def rel(loc, base=Path(spec["root"])):
            path, sep, member = loc.partition(ZIP_SEP)
            return os.path.relpath(base / path, root) + sep + member
        c["locator"] = c.locator.map(rel)
        c["source"] = f"image_sources.{region}"
        parts.append(c)
    return pd.concat(parts, ignore_index=True)


def resolve(parent_table: pd.DataFrame, root: Path) -> pd.DataFrame:
    """One row per parent tile: status in {ok, missing, ambiguous}, chosen locator, candidate counts."""
    cache = effective_cache(parent_table, root)
    want = parent_table[["region", "tile_id", "source_image_relpath"]].copy()
    want["scan_dir"] = want.source_image_relpath.str.split("/").str[0]
    want["filename"] = want.source_image_relpath.str.split("/").str[-1]
    m = want.merge(cache, on=["scan_dir", "filename"], how="left")
    out = []
    for (region, tile), g in m.groupby(["region", "tile_id"], sort=False):
        files, zips = g[g.kind == "file"], g[g.kind == "zip"]
        status, note = "ok", ""
        if files.empty and zips.empty:
            status = "missing"
        elif len(files) > 1 or len(zips) > 1:
            status = "ambiguous"
            note = f"{len(files)} extracted / {len(zips)} zipped copies"
        if len(files) and len(zips) and files.bytes.iloc[0] != zips.bytes.iloc[0]:
            status, note = "ambiguous", "extracted file size differs from zip member (truncated extraction?)"
        chosen = files.iloc[0] if len(files) else (zips.iloc[0] if len(zips) else None)
        out.append(dict(region=region, tile_id=tile, filename=g.filename.iloc[0], scan_dir=g.scan_dir.iloc[0],
                        status=status, kind=None if chosen is None else chosen.kind,
                        source=None if chosen is None else chosen.source,
                        locator=None if chosen is None else chosen.locator,
                        bytes=None if chosen is None else int(chosen.bytes),
                        n_extracted=len(files), n_zipped=len(zips), note=note))
    return pd.DataFrame(out)


def orphans(parent_table: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Images in benchmark scan dirs that map to no parent tile (other grids, duplicate downloads)."""
    cache = effective_cache(parent_table, root)
    scans = set(parent_table.source_image_relpath.str.split("/").str[0])
    expected = set(parent_table.source_image_relpath.str.split("/").str[-1])
    o = cache[cache.scan_dir.isin(scans) & ~cache.filename.isin(expected)].drop_duplicates(["scan_dir", "filename"]).copy()
    m = o.filename.str.extract(NAME_RE)
    o["lat"] = pd.to_numeric(m[2]); o["lon"] = pd.to_numeric(m[3])
    o["reason"] = m[5].notna().map({True: "duplicate_download_copy", False: "not_a_benchmark_parent (other grid)"})
    return o


# ---------------------------------------------------------------------------------------------- I/O

_ZIPS: dict = {}


def open_image(locator: str, root: Path) -> Image.Image:
    """Open an image from an extracted path or a 'zipfile::member' locator (zip handles cached per process)."""
    if ZIP_SEP in locator:
        zpath, member = locator.split(ZIP_SEP, 1)
        key = (os.getpid(), str(root / zpath))
        if key not in _ZIPS:
            _ZIPS[key] = zipfile.ZipFile(root / zpath)
        with _ZIPS[key].open(member) as fh:
            im = Image.open(fh)
            im.load()
            return im
    return Image.open(root / locator)


def _header(args):
    locator, root, deep = args
    try:
        im = open_image(locator, Path(root))
        size, mode = im.size, im.mode
        if deep:
            im.convert("RGB").load()
        return size[0], size[1], mode, ""
    except Exception as e:  # noqa: BLE001 - report any unreadable file
        return None, None, None, f"{type(e).__name__}: {e}"


def check_images(index: pd.DataFrame, root: Path, deep: bool = False, workers: int = 8) -> pd.DataFrame:
    ok = index[index.status != "missing"]
    jobs = [(loc, str(root), deep) for loc in ok.locator]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_header, jobs, chunksize=16))
    idx = index.copy()
    idx.loc[ok.index, ["width", "height", "mode", "read_error"]] = pd.DataFrame(res, index=ok.index,
                                                                               columns=["width", "height", "mode", "read_error"])
    return idx


def write_report(index: pd.DataFrame, orph: pd.DataFrame, parent_table: pd.DataFrame, root: Path, deep: bool, path: Path):
    cfg = load_config()
    lines = ["# Full-resolution image cache integrity report", "",
             f"Image root: `{root.relative_to(REPO_ROOT) if root.is_relative_to(REPO_ROOT) else root}` "
             f"(check level: {'header + full decode' if deep else 'header'})", "",
             "| AOI | parents | resolved | from extracted files | from zip only | missing | ambiguous | unreadable | dimensions | m/px (1.9 km / px) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for region in cfg["regions"]:
        g = index[index.region == region]
        dims = g.dropna(subset=["width"]).groupby(["width", "height"]).size()
        dim_s = ", ".join(f"{int(w)}x{int(h)} ({n})" for (w, h), n in dims.items()) or "-"
        mpp = ", ".join(f"{1900 / w:.3f}" for (w, _), _ in dims.items()) or "-"
        lines.append(f"| {region} | {len(g)} | {(g.status == 'ok').sum()} | {(g.kind == 'file').sum()} | "
                     f"{((g.kind == 'zip') & (g.status == 'ok')).sum()} | {(g.status == 'missing').sum()} | "
                     f"{(g.status == 'ambiguous').sum()} | {int(g.read_error.fillna('').ne('').sum())} | {dim_s} | {mpp} |")
    lines += ["", "## Orphan images in benchmark scan folders (not used)", ""]
    if orph.empty:
        lines.append("None.")
    for (scan, reason), g in orph.groupby(["scan_dir", "reason"]):
        lines.append(f"- `{scan}`: {len(g)} x {reason}; lat {g.lat.min():.3f}..{g.lat.max():.3f}, "
                     f"e.g. `{g.filename.iloc[0]}`")
    miss = index[index.status == "missing"]
    if len(miss):
        lines += ["", "## Missing parent tiles (training must not start)", ""]
        for region, g in miss.groupby("region"):
            pt = parent_table[(parent_table.region == region) & parent_table.tile_id.isin(g.tile_id)]
            lines.append(f"- **{region}: {len(g)} missing** (rows {pt.row.min()}..{pt.row.max()}, "
                         f"{int(pt.human_target.sum())} human-positive, nuclear-target tiles missing: "
                         f"{', '.join(pt.tile_id[pt.nuclear_target]) or 'none'}); e.g. `{g.filename.iloc[0]}`")
    amb = index[index.status == "ambiguous"]
    if len(amb):
        lines += ["", "## Ambiguous", ""] + [f"- {r.region} {r.tile_id}: {r.note}" for r in amb.itertuples()]
    path.write_text("\n".join(lines) + "\n")


def build_index(root: Path, deep: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    idx = check_images(resolve(pt, root), root, deep)
    return idx, orphans(pt, root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-root")
    ap.add_argument("--deep", action="store_true", help="fully decode every image (slow)")
    args = ap.parse_args(argv)
    root = image_root_from(args.image_root)
    pt = pd.read_csv(OUTPUT_DIR / "parent_tiles.csv")
    idx, orph = build_index(root, args.deep)
    idx.to_csv(OUTPUT_DIR / "image_index.csv", index=False)
    write_report(idx, orph, pt, root, args.deep, OUTPUT_DIR / "image_integrity_report.md")
    print((OUTPUT_DIR / "image_integrity_report.md").read_text())
    bad = (idx.status != "ok") | idx.read_error.fillna("").ne("") | (idx.width != EXPECTED_PX) | (idx.height != EXPECTED_PX)
    return 1 if bad.any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
