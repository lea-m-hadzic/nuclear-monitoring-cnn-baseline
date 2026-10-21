"""Keep the Google API key out of everything we write or print.

The vendor script puts the key in the `url` manifest column (the full Static Maps URL), and an HTTP error from
`resp.raise_for_status()` includes that URL in its message (printed to stdout and written to the manifest
`message` column). Everything coming out of the vendor script passes through `redact()` before it is logged,
and manifests are rewritten with `sanitize_manifest()` right after each run.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

KEY_PARAM_RE = re.compile(r"(key=)[^&\s'\",;<>]+")
GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
REDACTED = "REDACTED"


def redact(text: str, key: str | None = None) -> str:
    if key:
        text = text.replace(key, REDACTED)
    text = KEY_PARAM_RE.sub(rf"\g<1>{REDACTED}", text)
    return GOOGLE_KEY_RE.sub(REDACTED, text)


def contains_secret(text: str, key: str | None = None) -> bool:
    return bool((key and key in text) or GOOGLE_KEY_RE.search(text)
                or any(m.group(0) != f"key={REDACTED}" for m in KEY_PARAM_RE.finditer(text)))


def sanitize_manifest(path: Path, key: str | None = None) -> int:
    """Redact every cell of a vendor manifest in place; returns the number of cells changed."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    changed = 0
    for row in rows:
        for j, cell in enumerate(row):
            new = redact(cell, key)
            if new != cell:
                row[j], changed = new, changed + 1
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    os.replace(tmp, path)
    return changed


def scan_dir_for_secrets(root: Path, key: str | None = None) -> list[Path]:
    """Text files under root that still contain a key (never prints the key itself)."""
    hits = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".html", ".log", ".json", ".jsonl", ".txt", ".md"}:
            if contains_secret(p.read_text(encoding="utf-8", errors="ignore"), key):
                hits.append(p)
    return hits
