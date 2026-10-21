"""Run Qiqi's UNMODIFIED extraction script and meter its Static Maps requests exactly.

    python vendor_runner.py <vendor_script.py> <vendor args...>

`requests.Session.get` is wrapped (count only, the call is unchanged) before the script's own `__main__` block runs
via runpy, so the code path is byte-for-byte the vendor's. The count is printed last as
`__STATIC_MAPS_REQUESTS__=<n>` for the wrapper to parse; no URL (which carries the key) is ever printed here.
"""
from __future__ import annotations

import runpy
import sys

import requests

COUNT_TAG = "__STATIC_MAPS_REQUESTS__="
_count = 0
_orig_get = requests.Session.get


def _counting_get(self, url, *args, **kwargs):
    global _count
    if "maps.googleapis.com/maps/api/staticmap" in str(url):
        _count += 1
    return _orig_get(self, url, *args, **kwargs)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: vendor_runner.py <vendor_script.py> [args...]", file=sys.stderr)
        return 2
    requests.Session.get = _counting_get
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    code = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        print(f"{COUNT_TAG}{_count}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
