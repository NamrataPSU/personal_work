#!/usr/bin/env python3
"""01_acquire.py — Download the YC open-source companies dataset.

Source: https://github.com/yc-oss/api (mirrored via raw.githubusercontent.com,
since yc-oss.github.io is blocked by the network proxy).

Idempotent: skips download if the target file already exists and is non-empty,
unless --force is passed. Uses urllib.request, which respects the HTTPS_PROXY
environment variable. The proxy's CA bundle (/root/.ccr/ca-bundle.crt) is used
for TLS verification when present.

Usage: python src/01_acquire.py [--force]
"""
import argparse
import json
import os
import ssl
import sys
import urllib.request
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

BASE = "https://raw.githubusercontent.com/yc-oss/api/main"
TARGETS = {
    f"{BASE}/companies/all.json": RAW_DIR / "yc_companies_all.json",
    f"{BASE}/meta.json": RAW_DIR / "yc_meta.json",
}
CA_BUNDLE = "/root/.ccr/ca-bundle.crt"


def make_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if os.path.exists(CA_BUNDLE):
        ctx.load_verify_locations(CA_BUNDLE)
    return ctx


def download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[skip] {dest} already exists ({dest.stat().st_size:,} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}")
    with urllib.request.urlopen(url, context=make_context(), timeout=120) as resp:
        data = resp.read()
    # Validate it is parseable JSON before writing.
    json.loads(data)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    print(f"[ok  ] {dest} ({len(data):,} bytes)")


def write_provenance() -> None:
    prov = RAW_DIR / "PROVENANCE.md"
    n = "unknown"
    all_path = RAW_DIR / "yc_companies_all.json"
    if all_path.exists():
        n = len(json.loads(all_path.read_text()))
    prov.write_text(f"""# Data Provenance

- **Dataset**: Y Combinator companies (open-source mirror of the YC directory)
- **Source repo**: https://github.com/yc-oss/api
- **Download URLs**:
  - https://raw.githubusercontent.com/yc-oss/api/main/companies/all.json -> `yc_companies_all.json`
  - https://raw.githubusercontent.com/yc-oss/api/main/meta.json -> `yc_meta.json`
- **Download date**: {date.today().isoformat()} (originally acquired 2026-07-02)
- **Records**: {n} companies
- **Notes**: The canonical API host (yc-oss.github.io) was blocked by the network
  proxy, so files were fetched from the repo's `main` branch via
  raw.githubusercontent.com. The repo auto-updates daily from YC's Algolia index;
  `yc_meta.json.last_updated` records the upstream refresh timestamp.
- **License/terms**: Data originates from Y Combinator's public directory
  (https://www.ycombinator.com/companies); yc-oss/api is an unofficial mirror.
""")
    print(f"[ok  ] {prov}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args()
    for url, dest in TARGETS.items():
        download(url, dest, force=args.force)
    write_provenance()


if __name__ == "__main__":
    sys.exit(main())
