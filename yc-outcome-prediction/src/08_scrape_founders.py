"""08 — Scrape founder data from ycombinator.com company pages.

Each company page embeds a JSON blob (data-page attribute) containing
props.company.founders: [{full_name, title, founder_bio, is_active, ...}].
Notably is_active=false founders are retained (departed founders visible),
which softens — but does not eliminate — the current-day-snapshot concern.

Output: data/raw/founders.jsonl (one line per company, checkpointed —
rerunning skips already-scraped slugs). Keeps every founder field except
avatar URLs (signed, ephemeral, useless).
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = RAW / "founders.jsonl"

WORKERS = 6
TIMEOUT = 30
RETRIES = 3
UA = "Mozilla/5.0 (X11; Linux x86_64) research-scraper (contact: repo yc-outcome-prediction)"
DATA_PAGE = re.compile(r'data-page="([^"]+)"')
DROP_KEYS = {"avatar_thumb_url", "avatar_url", "avatar_medium_url"}


def fetch(slug: str) -> dict:
    url = f"https://www.ycombinator.com/companies/{slug}"
    last_err = None
    for attempt in range(RETRIES):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=TIMEOUT) as r:
                body = r.read().decode("utf-8", errors="replace")
            m = DATA_PAGE.search(body)
            if not m:
                return {"slug": slug, "error": "no data-page blob"}
            data = json.loads(html.unescape(m.group(1)))
            comp = data.get("props", {}).get("company", {}) or {}
            founders = comp.get("founders")
            if founders is None:
                return {"slug": slug, "error": "no founders key"}
            clean = [{k: v for k, v in f.items() if k not in DROP_KEYS}
                     for f in founders]
            return {"slug": slug, "n_founders_listed": len(clean),
                    "founders": clean}
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return {"slug": slug, "error": last_err}


def main() -> None:
    companies = json.load(open(RAW / "yc_companies_all.json"))
    slugs = [c["slug"] for c in companies]
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {json.loads(line)["slug"] for line in f if line.strip()}
    todo = [s for s in slugs if s not in done]
    print(f"total {len(slugs)}, done {len(done)}, todo {len(todo)}", flush=True)

    t0 = time.time()
    n_ok = n_err = 0
    with open(OUT, "a") as sink, ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(fetch, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            sink.write(json.dumps(rec) + "\n")
            if "error" in rec:
                n_err += 1
            else:
                n_ok += 1
            if i % 250 == 0:
                sink.flush()
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate / 60
                print(f"{i}/{len(todo)} ok={n_ok} err={n_err} "
                      f"({rate:.1f}/s, eta {eta:.0f}m)", flush=True)
    print(f"DONE ok={n_ok} err={n_err} elapsed={(time.time()-t0)/60:.1f}m",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
