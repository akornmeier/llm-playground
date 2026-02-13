#!/usr/bin/env python3
"""Fetch sample court opinions from the CourtListener REST API.

Usage:
    python fetch_courtlistener.py                # fetch 50 opinions (default)
    python fetch_courtlistener.py --limit 100    # fetch 100 opinions

Output is written to datasets/sample/court_opinions.jsonl.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit(
        "The 'requests' library is required. Install it with:\n"
        "  pip install requests"
    )

API_BASE = "https://www.courtlistener.com/api/rest/v4/opinions/"

# Resolve output path relative to this script so it works regardless of cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "sample" / "court_opinions.jsonl"


def fetch_opinions(limit: int, output: Path) -> None:
    """Download *limit* court opinions and save them as JSONL."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "llm-playground dataset fetcher (educational project)",
        }
    )

    collected: list[dict] = []
    next_url: str | None = f"{API_BASE}?format=json&page_size=20&order_by=-date_filed"

    print(f"Fetching up to {limit} court opinions from CourtListener ...")

    while next_url and len(collected) < limit:
        try:
            resp = session.get(next_url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"\n[error] API request failed: {exc}", file=sys.stderr)
            if collected:
                print(
                    f"[info]  Saving {len(collected)} opinions collected so far.",
                    file=sys.stderr,
                )
                break
            else:
                sys.exit(1)

        data = resp.json()
        results = data.get("results", [])

        if not results:
            print("[info] No more results returned by the API.")
            break

        for item in results:
            if len(collected) >= limit:
                break

            opinion = {
                "id": item.get("id"),
                "case_name": item.get("case_name", item.get("cluster", {}).get("case_name", "")),
                "court": item.get("court", item.get("cluster", {}).get("court", "")),
                "date_filed": item.get("date_filed", item.get("cluster", {}).get("date_filed", "")),
                "text": (
                    item.get("plain_text")
                    or item.get("html")
                    or item.get("html_lawbox")
                    or item.get("html_columbia")
                    or item.get("html_with_citations")
                    or ""
                ),
                "citations": item.get("citations", []),
            }
            collected.append(opinion)

        print(f"  ... {len(collected)}/{limit} opinions collected")

        next_url = data.get("next")

        # Be polite to the API — wait between pages.
        time.sleep(1.0)

    # Write output -------------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as fh:
        for record in collected:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(collected)} opinions to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch court opinions from the CourtListener API."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of opinions to fetch (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    fetch_opinions(limit=args.limit, output=args.output)


if __name__ == "__main__":
    main()
