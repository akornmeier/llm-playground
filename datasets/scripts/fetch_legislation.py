#!/usr/bin/env python3
"""Fetch sample bills from the Congress.gov API.

Usage:
    python fetch_legislation.py --api-key YOUR_KEY
    CONGRESS_API_KEY=YOUR_KEY python fetch_legislation.py
    python fetch_legislation.py --api-key YOUR_KEY --limit 100

Output is written to datasets/sample/legislation.jsonl.

An API key is required. You can obtain one for free at:
    https://api.congress.gov/sign-up/
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

API_BASE = "https://api.congress.gov/v3"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "sample" / "legislation.jsonl"


def fetch_bill_text(session: "requests.Session", bill: dict, api_key: str) -> str:
    """Attempt to retrieve the full text of a bill via its text-versions endpoint."""
    bill_type = bill.get("type", "").lower()
    bill_number = bill.get("number", "")
    congress = bill.get("congress", "")

    if not (bill_type and bill_number and congress):
        return ""

    text_url = f"{API_BASE}/bill/{congress}/{bill_type}/{bill_number}/text"
    try:
        resp = session.get(
            text_url,
            params={"api_key": api_key, "format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        versions = resp.json().get("textVersions", [])
        if versions:
            # Return the most recent version's plain-text URL content or
            # fall back to the formatted text snippet provided by the API.
            for version in versions:
                formats = version.get("formats", [])
                for fmt in formats:
                    if fmt.get("type") == "Formatted Text":
                        txt_url = fmt.get("url")
                        if txt_url:
                            txt_resp = session.get(txt_url, timeout=30)
                            txt_resp.raise_for_status()
                            return txt_resp.text[:50_000]  # cap length
            return ""
    except requests.RequestException:
        return ""


def fetch_legislation(limit: int, api_key: str, output: Path) -> None:
    """Download *limit* recent bills and save them as JSONL."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "llm-playground dataset fetcher (educational project)",
        }
    )

    collected: list[dict] = []
    offset = 0
    page_size = 20

    print(f"Fetching up to {limit} bills from Congress.gov ...")

    while len(collected) < limit:
        try:
            resp = session.get(
                f"{API_BASE}/bill",
                params={
                    "api_key": api_key,
                    "format": "json",
                    "limit": page_size,
                    "offset": offset,
                    "sort": "updateDate+desc",
                },
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"\n[error] API request failed: {exc}", file=sys.stderr)
            if collected:
                print(
                    f"[info]  Saving {len(collected)} bills collected so far.",
                    file=sys.stderr,
                )
                break
            else:
                sys.exit(1)

        data = resp.json()
        bills = data.get("bills", [])

        if not bills:
            print("[info] No more results returned by the API.")
            break

        for bill in bills:
            if len(collected) >= limit:
                break

            congress = bill.get("congress", "")
            bill_type = bill.get("type", "").lower()
            bill_number = bill.get("number", "")

            bill_id = f"{congress}-{bill_type}-{bill_number}" if congress else ""

            # Try to get full text (this makes an extra API call per bill).
            text = fetch_bill_text(session, bill, api_key)
            time.sleep(0.5)  # rate-limit politeness

            record = {
                "id": bill_id,
                "title": bill.get("title", ""),
                "congress": congress,
                "bill_type": bill_type,
                "text": text,
            }
            collected.append(record)

        print(f"  ... {len(collected)}/{limit} bills collected")

        offset += page_size
        time.sleep(0.5)

    # Write output -------------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as fh:
        for record in collected:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(collected)} bills to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch bills from the Congress.gov API."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CONGRESS_API_KEY", ""),
        help="Congress.gov API key (or set CONGRESS_API_KEY env var)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of bills to fetch (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.api_key:
        sys.exit(
            "An API key is required. Provide one via --api-key or the "
            "CONGRESS_API_KEY environment variable.\n"
            "Sign up for a free key at: https://api.congress.gov/sign-up/"
        )

    fetch_legislation(limit=args.limit, api_key=args.api_key, output=args.output)


if __name__ == "__main__":
    main()
