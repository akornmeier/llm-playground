# Module 01: Data Collection

This module explores how LLM training data is collected, from broad web crawls to
curated domain-specific sources. You will work hands-on with Common Crawl WARC files
and compare them against structured legal data from CourtListener.

## Why This Matters for Legal AI

Understanding data provenance is critical when building AI systems for the legal
domain. CoCounsel and similar legal AI products must be grounded in reliable,
authoritative sources. A model trained primarily on general web text will behave
differently from one trained on curated judicial opinions. This module gives you
first-hand experience with both kinds of data so you can reason about source quality,
coverage gaps, and the practical trade-offs involved in assembling a training corpus.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_common_crawl.ipynb` | Download and parse WARC files from Common Crawl; filter for legal content; see how sparse legal text is in a general web crawl. |
| 2 | `02_courtlistener_api.ipynb` | Load structured court opinions from CourtListener; compare data quality against raw web extracts; compute corpus statistics. |

## Prerequisites

- Python 3.10+
- Packages: `warcio`, `beautifulsoup4`, `requests`
- Sample data in `../../datasets/sample/court_opinions.jsonl` (included in this repo)

## Key Takeaways

- General web crawls (Common Crawl) contain enormous amounts of text, but only a
  tiny fraction is relevant to any specific domain.
- Purpose-built legal datasets (CourtListener) provide cleaner text, richer metadata,
  and reliable provenance -- all essential qualities for legal AI applications.
- Data collection choices made at this stage propagate through every downstream step:
  cleaning, tokenization, training, and evaluation.
