# Module 02: Data Cleaning

This module covers **data cleaning pipelines for LLM training data**, inspired
by the techniques described in the
[RefinedWeb](https://arxiv.org/abs/2306.01116) and
[FineWeb](https://arxiv.org/abs/2406.17557) papers.

You will build quality filters and deduplication systems using legal text
(court opinions) as the working domain. Legal documents present unique cleaning
challenges: OCR artifacts, boilerplate headers and footers, page numbers,
standardized citation formats, and near-duplicate opinions issued by the same
court.

## Notebooks

| # | Notebook | Topics |
|---|----------|--------|
| 1 | `01_quality_filtering.ipynb` | Language detection, content heuristics (line length, symbol ratio, repetition, boilerplate), PII detection and redaction, building a full cleaning pipeline |
| 2 | `02_deduplication.ipynb` | Exact hash-based deduplication, MinHash + LSH for near-duplicate detection, parameter tuning for precision/recall trade-offs |

## Prerequisites

- **Module 01** (Data Collection) -- or use the provided sample data at
  `datasets/sample/court_opinions.jsonl`.
- Python packages: `langdetect`, `datasketch` (install via
  `pip install langdetect datasketch`).
