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

## Environment Setup

The notebooks require Python 3.11 and several dependencies. macOS ships with a newer
Python that isn't compatible with all ML libraries yet, so we use **pyenv** to install
3.11 and create an isolated virtual environment.

### 1. Install pyenv (Python version manager)

```bash
brew install pyenv
```

### 2. Install Python 3.11

```bash
pyenv install 3.11
```

Verify the installed version (e.g. 3.11.14):

```bash
ls ~/.pyenv/versions/
```

### 3. Create a virtual environment

From the repository root, create a `.venv` inside `notebooks/` using the pyenv-managed
Python. Use the exact version number from the previous step:

```bash
~/.pyenv/versions/3.11.14/bin/python -m venv notebooks/.venv
```

### 4. Install dependencies

```bash
notebooks/.venv/bin/pip install -r notebooks/requirements.txt
```

This will take a few minutes — PyTorch alone is ~2 GB.

### 5. Launch a notebook

```bash
notebooks/.venv/bin/jupyter notebook notebooks/01-data-collection/01_common_crawl.ipynb
```

## Prerequisites

- macOS with Homebrew
- Packages listed in `notebooks/requirements.txt`
- Sample data in `../../datasets/sample/court_opinions.jsonl` (included in this repo)

## Key Takeaways

- General web crawls (Common Crawl) contain enormous amounts of text, but only a
  tiny fraction is relevant to any specific domain.
- Purpose-built legal datasets (CourtListener) provide cleaner text, richer metadata,
  and reliable provenance -- all essential qualities for legal AI applications.
- Data collection choices made at this stage propagate through every downstream step:
  cleaning, tokenization, training, and evaluation.
