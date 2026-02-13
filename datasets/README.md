# Legal Datasets

This directory contains sample legal data and scripts for fetching larger datasets from public APIs. The data is used throughout the LLM Playground notebooks for tokenization, chunking, retrieval, evaluation, and other exercises.

## Data Sources

| Source | API | Description |
|--------|-----|-------------|
| [CourtListener](https://www.courtlistener.com/) | `https://www.courtlistener.com/api/rest/v4/opinions/` | Free, open database of U.S. court opinions maintained by Free Law Project. No API key required. |
| [Congress.gov](https://www.congress.gov/) | `https://api.congress.gov/v3/` | Official source for U.S. federal legislation, including bill text, summaries, and status. Requires a free API key ([sign up here](https://api.congress.gov/sign-up/)). |

## Directory Structure

```
datasets/
├── README.md
├── sample/
│   ├── court_opinions.jsonl   # Small offline sample (5 records)
│   └── legislation.jsonl      # Small offline sample (5 records)
└── scripts/
    ├── fetch_courtlistener.py  # Fetch opinions from CourtListener API
    └── fetch_legislation.py    # Fetch bills from Congress.gov API
```

## Sample Data (Offline)

The `sample/` directory ships with small hardcoded datasets (5 records each) so that the notebooks work without network access or API keys.

### court_opinions.jsonl

Each line is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `integer` | Unique opinion identifier |
| `case_name` | `string` | Name of the case (e.g., "Smith v. Jones") |
| `court` | `string` | Court that issued the opinion |
| `date_filed` | `string` | Filing date in `YYYY-MM-DD` format |
| `text` | `string` | Full text of the court opinion |
| `citations` | `string[]` | List of cases cited in the opinion |

### legislation.jsonl

Each line is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Bill identifier in `{congress}-{type}-{number}` format |
| `title` | `string` | Short title of the bill |
| `congress` | `integer` | Congress number (e.g., 118) |
| `bill_type` | `string` | Bill type code (`hr`, `s`, `hjres`, `sjres`, etc.) |
| `text` | `string` | Full text of the bill (or excerpt) |

## Fetch Scripts

The scripts in `scripts/` call the public APIs to download larger, real datasets. They require Python 3.10+ and the `requests` library.

```bash
pip install requests
```

### Fetch Court Opinions

```bash
# Fetch 50 opinions (default)
python datasets/scripts/fetch_courtlistener.py

# Fetch 100 opinions
python datasets/scripts/fetch_courtlistener.py --limit 100

# Write to a custom path
python datasets/scripts/fetch_courtlistener.py --output /tmp/opinions.jsonl
```

No API key is required. The script handles pagination automatically and is polite to the API with a 1-second delay between pages.

### Fetch Legislation

```bash
# Set your API key (get one at https://api.congress.gov/sign-up/)
export CONGRESS_API_KEY="your-key-here"

# Fetch 50 bills (default)
python datasets/scripts/fetch_legislation.py

# Or pass the key directly
python datasets/scripts/fetch_legislation.py --api-key your-key-here --limit 100
```

The script fetches bill metadata and attempts to retrieve the full text of each bill. Some bills may have empty text if no formatted version is available through the API.

## Usage in Notebooks

Most notebooks load the sample data like this:

```python
import json
from pathlib import Path

data_path = Path("datasets/sample/court_opinions.jsonl")
opinions = [json.loads(line) for line in data_path.read_text().splitlines()]
```

If you have fetched larger datasets with the scripts above, the output files will overwrite the sample files in `sample/`. To restore the originals, check out the files from git:

```bash
git checkout -- datasets/sample/
```
