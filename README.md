# LLM Playground

A project-based learning repository for senior developers who want to understand LLM internals and production patterns. Rather than treating models as black boxes, you build key components from scratch -- BPE tokenizers, transformer blocks, decoding strategies, RLHF pipelines -- then apply those concepts in a progressive TypeScript legal-AI application grounded in open legal data.

## Learning Path

1. **Foundations (Modules 01-05):** Data collection, cleaning, tokenization, transformer architecture, text generation
2. **Post-training (Modules 06-08):** Supervised fine-tuning, RLHF/DPO alignment, evaluation metrics
3. **Application (App Stages 1-5):** Document ingestion, RAG retrieval, citation grounding, conversation memory, guardrails and evaluation
4. **Optional (Module 09):** QLoRA fine-tuning lab (requires GPU)

## Prerequisites

- Python 3.11+
- Node.js 20+
- pnpm (`corepack enable && corepack prepare pnpm@latest --activate`)

## Quick Start

```bash
# Clone
git clone <repo-url> && cd llm-playground

# Python dependencies (for notebooks)
pip install -r notebooks/requirements.txt

# TypeScript dependencies (for app)
cd app && pnpm install && cd ..

# Run a notebook
jupyter notebook notebooks/03-tokenization/01_bpe_from_scratch.ipynb

# Run the app (interactive CLI)
cd app && pnpm run dev

# Run tests
cd app && pnpm test

# Run tests with coverage
cd app && pnpm test:coverage
```

## Notebook Modules

| Module | Name | Description |
|--------|------|-------------|
| 01 | Data Collection | Crawl Common Crawl and query the CourtListener API for legal text |
| 02 | Data Cleaning | Quality filtering, language detection, and MinHash deduplication |
| 03 | Tokenization | Build a BPE tokenizer from scratch, then use HuggingFace tokenizers |
| 04 | Architecture | Implement a transformer block and compare GPT vs LLaMA designs |
| 05 | Text Generation | Explore greedy, top-k, top-p, and temperature-based decoding |
| 06 | SFT | Create instruction datasets and run supervised fine-tuning |
| 07 | RLHF | Build a reward model and implement Direct Preference Optimization |
| 08 | Evaluation | Compute ROUGE, BLEU, perplexity and use evaluation harnesses |
| 09 | Fine-Tuning Lab | End-to-end QLoRA fine-tuning on legal data (requires GPU) |

## App Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | Ingestion | Chunk documents and store embeddings in a vector store |
| 2 | RAG | Retrieve relevant chunks and generate answers with an LLM |
| 3 | Verification | Extract citations from answers and verify against source documents |
| 4 | Conversation | Manage multi-turn conversation history with token-aware truncation |
| 5 | Guardrails/Eval | Input scope filtering, PII detection, output safety checks, and metrics |

All stages are wired together in `app/src/pipeline.ts` and exposed through a CLI REPL in `app/src/index.ts`.

## Data Sources

Sample datasets ship in `datasets/sample/` so notebooks work offline. For larger datasets, use the fetch scripts:

- **CourtListener** -- U.S. court opinions (no API key required): `python datasets/scripts/fetch_courtlistener.py --limit 50`
- **Congress.gov** -- Federal legislation (free API key required): `python datasets/scripts/fetch_legislation.py --api-key <KEY> --limit 50`

See `datasets/README.md` for schema details and usage.

## Design Document

The full design document is at `docs/plans/2026-02-13-llm-playground-design.md`.
