# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM Playground is a project-based learning repository for senior developers exploring LLM internals and production patterns. It pairs concept notebooks (data pipelines, tokenization, transformer architecture, fine-tuning) with a progressive TypeScript legal-AI application, all grounded in open legal data sources.

## Stack

- **Notebooks (notebooks/):** Python 3.11+, Jupyter, PyTorch, HuggingFace Transformers/Tokenizers/TRL
- **Application (app/):** TypeScript, Vitest, Vercel AI SDK, pnpm
- **Package manager:** pnpm 10.28.1

## Commands

### Python (notebooks and datasets)

```bash
pip install -r notebooks/requirements.txt
jupyter notebook notebooks/01-data-collection/01_common_crawl.ipynb
python datasets/scripts/fetch_courtlistener.py --limit 50
python datasets/scripts/fetch_legislation.py --api-key <KEY> --limit 50
```

### TypeScript app

```bash
cd app && pnpm install
pnpm test                          # run all tests
pnpm test:coverage                 # run tests with coverage report
pnpm test -- src/ingestion/        # run tests for a single module
pnpm run dev                       # start the CLI REPL
pnpm lint                          # ESLint
pnpm format                        # Prettier
```

## Architecture

Two-part structure:

1. **notebooks/** -- 9 Python Jupyter notebook modules (01-09) covering pre-training foundations (data collection, cleaning, tokenization, architecture, generation) and post-training (SFT, RLHF, evaluation, fine-tuning lab). Numbered in recommended order but independently approachable.
2. **app/** -- TypeScript legal-AI assistant with 5 modules (ingestion, rag, verification, conversation, guardrails/eval) wired together by `pipeline.ts`. A CLI entry point (`index.ts`) provides an interactive REPL.

Shared sample data lives in **datasets/** (court opinions, legislation).

See `docs/plans/2026-02-13-llm-playground-design.md` for the full design document.
