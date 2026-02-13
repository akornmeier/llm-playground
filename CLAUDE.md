# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM Playground — a project-based learning repository for senior developers exploring LLM internals and production patterns. Concepts are grounded in a legal-AI context (CoCounsel-inspired) using open legal data sources.

## Stack

- **Concept notebooks (notebooks/):** Python, Jupyter, PyTorch, HuggingFace
- **Progressive application (app/):** TypeScript, Vercel AI SDK, pnpm
- **Package manager:** pnpm 10.28.1

## Commands

- **Install JS dependencies:** `pnpm install`
- **Install Python dependencies:** `pip install -r notebooks/requirements.txt`
- **Run tests:** `pnpm test` (not yet configured)

## Architecture

Two-part structure:

1. **notebooks/** — Self-contained Python Jupyter notebooks covering LLM foundations (data collection, cleaning, tokenization, architecture, generation) and post-training (SFT, RLHF, evaluation). Numbered 01–09 in recommended order but independently approachable.
2. **app/** — TypeScript legal-AI assistant built in progressive stages: document ingestion, RAG pipeline, citation grounding, conversation memory, evaluation & guardrails.

Shared sample data lives in **datasets/** (court opinions, legislation, SEC filings).

See `docs/plans/2026-02-13-llm-playground-design.md` for the full design document.
