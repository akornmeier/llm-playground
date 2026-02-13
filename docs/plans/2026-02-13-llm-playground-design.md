# LLM Playground — Design Document

## Purpose

A project-based learning playground for senior developers with some LLM experience who want deeper understanding of the full stack: pre-training through production. Concepts are grounded in a legal-AI context inspired by Thomson Reuters CoCounsel, using freely available legal data (CourtListener, congress.gov, SEC filings).

## Audience

Senior developers who have used LLM APIs and possibly fine-tuned models, but want to understand the mechanics — tokenization internals, training loops, alignment techniques, evaluation methodology — and how these connect to production system design.

## Stack

- **Foundations (notebooks/):** Python, Jupyter, PyTorch, HuggingFace transformers/tokenizers
- **Application (app/):** TypeScript, Vercel AI SDK, pnpm
- **Shared data (datasets/):** Court opinions, legislation, SEC filings

## Project Structure

```
llm-playground/
├── notebooks/                    # Python — foundational concepts
│   ├── 01-data-collection/       # Manual crawling, Common Crawl exploration
│   ├── 02-data-cleaning/         # RefinedWeb/FineWeb pipelines, quality filtering
│   ├── 03-tokenization/          # BPE from scratch, HuggingFace tokenizers
│   ├── 04-architecture/          # Transformer internals, attention visualization
│   ├── 05-text-generation/       # Greedy, beam, top-k, top-p sampling
│   ├── 06-sft/                   # Supervised fine-tuning on small models
│   ├── 07-rlhf/                  # Reward models, PPO, DPO walkthrough
│   ├── 08-evaluation/            # Metrics, benchmarks, human eval patterns
│   └── 09-fine-tuning-lab/       # Optional: QLoRA fine-tuning with legal data
├── app/                          # TypeScript — progressive legal-AI assistant
│   ├── src/
│   └── package.json
├── datasets/                     # Shared sample data (court opinions, legislation)
├── docs/                         # Design docs and plans
├── notebooks/requirements.txt    # Python dependencies
└── package.json                  # Root pnpm workspace
```

Each notebook module contains a README.md, one or more .ipynb files, and supporting scripts. Modules are numbered in recommended learning order but 01–08 can be approached independently.

## Concept Notebooks — Pre-Training (01–05)

Each notebook follows a consistent pattern: **context** (why this matters for a product like CoCounsel), **theory** (concise explanation), **hands-on code**, and **exercises**.

### 01 — Data Collection

Explore Common Crawl's WARC files, extract a small corpus of legal text, and compare against curated sources like CourtListener's API. Learners see firsthand how noisy raw web data is versus purpose-scraped legal text.

### 02 — Data Cleaning

Take the raw corpus from module 01 and apply quality filtering inspired by RefinedWeb and FineWeb pipelines — deduplication (MinHash), language filtering, content extraction, PII removal. Output a clean dataset that feeds into later modules.

### 03 — Tokenization

Implement BPE from scratch in ~50 lines, then compare against HuggingFace's `tokenizers` library. Visualize how legal terminology (case citations, statute numbers) tokenizes differently across vocabularies. Train a custom tokenizer on the legal corpus.

### 04 — Architecture

Build a minimal transformer block (attention + FFN) in PyTorch. Visualize attention patterns. Walk through GPT and Llama architecture differences (positional encoding, RMSNorm, GQA). No training — forward passes with pre-trained weights to inspect internals.

### 05 — Text Generation

Load a small pre-trained model and implement generation strategies from scratch: greedy, beam search, top-k, top-p, temperature scaling. Compare outputs on legal prompts. Visualize probability distributions at each token to build intuition about why sampling strategies matter.

## Concept Notebooks — Post-Training & Evaluation (06–09)

### 06 — Supervised Fine-Tuning (SFT)

Start with a small base model (SmolLM or TinyLlama). Build an instruction dataset from legal Q&A pairs — format raw court opinions into instruction/response pairs. Walk through the SFT training loop: data formatting, loss masking on prompts vs completions, learning rate scheduling. Train on a tiny dataset to observe behavior change before and after.

### 07 — RLHF & Alignment

Build a preference dataset by ranking model outputs on legal questions. Train a small reward model, visualize what it scores highly, then walk through PPO conceptually with annotated pseudocode. Implement DPO as a simpler alternative that learners can actually run. Compare aligned vs unaligned outputs on tricky legal prompts (hallucinated citations, hedging vs overconfidence). Cover verifiable rewards for tasks with ground truth (e.g., citation accuracy).

### 08 — Evaluation

Apply traditional metrics (perplexity, BLEU, ROUGE) to the legal domain and show where they break down. Build a small evaluation harness: test the model against legal reasoning benchmarks, measure citation accuracy, check for hallucination. Introduce LLM-as-judge patterns. Discuss leaderboard methodology and why benchmark gaming matters for product teams.

### 09 — Fine-Tuning Lab (Optional)

For learners with GPU access. QLoRA fine-tuning of a 7B model on the legal corpus. Covers adapter merging, quantization trade-offs, and evaluation against the base model. Structured as a self-guided lab with checkpoints rather than a linear notebook.

## Progressive Application — Legal AI Assistant (TypeScript)

The `app/` directory builds a CoCounsel-inspired assistant incrementally. Each stage is a tagged milestone so learners can check out any point in the progression.

### Stage 1 — Document Ingestion

Ingest court opinions and legislation into a vector store. Covers chunking strategies for legal text (section-aware splitting vs naive), embedding model selection, and storage (local SQLite with pgvector or ChromaDB). Learners see how chunking choices directly affect retrieval quality.

### Stage 2 — RAG Pipeline

Build retrieval-augmented generation: query the vector store, construct prompts with retrieved context, generate answers with citations. Use Vercel AI SDK with configurable providers (OpenAI, Anthropic, Ollama for local models). Focus on prompt engineering for legal accuracy — instructing the model to cite sources and refuse when evidence is insufficient.

### Stage 3 — Citation Grounding & Hallucination Detection

Add a verification layer that cross-references generated citations against the ingested corpus. Flag unverifiable claims. This is the core trust problem for legal AI and connects directly back to the evaluation notebook.

### Stage 4 — Conversation & Memory

Add multi-turn conversation with context management. Cover token budget strategies, conversation summarization, and how to maintain coherence across a long legal research session.

### Stage 5 — Evaluation & Guardrails

Wire in the evaluation patterns from notebook 08. Automated tests for citation accuracy, hallucination rate, and refusal behavior. Add input/output guardrails relevant to legal use (PII detection, scope limitations). This becomes the quality gate for the full system.

## Learning Path

The default path accessible without GPU:

1. Notebooks 01–05 (foundations, in order or selectively)
2. Notebooks 06–08 (post-training concepts with small models)
3. App stages 1–5 (progressive application build)

Optional GPU path: Notebook 09 after completing 06–07.

## Data Sources

- **CourtListener** (courtlistener.com) — Court opinions via REST API
- **Congress.gov** — Legislation text
- **SEC EDGAR** — Public filings
- Small curated samples committed to `datasets/` for offline use
