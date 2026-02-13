# LLM Playground Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a project-based LLM learning playground with Python concept notebooks (01–09) and a progressive TypeScript legal-AI assistant application.

**Architecture:** Python Jupyter notebooks for foundational LLM concepts using PyTorch/HuggingFace, TypeScript application using Vercel AI SDK for the progressive legal-AI assistant. Shared legal datasets (CourtListener, congress.gov) in a common `datasets/` directory.

**Tech Stack:** Python 3.11+, Jupyter, PyTorch, HuggingFace transformers/tokenizers, TypeScript, Vercel AI SDK, pnpm, Vitest, ChromaDB

---

## Task 1: Project Scaffolding — Directory Structure

**Files:**
- Create: `notebooks/01-data-collection/README.md`
- Create: `notebooks/02-data-cleaning/README.md`
- Create: `notebooks/03-tokenization/README.md`
- Create: `notebooks/04-architecture/README.md`
- Create: `notebooks/05-text-generation/README.md`
- Create: `notebooks/06-sft/README.md`
- Create: `notebooks/07-rlhf/README.md`
- Create: `notebooks/08-evaluation/README.md`
- Create: `notebooks/09-fine-tuning-lab/README.md`
- Create: `datasets/.gitkeep`
- Create: `app/package.json`
- Create: `app/src/.gitkeep`

**Step 1: Create all notebook module directories with placeholder READMEs**

Each README should contain a one-paragraph description of the module, what the learner will build, and prerequisites (other module numbers if any). Use the descriptions from the design doc at `docs/plans/2026-02-13-llm-playground-design.md`.

**Step 2: Create the datasets directory**

```bash
mkdir -p datasets
touch datasets/.gitkeep
```

**Step 3: Create app scaffold**

Create `app/package.json`:
```json
{
  "name": "legal-ai-assistant",
  "version": "0.0.1",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage",
    "lint": "eslint src/",
    "format": "prettier --write 'src/**/*.ts'"
  }
}
```

**Step 4: Set up pnpm workspace**

Create `pnpm-workspace.yaml` at repo root:
```yaml
packages:
  - "app"
```

**Step 5: Update root .gitignore**

Add Python-specific ignores:
```
node_modules/
.claude/
logs/
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
*.egg-info/
dist/
.env
```

**Step 6: Commit**

```bash
git add .
git commit -m "chore: scaffold project directory structure"
```

---

## Task 2: Python Environment Setup

**Files:**
- Create: `notebooks/requirements.txt`
- Create: `notebooks/pyproject.toml`

**Step 1: Create requirements.txt**

```text
jupyter>=1.0
notebook>=7.0
torch>=2.1
transformers>=4.36
tokenizers>=0.15
datasets>=2.16
accelerate>=0.25
peft>=0.7
trl>=0.7
bitsandbytes>=0.41
evaluate>=0.4
rouge-score>=0.1
nltk>=3.8
scikit-learn>=1.3
pandas>=2.1
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
plotly>=5.18
warcio>=1.7
beautifulsoup4>=4.12
requests>=2.31
datasketch>=1.6
langdetect>=1.0
```

**Step 2: Create pyproject.toml for notebook configuration**

```toml
[project]
name = "llm-playground-notebooks"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 3: Verify Python environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r notebooks/requirements.txt
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

Expected: Version numbers printed, no import errors.

**Step 4: Commit**

```bash
git add notebooks/requirements.txt notebooks/pyproject.toml
git commit -m "chore: add Python dependencies for notebook modules"
```

---

## Task 3: TypeScript App Environment Setup

**Files:**
- Modify: `app/package.json`
- Create: `app/tsconfig.json`
- Create: `app/vitest.config.ts`
- Create: `app/.eslintrc.json`
- Create: `app/.prettierrc`

**Step 1: Install app dependencies**

```bash
cd app
pnpm add ai @ai-sdk/openai @ai-sdk/anthropic chromadb
pnpm add -D typescript tsx vitest @vitest/coverage-v8 eslint prettier @types/node
```

**Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "esModuleInterop": true,
    "strict": true,
    "outDir": "dist",
    "rootDir": "src",
    "declaration": true,
    "sourceMap": true,
    "skipLibCheck": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
```

**Step 3: Create vitest.config.ts**

```typescript
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      thresholds: { lines: 80, functions: 80, branches: 80 },
    },
  },
});
```

**Step 4: Verify setup**

```bash
cd app
pnpm exec tsc --noEmit
pnpm test
```

Expected: TypeScript compiles with no errors. Vitest runs (0 tests).

**Step 5: Commit**

```bash
git add app/ pnpm-lock.yaml
git commit -m "chore: set up TypeScript app with Vitest, ESLint, Prettier"
```

---

## Task 4: Sample Legal Dataset Curation

**Files:**
- Create: `datasets/README.md`
- Create: `datasets/scripts/fetch_courtlistener.py`
- Create: `datasets/scripts/fetch_legislation.py`
- Create: `datasets/sample/court_opinions.jsonl`
- Create: `datasets/sample/legislation.jsonl`

**Step 1: Create dataset fetch scripts**

`datasets/scripts/fetch_courtlistener.py` — Script that calls the CourtListener REST API (`https://www.courtlistener.com/api/rest/v4/opinions/`) to download 50–100 sample court opinions. Store as JSONL with fields: `id`, `case_name`, `court`, `date_filed`, `text`, `citations`.

`datasets/scripts/fetch_legislation.py` — Script that fetches sample bills from congress.gov API. Store as JSONL with fields: `id`, `title`, `congress`, `bill_type`, `text`.

**Step 2: Run fetch scripts to populate sample data**

```bash
cd datasets
python scripts/fetch_courtlistener.py
python scripts/fetch_legislation.py
```

Expected: `sample/court_opinions.jsonl` and `sample/legislation.jsonl` each contain records.

**Step 3: Create datasets/README.md**

Document the data sources, schema for each JSONL file, and how to refresh the data using the fetch scripts.

**Step 4: Verify data loads**

```bash
python -c "
import json
with open('datasets/sample/court_opinions.jsonl') as f:
    records = [json.loads(line) for line in f]
    print(f'Court opinions: {len(records)} records')
    print(f'Fields: {list(records[0].keys())}')
"
```

Expected: Record count and field list printed.

**Step 5: Commit**

```bash
git add datasets/
git commit -m "feat: add legal dataset fetch scripts and sample data"
```

---

## Task 5: Notebook 01 — Data Collection

**Files:**
- Create: `notebooks/01-data-collection/01_common_crawl.ipynb`
- Create: `notebooks/01-data-collection/02_courtlistener_api.ipynb`
- Modify: `notebooks/01-data-collection/README.md`

**Step 1: Update module README**

Explain: this module explores how LLM training data is collected. Learners will work with Common Crawl WARC files and compare against curated legal data from CourtListener. CoCounsel context: understanding data provenance matters for legal AI where source reliability is critical.

**Step 2: Create 01_common_crawl.ipynb**

Notebook sections:
1. **Context** — Why data collection matters for legal AI. Brief on Common Crawl's role in training GPT/Llama models.
2. **Theory** — How web crawling works, WARC format, how Common Crawl organizes data by segments.
3. **Hands-on** — Download a single WARC segment using `warcio`. Parse HTML records, extract text with BeautifulSoup. Filter for legal-domain pages (look for `.gov`, `law`, `court` in URLs). Count: how much of a random WARC segment is legal content? (Answer: very little.)
4. **Exercises** — (a) Write a filter that extracts pages containing legal citations (regex for case reporters like "F.3d", "U.S."). (b) Estimate how many WARC segments you'd need to build a 1GB legal corpus.

**Step 3: Create 02_courtlistener_api.ipynb**

Notebook sections:
1. **Context** — Purpose-built legal data vs general web crawls. CourtListener as an open legal data source.
2. **Hands-on** — Use CourtListener REST API to fetch court opinions. Compare text quality, structure, and metadata richness against Common Crawl extracts. Show side-by-side: same case from WARC vs CourtListener.
3. **Exercises** — (a) Fetch opinions from a specific court and date range. (b) Compute basic corpus statistics (document count, average length, vocabulary size).

**Step 4: Verify notebooks run**

```bash
cd notebooks/01-data-collection
jupyter nbconvert --to notebook --execute 01_common_crawl.ipynb --ExecutePreprocessor.timeout=300
jupyter nbconvert --to notebook --execute 02_courtlistener_api.ipynb --ExecutePreprocessor.timeout=300
```

Expected: Both notebooks execute without errors.

**Step 5: Commit**

```bash
git add notebooks/01-data-collection/
git commit -m "feat(notebooks): add module 01 — data collection"
```

---

## Task 6: Notebook 02 — Data Cleaning

**Files:**
- Create: `notebooks/02-data-cleaning/01_quality_filtering.ipynb`
- Create: `notebooks/02-data-cleaning/02_deduplication.ipynb`
- Modify: `notebooks/02-data-cleaning/README.md`

**Step 1: Update module README**

Prerequisites: Module 01 (or use provided sample data from `datasets/sample/`).

**Step 2: Create 01_quality_filtering.ipynb**

Notebook sections:
1. **Context** — Why cleaning matters: garbage in, garbage out. Reference RefinedWeb and FineWeb papers. CoCounsel context: legal text has unique noise (headers, footers, page numbers from OCR'd documents).
2. **Theory** — Pipeline stages: language detection, content extraction, heuristic quality filters (line length, symbol ratio, repetition), PII removal patterns.
3. **Hands-on** — Load sample court opinions. Implement each filter step by step: `langdetect` for language filtering, regex-based PII detection (SSNs, phone numbers), heuristic quality scores. Show before/after statistics at each stage.
4. **Exercises** — (a) Add a custom filter for legal boilerplate (standard headers that appear in every opinion from the same court). (b) Tune quality thresholds and measure impact on corpus size vs quality.

**Step 3: Create 02_deduplication.ipynb**

Notebook sections:
1. **Theory** — Exact vs fuzzy dedup. MinHash + LSH for near-duplicate detection. Why deduplication matters for training (memorization, benchmark contamination).
2. **Hands-on** — Implement exact dedup (hash-based). Then implement MinHash using `datasketch` library. Run on sample corpus, visualize duplicate clusters. Show examples of near-duplicates (same opinion published by different reporters).
3. **Exercises** — (a) Experiment with MinHash parameters (num_perm, threshold) and measure precision/recall of dedup. (b) Estimate dedup ratio for different legal document types.

**Step 4: Verify notebooks run**

```bash
cd notebooks/02-data-cleaning
jupyter nbconvert --to notebook --execute 01_quality_filtering.ipynb --ExecutePreprocessor.timeout=300
jupyter nbconvert --to notebook --execute 02_deduplication.ipynb --ExecutePreprocessor.timeout=300
```

**Step 5: Commit**

```bash
git add notebooks/02-data-cleaning/
git commit -m "feat(notebooks): add module 02 — data cleaning"
```

---

## Task 7: Notebook 03 — Tokenization

**Files:**
- Create: `notebooks/03-tokenization/01_bpe_from_scratch.ipynb`
- Create: `notebooks/03-tokenization/02_huggingface_tokenizers.ipynb`
- Modify: `notebooks/03-tokenization/README.md`

**Step 1: Update module README**

No prerequisites. Self-contained module.

**Step 2: Create 01_bpe_from_scratch.ipynb**

Notebook sections:
1. **Context** — Why tokenization is the hidden foundation. CoCounsel context: legal citations like "42 U.S.C. § 1983" tokenize poorly with general-purpose tokenizers, affecting model comprehension.
2. **Theory** — Character-level vs word-level vs subword. BPE algorithm step-by-step with visual merge tables.
3. **Hands-on** — Implement BPE from scratch in ~50 lines of Python. Train on a small legal text sample. Visualize the merge sequence. Encode/decode round-trip. Compare vocabulary at different merge counts (256, 1000, 5000 merges).
4. **Exercises** — (a) Modify the BPE implementation to handle Unicode properly. (b) Compare compression ratios on legal text vs general English text.

**Step 3: Create 02_huggingface_tokenizers.ipynb**

Notebook sections:
1. **Hands-on** — Load GPT-2, Llama, and BERT tokenizers from HuggingFace. Tokenize the same legal passages with each. Visualize token boundaries on legal text. Show: how "42 U.S.C. § 1983" tokenizes in each. Count tokens for a full court opinion in each tokenizer.
2. **Train a custom tokenizer** — Use `tokenizers` library to train a BPE tokenizer on the legal corpus from `datasets/sample/`. Compare its vocabulary against GPT-2's. Show improved tokenization of legal terms.
3. **Exercises** — (a) Compute the "fertility" (tokens per word) for legal vs general text across tokenizers. (b) Estimate cost implications: if a legal document is 20% more tokens than general text, what does that mean for API costs?

**Step 4: Verify notebooks run**

```bash
cd notebooks/03-tokenization
jupyter nbconvert --to notebook --execute 01_bpe_from_scratch.ipynb --ExecutePreprocessor.timeout=300
jupyter nbconvert --to notebook --execute 02_huggingface_tokenizers.ipynb --ExecutePreprocessor.timeout=300
```

**Step 5: Commit**

```bash
git add notebooks/03-tokenization/
git commit -m "feat(notebooks): add module 03 — tokenization"
```

---

## Task 8: Notebook 04 — Architecture

**Files:**
- Create: `notebooks/04-architecture/01_transformer_block.ipynb`
- Create: `notebooks/04-architecture/02_gpt_vs_llama.ipynb`
- Modify: `notebooks/04-architecture/README.md`

**Step 1: Update module README**

Prerequisites: Basic PyTorch familiarity. No dependency on prior modules.

**Step 2: Create 01_transformer_block.ipynb**

Notebook sections:
1. **Context** — The transformer is the architecture behind every modern LLM. CoCounsel context: understanding attention helps explain why models sometimes lose track of long legal arguments.
2. **Theory** — Self-attention (Q/K/V), multi-head attention, feed-forward network, layer normalization, residual connections. Diagrams for each.
3. **Hands-on** — Build a single transformer block in PyTorch from raw nn.Linear layers (no nn.TransformerEncoder). Implement scaled dot-product attention manually. Pass a sample sequence through. Visualize the attention weights as a heatmap. Show how attention distributes across a legal sentence.
4. **Exercises** — (a) Experiment with head count: how do attention patterns change with 1 vs 4 vs 8 heads? (b) Remove layer norm and observe what happens to attention weights.

**Step 3: Create 02_gpt_vs_llama.ipynb**

Notebook sections:
1. **Theory** — GPT architecture (learned positional embeddings, LayerNorm, vanilla multi-head attention) vs Llama (RoPE, RMSNorm, Grouped Query Attention, SwiGLU). Side-by-side comparison table.
2. **Hands-on** — Load GPT-2 small and a small Llama model. Inspect their architectures programmatically (`model.named_parameters()`, `model.config`). Run the same legal prompt through both. Extract and visualize attention patterns from intermediate layers. Compare parameter counts and memory footprint.
3. **Exercises** — (a) Implement RoPE from scratch and verify it matches the model's positional encoding. (b) Compare inference speed between GPT-2 and Llama on sequences of different lengths.

**Step 4: Verify notebooks run**

```bash
cd notebooks/04-architecture
jupyter nbconvert --to notebook --execute 01_transformer_block.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute 02_gpt_vs_llama.ipynb --ExecutePreprocessor.timeout=600
```

**Step 5: Commit**

```bash
git add notebooks/04-architecture/
git commit -m "feat(notebooks): add module 04 — architecture"
```

---

## Task 9: Notebook 05 — Text Generation

**Files:**
- Create: `notebooks/05-text-generation/01_decoding_strategies.ipynb`
- Modify: `notebooks/05-text-generation/README.md`

**Step 1: Update module README**

Prerequisites: Module 04 (understanding of transformer forward pass).

**Step 2: Create 01_decoding_strategies.ipynb**

Notebook sections:
1. **Context** — Generation strategy directly affects output quality. CoCounsel context: legal responses need deterministic, accurate text — not creative sampling. Understanding these trade-offs is essential for product configuration.
2. **Theory** — Autoregressive generation. Greedy decoding, beam search, top-k sampling, top-p (nucleus) sampling, temperature scaling. Diagrams showing probability distributions and how each strategy selects tokens.
3. **Hands-on** — Load a small pre-trained model (GPT-2 or SmolLM). Implement each strategy from scratch (no `model.generate()`):
   - **Greedy**: Always pick argmax. Show where it produces repetitive text.
   - **Beam search**: Maintain k beams. Show how it finds higher-probability sequences than greedy.
   - **Top-k**: Sample from top k tokens. Vary k and observe output diversity.
   - **Top-p**: Sample from smallest set summing to p. Compare against fixed top-k.
   - **Temperature**: Scale logits before softmax. Visualize the probability distribution at T=0.1, 0.5, 1.0, 2.0.
   Run all strategies on the same legal prompt ("The court held that"). Compare outputs side-by-side.
4. **Visualization** — For a single generation step, plot the full token probability distribution and highlight which tokens each strategy would select.
5. **Exercises** — (a) Combine top-k + top-p + temperature and find settings that produce good legal text. (b) Implement repetition penalty and show its effect on legal summarization.

**Step 3: Verify notebook runs**

```bash
cd notebooks/05-text-generation
jupyter nbconvert --to notebook --execute 01_decoding_strategies.ipynb --ExecutePreprocessor.timeout=600
```

**Step 4: Commit**

```bash
git add notebooks/05-text-generation/
git commit -m "feat(notebooks): add module 05 — text generation"
```

---

## Task 10: Notebook 06 — Supervised Fine-Tuning

**Files:**
- Create: `notebooks/06-sft/01_instruction_dataset.ipynb`
- Create: `notebooks/06-sft/02_sft_training.ipynb`
- Modify: `notebooks/06-sft/README.md`

**Step 1: Update module README**

Prerequisites: Modules 03 (tokenization) and 04 (architecture). Uses data from `datasets/sample/`.

**Step 2: Create 01_instruction_dataset.ipynb**

Notebook sections:
1. **Context** — SFT is how base models learn to follow instructions. CoCounsel context: the difference between a model that rambles about law and one that answers legal questions precisely is SFT.
2. **Theory** — Instruction tuning format (system/user/assistant), chat templates, loss masking (only compute loss on assistant tokens).
3. **Hands-on** — Take court opinions from `datasets/sample/court_opinions.jsonl`. Transform into instruction pairs: "Summarize this opinion" → summary, "What was the holding?" → extracted holding, "List the cited statutes" → citation list. Build 100–200 training examples. Format using chat template. Visualize which tokens are masked vs trained.
4. **Exercises** — (a) Create instruction pairs for a different task (e.g., "Is this opinion from a federal or state court?"). (b) Analyze the distribution of response lengths and discuss implications for training.

**Step 3: Create 02_sft_training.ipynb**

Notebook sections:
1. **Hands-on** — Load SmolLM-135M (or TinyLlama-1.1B if resources allow). Set up the SFT training loop manually (not using `trl.SFTTrainer` initially): tokenize, create DataLoader, forward pass, loss with masking, backward pass, optimizer step. Train for a few epochs on the small dataset. Plot training loss.
2. **Compare** — Run the same legal prompts before and after SFT. Show the behavior change: base model generates random legal text, SFT model follows the instruction format.
3. **Then with trl** — Redo the training using `trl.SFTTrainer` to show the standard tooling. Compare: same result, less code.
4. **Exercises** — (a) Experiment with learning rate and batch size. (b) Train for too many epochs and observe overfitting on the small dataset.

**Step 4: Verify notebooks run**

```bash
cd notebooks/06-sft
jupyter nbconvert --to notebook --execute 01_instruction_dataset.ipynb --ExecutePreprocessor.timeout=300
jupyter nbconvert --to notebook --execute 02_sft_training.ipynb --ExecutePreprocessor.timeout=900
```

Note: SFT training notebook may take 10–15 minutes on CPU.

**Step 5: Commit**

```bash
git add notebooks/06-sft/
git commit -m "feat(notebooks): add module 06 — supervised fine-tuning"
```

---

## Task 11: Notebook 07 — RLHF & Alignment

**Files:**
- Create: `notebooks/07-rlhf/01_reward_modeling.ipynb`
- Create: `notebooks/07-rlhf/02_dpo.ipynb`
- Modify: `notebooks/07-rlhf/README.md`

**Step 1: Update module README**

Prerequisites: Module 06 (SFT). Builds on the SFT-tuned model.

**Step 2: Create 01_reward_modeling.ipynb**

Notebook sections:
1. **Context** — RLHF aligns models with human preferences. CoCounsel context: a legal AI must prefer citing real cases over plausible-sounding fabrications. Alignment is how we encode that preference.
2. **Theory** — Preference data format (chosen/rejected pairs). Reward model architecture (LM with scalar head). Bradley-Terry model. PPO overview with annotated pseudocode (conceptual — not implemented end-to-end due to compute requirements). Verifiable rewards for tasks with ground truth.
3. **Hands-on** — Create a small preference dataset: for each legal question, write a good answer (with real citations) and a bad answer (with hallucinated citations). Train a tiny reward model on top of the SFT model. Visualize reward scores: does it rank good answers above bad?
4. **Exercises** — (a) Add a "verifiable reward" check: does the cited case actually exist in the corpus? (b) Analyze failure modes of the reward model.

**Step 3: Create 02_dpo.ipynb**

Notebook sections:
1. **Theory** — DPO as a simpler alternative to PPO. The DPO loss function derived from the RLHF objective. Why DPO is practical for small-scale alignment.
2. **Hands-on** — Use the same preference dataset. Implement DPO training using `trl.DPOTrainer`. Train the SFT model with DPO. Compare outputs: SFT-only vs DPO-aligned on tricky prompts (questions where the model might hallucinate citations or be overconfident).
3. **Exercises** — (a) Vary the DPO beta parameter and observe its effect on output style. (b) Create adversarial prompts that try to get the aligned model to hallucinate.

**Step 4: Verify notebooks run**

```bash
cd notebooks/07-rlhf
jupyter nbconvert --to notebook --execute 01_reward_modeling.ipynb --ExecutePreprocessor.timeout=900
jupyter nbconvert --to notebook --execute 02_dpo.ipynb --ExecutePreprocessor.timeout=900
```

**Step 5: Commit**

```bash
git add notebooks/07-rlhf/
git commit -m "feat(notebooks): add module 07 — RLHF and alignment"
```

---

## Task 12: Notebook 08 — Evaluation

**Files:**
- Create: `notebooks/08-evaluation/01_metrics.ipynb`
- Create: `notebooks/08-evaluation/02_eval_harness.ipynb`
- Modify: `notebooks/08-evaluation/README.md`

**Step 1: Update module README**

Prerequisites: Module 06 (for a model to evaluate). Concepts here feed directly into App Stage 5.

**Step 2: Create 01_metrics.ipynb**

Notebook sections:
1. **Context** — Evaluation is how you know if your model actually works. CoCounsel context: for legal AI, "works" means factually accurate, properly cited, and appropriately uncertain. Standard NLP metrics don't capture this.
2. **Theory** — Perplexity (measures language modeling quality), BLEU/ROUGE (measure overlap with reference text), their limitations. Task-specific metrics: citation accuracy, hallucination rate, refusal appropriateness.
3. **Hands-on** — Compute perplexity of the SFT model on held-out legal text. Compute ROUGE scores for legal summarization. Show where ROUGE fails: a summary with fabricated citations can score high on ROUGE if it matches reference structure. Implement a citation accuracy checker: extract cited cases from model output, verify against corpus.
4. **Exercises** — (a) Design a metric for "appropriate uncertainty" — does the model say "I'm not sure" when it should? (b) Compare metric scores across base, SFT, and DPO models.

**Step 3: Create 02_eval_harness.ipynb**

Notebook sections:
1. **Theory** — Evaluation harnesses (lm-evaluation-harness), benchmark design, LLM-as-judge pattern, human evaluation methodology, leaderboard gaming and Goodhart's law.
2. **Hands-on** — Build a minimal evaluation harness: define a set of 20 legal questions with ground-truth answers. Run the model, auto-score with the citation accuracy metric and ROUGE. Implement a simple LLM-as-judge: use a larger model (via API) to rate the smaller model's outputs on accuracy, helpfulness, and citation quality.
3. **Exercises** — (a) Add a new evaluation dimension (e.g., "conciseness") and modify the judge prompt. (b) Discuss: how would you prevent benchmark contamination if this eval set were public?

**Step 4: Verify notebooks run**

```bash
cd notebooks/08-evaluation
jupyter nbconvert --to notebook --execute 01_metrics.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute 02_eval_harness.ipynb --ExecutePreprocessor.timeout=600
```

**Step 5: Commit**

```bash
git add notebooks/08-evaluation/
git commit -m "feat(notebooks): add module 08 — evaluation"
```

---

## Task 13: Notebook 09 — Fine-Tuning Lab (Optional)

**Files:**
- Create: `notebooks/09-fine-tuning-lab/01_qlora_finetuning.ipynb`
- Modify: `notebooks/09-fine-tuning-lab/README.md`

**Step 1: Update module README**

Mark as **optional — requires GPU**. Prerequisites: Modules 06–07. This is a self-guided lab.

**Step 2: Create 01_qlora_finetuning.ipynb**

Notebook sections (structured as lab checkpoints):
1. **Checkpoint 1: Setup** — Load a 7B model (Llama-2-7B or Mistral-7B) with 4-bit quantization using `bitsandbytes`. Verify GPU memory usage. Explain quantization trade-offs (4-bit vs 8-bit, NF4 vs FP4).
2. **Checkpoint 2: LoRA Configuration** — Set up LoRA adapters using `peft`. Explain rank, alpha, target modules. Show parameter count: full model vs LoRA trainable parameters.
3. **Checkpoint 3: Training** — Fine-tune on the legal instruction dataset from module 06 using `trl.SFTTrainer` with QLoRA config. Train for 1–3 epochs. Monitor loss and GPU memory.
4. **Checkpoint 4: Evaluation** — Compare base model, QLoRA fine-tuned model on the eval harness from module 08. Merge adapters back into base model. Measure quality vs compute trade-off.
5. **Exercises** — (a) Experiment with LoRA rank (4, 16, 64) and measure quality impact. (b) Try different target modules (attention only vs attention + MLP).

**Step 3: Commit**

```bash
git add notebooks/09-fine-tuning-lab/
git commit -m "feat(notebooks): add module 09 — optional QLoRA fine-tuning lab"
```

---

## Task 14: App Stage 1 — Document Ingestion

**Files:**
- Create: `app/src/ingestion/chunker.ts`
- Create: `app/src/ingestion/embedder.ts`
- Create: `app/src/ingestion/store.ts`
- Create: `app/src/ingestion/ingest.ts`
- Create: `app/src/ingestion/index.ts`
- Test: `app/src/ingestion/__tests__/chunker.test.ts`
- Test: `app/src/ingestion/__tests__/store.test.ts`

**Step 1: Write failing test for chunker**

```typescript
// app/src/ingestion/__tests__/chunker.test.ts
import { describe, it, expect } from "vitest";
import { chunkDocument, ChunkStrategy } from "../chunker";

describe("chunkDocument", () => {
  it("splits by section headers in legal text", () => {
    const text = "I. BACKGROUND\nSome facts here.\nII. ANALYSIS\nLegal analysis.";
    const chunks = chunkDocument(text, { strategy: "section-aware" });
    expect(chunks).toHaveLength(2);
    expect(chunks[0].content).toContain("BACKGROUND");
    expect(chunks[1].content).toContain("ANALYSIS");
  });

  it("falls back to naive splitting for unstructured text", () => {
    const text = "A ".repeat(500);
    const chunks = chunkDocument(text, {
      strategy: "naive",
      maxTokens: 100,
    });
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach((c) => expect(c.content.length).toBeLessThanOrEqual(500));
  });

  it("preserves metadata through chunking", () => {
    const text = "I. BACKGROUND\nFacts.";
    const chunks = chunkDocument(text, {
      strategy: "section-aware",
      metadata: { caseId: "123", court: "SCOTUS" },
    });
    expect(chunks[0].metadata).toEqual(
      expect.objectContaining({ caseId: "123" })
    );
  });
});
```

**Step 2: Run test to verify it fails**

```bash
cd app && pnpm test -- src/ingestion/__tests__/chunker.test.ts
```

Expected: FAIL — module not found.

**Step 3: Implement chunker**

`app/src/ingestion/chunker.ts` — Export `chunkDocument(text, options)` that supports two strategies:
- `section-aware`: Split on legal section headers (Roman numerals, "Section", "Article"). Preserve header in chunk.
- `naive`: Split on sentence boundaries up to maxTokens per chunk.

Each chunk has `{ content, metadata, index }`.

**Step 4: Run test to verify it passes**

```bash
cd app && pnpm test -- src/ingestion/__tests__/chunker.test.ts
```

Expected: PASS.

**Step 5: Write failing test for store**

```typescript
// app/src/ingestion/__tests__/store.test.ts
import { describe, it, expect, beforeEach } from "vitest";
import { VectorStore } from "../store";

describe("VectorStore", () => {
  let store: VectorStore;

  beforeEach(() => {
    store = new VectorStore({ inMemory: true });
  });

  it("stores and retrieves documents by similarity", async () => {
    await store.add([
      { content: "The Fourth Amendment protects against unreasonable searches", metadata: { id: "1" } },
      { content: "Contract law requires offer and acceptance", metadata: { id: "2" } },
    ]);
    const results = await store.query("search and seizure", { topK: 1 });
    expect(results[0].metadata.id).toBe("1");
  });

  it("returns empty array when store is empty", async () => {
    const results = await store.query("anything", { topK: 5 });
    expect(results).toEqual([]);
  });
});
```

**Step 6: Implement store**

`app/src/ingestion/store.ts` — `VectorStore` class wrapping ChromaDB with an `inMemory` option for testing.

**Step 7: Run all ingestion tests**

```bash
cd app && pnpm test -- src/ingestion/
```

Expected: All PASS.

**Step 8: Create ingest.ts entry point**

Wire chunker + embedder + store into a pipeline: `ingestDocuments(jsonlPath, options)` reads a JSONL file, chunks each document, embeds, and stores.

**Step 9: Commit and tag**

```bash
git add app/src/ingestion/
git commit -m "feat(app): add document ingestion with section-aware chunking"
git tag stage-1-ingestion
```

---

## Task 15: App Stage 2 — RAG Pipeline

**Files:**
- Create: `app/src/rag/retriever.ts`
- Create: `app/src/rag/promptBuilder.ts`
- Create: `app/src/rag/generate.ts`
- Create: `app/src/rag/index.ts`
- Test: `app/src/rag/__tests__/retriever.test.ts`
- Test: `app/src/rag/__tests__/promptBuilder.test.ts`
- Test: `app/src/rag/__tests__/generate.test.ts`

**Step 1: Write failing test for retriever**

```typescript
// app/src/rag/__tests__/retriever.test.ts
import { describe, it, expect, vi } from "vitest";
import { retrieve } from "../retriever";

describe("retrieve", () => {
  it("returns ranked chunks with relevance scores", async () => {
    const mockStore = {
      query: vi.fn().mockResolvedValue([
        { content: "relevant chunk", metadata: { caseId: "1" }, score: 0.9 },
      ]),
    };
    const results = await retrieve("search query", mockStore, { topK: 5 });
    expect(results).toHaveLength(1);
    expect(results[0].score).toBeGreaterThan(0);
  });
});
```

**Step 2: Run test to verify it fails, then implement retriever**

`app/src/rag/retriever.ts` — Thin wrapper over VectorStore.query that adds re-ranking logic and score thresholding.

**Step 3: Write failing test for promptBuilder**

```typescript
// app/src/rag/__tests__/promptBuilder.test.ts
import { describe, it, expect } from "vitest";
import { buildPrompt } from "../promptBuilder";

describe("buildPrompt", () => {
  it("includes retrieved context with source citations", () => {
    const chunks = [
      { content: "The court held X", metadata: { caseId: "Smith v Jones", court: "SCOTUS" } },
    ];
    const prompt = buildPrompt("What did the court decide?", chunks);
    expect(prompt).toContain("Smith v Jones");
    expect(prompt).toContain("The court held X");
    expect(prompt).toContain("What did the court decide?");
  });

  it("instructs the model to cite sources and refuse when unsure", () => {
    const prompt = buildPrompt("question", []);
    expect(prompt.toLowerCase()).toContain("cite");
    expect(prompt.toLowerCase()).toMatch(/cannot|insufficient|unable/);
  });
});
```

**Step 4: Implement promptBuilder**

`app/src/rag/promptBuilder.ts` — Constructs a system prompt + user prompt with retrieved context formatted as numbered sources. Includes instructions for citation and refusal behavior.

**Step 5: Write failing test for generate**

```typescript
// app/src/rag/__tests__/generate.test.ts
import { describe, it, expect, vi } from "vitest";
import { generateAnswer } from "../generate";

describe("generateAnswer", () => {
  it("returns an answer with cited sources", async () => {
    const mockProvider = {
      generate: vi.fn().mockResolvedValue({
        text: "Based on Smith v Jones [1], the court held X.",
      }),
    };
    const result = await generateAnswer(
      "What happened?",
      [{ content: "The court held X", metadata: { caseId: "Smith v Jones" } }],
      { provider: mockProvider }
    );
    expect(result.text).toContain("Smith v Jones");
    expect(result.sources).toBeDefined();
  });
});
```

**Step 6: Implement generate**

`app/src/rag/generate.ts` — Uses Vercel AI SDK (`ai` package) with configurable provider. Calls `generateText()` with the built prompt. Extracts cited sources from the response.

**Step 7: Run all RAG tests**

```bash
cd app && pnpm test -- src/rag/
```

Expected: All PASS.

**Step 8: Commit and tag**

```bash
git add app/src/rag/
git commit -m "feat(app): add RAG pipeline with citation-aware prompting"
git tag stage-2-rag
```

---

## Task 16: App Stage 3 — Citation Grounding & Hallucination Detection

**Files:**
- Create: `app/src/verification/citationExtractor.ts`
- Create: `app/src/verification/groundingChecker.ts`
- Create: `app/src/verification/index.ts`
- Test: `app/src/verification/__tests__/citationExtractor.test.ts`
- Test: `app/src/verification/__tests__/groundingChecker.test.ts`

**Step 1: Write failing test for citationExtractor**

```typescript
// app/src/verification/__tests__/citationExtractor.test.ts
import { describe, it, expect } from "vitest";
import { extractCitations } from "../citationExtractor";

describe("extractCitations", () => {
  it("extracts case citations from text", () => {
    const text = "As held in Smith v. Jones, 550 U.S. 544 (2007), the standard requires...";
    const citations = extractCitations(text);
    expect(citations).toContainEqual(
      expect.objectContaining({
        caseName: "Smith v. Jones",
        reporter: "U.S.",
        volume: "550",
        page: "544",
      })
    );
  });

  it("extracts statute citations", () => {
    const text = "Under 42 U.S.C. § 1983, a plaintiff may...";
    const citations = extractCitations(text);
    expect(citations).toContainEqual(
      expect.objectContaining({
        type: "statute",
        title: "42",
        section: "1983",
      })
    );
  });

  it("returns empty array for text without citations", () => {
    expect(extractCitations("No legal citations here.")).toEqual([]);
  });
});
```

**Step 2: Implement citationExtractor**

`app/src/verification/citationExtractor.ts` — Regex-based extraction of case citations (reporter format) and statute citations (U.S.C. format).

**Step 3: Write failing test for groundingChecker**

```typescript
// app/src/verification/__tests__/groundingChecker.test.ts
import { describe, it, expect, vi } from "vitest";
import { checkGrounding } from "../groundingChecker";

describe("checkGrounding", () => {
  it("marks citations found in corpus as grounded", async () => {
    const mockStore = {
      query: vi.fn().mockResolvedValue([
        { content: "Smith v. Jones, 550 U.S. 544", score: 0.95 },
      ]),
    };
    const result = await checkGrounding(
      "As held in Smith v. Jones, 550 U.S. 544 (2007)...",
      mockStore
    );
    expect(result.citations[0].grounded).toBe(true);
  });

  it("flags citations not found in corpus as unverified", async () => {
    const mockStore = {
      query: vi.fn().mockResolvedValue([]),
    };
    const result = await checkGrounding(
      "As held in Fake v. Case, 999 U.S. 1 (2099)...",
      mockStore
    );
    expect(result.citations[0].grounded).toBe(false);
    expect(result.hasUnverifiedCitations).toBe(true);
  });
});
```

**Step 4: Implement groundingChecker**

`app/src/verification/groundingChecker.ts` — Extracts citations from text, queries the vector store for each, marks each as grounded or unverified based on similarity score threshold.

**Step 5: Run all verification tests**

```bash
cd app && pnpm test -- src/verification/
```

Expected: All PASS.

**Step 6: Commit and tag**

```bash
git add app/src/verification/
git commit -m "feat(app): add citation grounding and hallucination detection"
git tag stage-3-verification
```

---

## Task 17: App Stage 4 — Conversation & Memory

**Files:**
- Create: `app/src/conversation/history.ts`
- Create: `app/src/conversation/summarizer.ts`
- Create: `app/src/conversation/manager.ts`
- Create: `app/src/conversation/index.ts`
- Test: `app/src/conversation/__tests__/history.test.ts`
- Test: `app/src/conversation/__tests__/manager.test.ts`

**Step 1: Write failing test for history**

```typescript
// app/src/conversation/__tests__/history.test.ts
import { describe, it, expect } from "vitest";
import { ConversationHistory } from "../history";

describe("ConversationHistory", () => {
  it("tracks messages in order", () => {
    const history = new ConversationHistory();
    history.add({ role: "user", content: "What is habeas corpus?" });
    history.add({ role: "assistant", content: "Habeas corpus is..." });
    expect(history.messages).toHaveLength(2);
    expect(history.messages[0].role).toBe("user");
  });

  it("estimates token count", () => {
    const history = new ConversationHistory();
    history.add({ role: "user", content: "Short question" });
    expect(history.estimatedTokens).toBeGreaterThan(0);
  });
});
```

**Step 2: Implement history, then run test**

`app/src/conversation/history.ts` — `ConversationHistory` class that stores messages and estimates token count using a word-based heuristic (words * 1.3).

**Step 3: Write failing test for manager**

```typescript
// app/src/conversation/__tests__/manager.test.ts
import { describe, it, expect, vi } from "vitest";
import { ConversationManager } from "../manager";

describe("ConversationManager", () => {
  it("maintains context within token budget", async () => {
    const manager = new ConversationManager({ maxTokens: 100 });
    // Add enough messages to exceed budget
    for (let i = 0; i < 20; i++) {
      manager.addMessage({ role: "user", content: `Question ${i} about legal precedent and case law` });
      manager.addMessage({ role: "assistant", content: `Answer ${i} with detailed legal analysis` });
    }
    const context = await manager.getContext();
    expect(context.estimatedTokens).toBeLessThanOrEqual(100);
    // Most recent messages should be preserved
    expect(context.messages.at(-1)?.content).toContain("19");
  });

  it("includes system message in every context", async () => {
    const manager = new ConversationManager({
      maxTokens: 500,
      systemMessage: "You are a legal assistant.",
    });
    manager.addMessage({ role: "user", content: "Hello" });
    const context = await manager.getContext();
    expect(context.messages[0].content).toContain("legal assistant");
  });
});
```

**Step 4: Implement manager and summarizer**

`app/src/conversation/manager.ts` — Manages token budget by truncating older messages and optionally summarizing them using `summarizer.ts`.

`app/src/conversation/summarizer.ts` — Summarizes a batch of older messages into a single condensed message to preserve context while freeing token budget.

**Step 5: Run all conversation tests**

```bash
cd app && pnpm test -- src/conversation/
```

Expected: All PASS.

**Step 6: Commit and tag**

```bash
git add app/src/conversation/
git commit -m "feat(app): add conversation management with token budgeting"
git tag stage-4-conversation
```

---

## Task 18: App Stage 5 — Evaluation & Guardrails

**Files:**
- Create: `app/src/guardrails/inputFilter.ts`
- Create: `app/src/guardrails/outputFilter.ts`
- Create: `app/src/guardrails/index.ts`
- Create: `app/src/eval/metrics.ts`
- Create: `app/src/eval/runner.ts`
- Create: `app/src/eval/index.ts`
- Test: `app/src/guardrails/__tests__/inputFilter.test.ts`
- Test: `app/src/guardrails/__tests__/outputFilter.test.ts`
- Test: `app/src/eval/__tests__/metrics.test.ts`

**Step 1: Write failing test for inputFilter**

```typescript
// app/src/guardrails/__tests__/inputFilter.test.ts
import { describe, it, expect } from "vitest";
import { filterInput } from "../inputFilter";

describe("filterInput", () => {
  it("detects PII in user input", () => {
    const result = filterInput("Find cases about John Smith, SSN 123-45-6789");
    expect(result.containsPII).toBe(true);
    expect(result.piiTypes).toContain("ssn");
  });

  it("detects out-of-scope requests", () => {
    const result = filterInput("Write me a poem about flowers");
    expect(result.inScope).toBe(false);
  });

  it("passes clean legal queries", () => {
    const result = filterInput("What is the standard for summary judgment?");
    expect(result.containsPII).toBe(false);
    expect(result.inScope).toBe(true);
  });
});
```

**Step 2: Implement inputFilter, run test**

**Step 3: Write failing test for outputFilter**

```typescript
// app/src/guardrails/__tests__/outputFilter.test.ts
import { describe, it, expect } from "vitest";
import { filterOutput } from "../outputFilter";

describe("filterOutput", () => {
  it("flags responses with no citations on factual claims", () => {
    const result = filterOutput(
      "The Supreme Court ruled that this is unconstitutional.",
      { requireCitations: true }
    );
    expect(result.warnings).toContain("factual_claim_without_citation");
  });

  it("flags responses that give legal advice", () => {
    const result = filterOutput(
      "You should file a motion to dismiss immediately."
    );
    expect(result.warnings).toContain("direct_legal_advice");
  });

  it("passes well-cited informational responses", () => {
    const result = filterOutput(
      "In Smith v. Jones, 550 U.S. 544 (2007), the Court held that..."
    );
    expect(result.warnings).toHaveLength(0);
  });
});
```

**Step 4: Implement outputFilter, run test**

**Step 5: Write failing test for eval metrics**

```typescript
// app/src/eval/__tests__/metrics.test.ts
import { describe, it, expect } from "vitest";
import { citationAccuracy, hallucinationRate } from "../metrics";

describe("citationAccuracy", () => {
  it("computes ratio of grounded citations", () => {
    const groundingResult = {
      citations: [
        { text: "Smith v Jones", grounded: true },
        { text: "Fake v Case", grounded: false },
      ],
      hasUnverifiedCitations: true,
    };
    expect(citationAccuracy(groundingResult)).toBe(0.5);
  });
});

describe("hallucinationRate", () => {
  it("computes ratio of ungrounded citations", () => {
    const groundingResult = {
      citations: [
        { text: "A", grounded: true },
        { text: "B", grounded: false },
        { text: "C", grounded: false },
      ],
      hasUnverifiedCitations: true,
    };
    expect(hallucinationRate(groundingResult)).toBeCloseTo(0.667, 2);
  });
});
```

**Step 6: Implement metrics and eval runner**

`app/src/eval/metrics.ts` — Citation accuracy and hallucination rate computed from grounding results.

`app/src/eval/runner.ts` — Runs a set of test cases through the full pipeline (RAG → generate → verify → score), outputs a results report.

**Step 7: Run all Stage 5 tests**

```bash
cd app && pnpm test -- src/guardrails/ src/eval/
```

Expected: All PASS.

**Step 8: Run full test suite with coverage**

```bash
cd app && pnpm test:coverage
```

Expected: All tests pass, coverage meets 80% threshold.

**Step 9: Commit and tag**

```bash
git add app/src/guardrails/ app/src/eval/
git commit -m "feat(app): add evaluation metrics and input/output guardrails"
git tag stage-5-eval-guardrails
```

---

## Task 19: Integration — Wire Full Pipeline

**Files:**
- Create: `app/src/pipeline.ts`
- Create: `app/src/index.ts`
- Test: `app/src/pipeline.test.ts`

**Step 1: Write failing integration test**

```typescript
// app/src/pipeline.test.ts
import { describe, it, expect } from "vitest";
import { LegalAIPipeline } from "./pipeline";

describe("LegalAIPipeline", () => {
  it("processes a legal question through the full pipeline", async () => {
    const pipeline = await LegalAIPipeline.create({ inMemory: true });
    // Ingest sample data
    await pipeline.ingest("../../datasets/sample/court_opinions.jsonl");
    // Ask a question
    const result = await pipeline.ask("What is the standard for summary judgment?");
    expect(result.answer).toBeDefined();
    expect(result.groundingReport).toBeDefined();
    expect(result.guardrailReport).toBeDefined();
  });
});
```

**Step 2: Implement pipeline.ts**

Wire together: ingestion → retrieval → prompt building → generation → citation grounding → output filtering → metrics. Single `LegalAIPipeline` class with `ingest()` and `ask()` methods.

**Step 3: Create index.ts CLI entry point**

Minimal CLI that loads data and accepts questions in a REPL loop using Node readline.

**Step 4: Run full test suite**

```bash
cd app && pnpm test:coverage
```

Expected: All tests pass, coverage ≥ 80%.

**Step 5: Commit**

```bash
git add app/src/pipeline.ts app/src/index.ts app/src/pipeline.test.ts
git commit -m "feat(app): wire full legal-AI pipeline with CLI entry point"
```

---

## Task 20: Final — Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Create: `README.md`

**Step 1: Update CLAUDE.md with final commands**

Add all verified commands: how to run notebooks, how to run the app, how to run tests, how to fetch fresh datasets.

**Step 2: Create root README.md**

Project overview, learning path, prerequisites (Python 3.11+, Node 20+, pnpm), quickstart for both notebooks and app, links to each module README.

**Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: add README and update CLAUDE.md with final project docs"
```
