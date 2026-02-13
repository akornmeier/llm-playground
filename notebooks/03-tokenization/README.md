# Module 03: Tokenization

Tokenization is the process of converting raw text into the integer tokens that
large language models actually process. Every LLM -- GPT, LLaMA, Claude -- has a
tokenizer sitting between the user's text and the model's embedding layer.
Understanding how tokenization works is essential for anyone building with or on
top of LLMs.

In this module, learners build a Byte Pair Encoding (BPE) tokenizer from scratch
and then explore the production tokenizers used by real models through the
HuggingFace ecosystem.

## CoCounsel Context

Legal text tokenizes differently than general English. Citations like
"42 U.S.C. § 1983" or "F. Supp. 3d" are sliced into many small tokens by
general-purpose tokenizers, which affects both model comprehension and API costs.
A court opinion that reads as 2,000 words of plain English might consume 20%
more tokens than an equivalent amount of general prose. Understanding this
phenomenon -- and knowing how to measure and mitigate it -- is a practical skill
for legal AI engineers.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_bpe_from_scratch.ipynb` | Implement Byte Pair Encoding in ~50 lines of Python. Train on a legal corpus and visualize how subword vocabularies grow. |
| 2 | `02_huggingface_tokenizers.ipynb` | Load GPT-2 and LLaMA tokenizers, compare how they handle legal text, compute token fertility, and train a custom legal tokenizer. |

## Prerequisites

None. This module is self-contained.
