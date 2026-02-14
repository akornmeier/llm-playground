# Module 06: Supervised Fine-Tuning (SFT)

Supervised fine-tuning is the process of teaching a base language model to follow instructions. A base model is trained only to predict the next token -- it has no concept of "question" and "answer." SFT bridges that gap by training the model on instruction-response pairs so it learns the pattern: given an instruction, produce a helpful response.

**CoCounsel context:** The difference between a model that rambles about law and one that answers legal questions precisely is SFT. A base model might continue a legal paragraph with plausible-sounding but directionless text. After SFT, the same model can summarize an opinion, extract holdings, or list citations on command -- because it has learned to follow the instruction format.

In this module, learners build an instruction dataset from real court opinions and then train a small model (SmolLM-135M) to follow legal instructions. The training loop is implemented from scratch in PyTorch before being compared with the standard `trl` library.

## Notebooks

| Notebook | Topic |
|----------|-------|
| `01_instruction_dataset.ipynb` | Build an instruction-tuning dataset from court opinions. Covers chat templates, loss masking, and dataset statistics. |
| `02_sft_training.ipynb` | Train a model with SFT. Manual PyTorch training loop, before/after comparison, and `trl.SFTTrainer` equivalent. |

## Prerequisites

- **Module 03 (Tokenization)** -- You need to understand how text becomes token IDs, including special tokens and chat templates.
- **Module 04 (Architecture)** -- You need to understand the transformer forward pass, logits, and how cross-entropy loss works.
