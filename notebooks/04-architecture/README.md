# Module 04: Transformer Architecture

This module dives into the internals of the transformer architecture -- the
engine behind every modern large language model. You will build a transformer
block from scratch in PyTorch, visualize attention patterns on legal text, and
compare the architectural choices that distinguish GPT-2 from Llama.

**CoCounsel context:** Understanding how attention works helps explain why models
sometimes lose track of long legal arguments or miss citations buried in lengthy
briefs. When you know that attention scores decay over distance, or that a
model's "memory" is a fixed-size context window, you can design better prompts
and anticipate failure modes.

## Notebooks

| # | Notebook | What You Build |
|---|----------|---------------|
| 1 | `01_transformer_block.ipynb` | Scaled dot-product attention, multi-head attention, feed-forward network, and a full transformer block -- all from scratch in PyTorch. Visualize attention weights on legal text. |
| 2 | `02_gpt_vs_llama.ipynb` | Side-by-side comparison of GPT-2 and Llama architectures. Load GPT-2, inspect internals, implement RoPE and RMSNorm, and understand grouped-query attention. |

## Prerequisites

- Basic PyTorch familiarity (tensors, `nn.Module`, `forward()`)
- Completion of Modules 01-03 (data collection, cleaning, tokenization)
