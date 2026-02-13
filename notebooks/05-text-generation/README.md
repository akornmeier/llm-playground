# Module 05: Text Generation

This module covers how LLMs generate text token by token. You will implement every major decoding strategy from scratch -- greedy decoding, beam search, temperature scaling, top-k sampling, and top-p (nucleus) sampling -- and compare their outputs on legal prompts.

**CoCounsel context:** Legal responses need deterministic, accurate text. Choosing the right generation strategy directly affects output quality and reliability. A contract clause that hallucinates a nonexistent statute is worse than useless -- it is dangerous. Understanding how each decoding strategy trades off diversity against precision lets you pick the right configuration for each legal task.

## What You Will Learn

1. **Autoregressive generation** -- how models produce one token at a time, feeding each output back as the next input.
2. **Greedy decoding** -- always picking the highest-probability token, and why this leads to repetitive loops.
3. **Beam search** -- maintaining multiple candidate sequences to find higher-probability completions.
4. **Temperature scaling** -- controlling the sharpness of the probability distribution before sampling.
5. **Top-k sampling** -- restricting the candidate pool to the k most likely tokens.
6. **Top-p (nucleus) sampling** -- dynamically sizing the candidate pool based on cumulative probability.
7. **Comparing strategies** -- side-by-side evaluation on legal prompts with visualizations.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_decoding_strategies.ipynb` | Implement all decoding strategies from scratch and compare outputs on legal text |

## Prerequisites

- **Module 04** (Architecture) -- understanding the transformer forward pass is essential. You need to know that a model takes a token sequence and produces logits (a score for every token in the vocabulary) at each position.
