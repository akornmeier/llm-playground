# Module 07: RLHF and Alignment

This module covers RLHF (Reinforcement Learning from Human Feedback) and alignment -- how we teach models to prefer good outputs over bad ones. Base language models predict the next token. SFT (Module 06) teaches them to follow instructions. But following instructions is not enough: the model also needs to know *which* response is better when multiple valid completions exist. Alignment is the process of encoding those preferences into the model.

Learners build a preference dataset of chosen/rejected response pairs, train a reward model that scores response quality, and implement Direct Preference Optimization (DPO) as a simpler alternative to full RLHF with PPO.

**CoCounsel context:** A legal AI must prefer citing real cases over plausible-sounding fabrications. It must prefer appropriate hedging ("the court may find...") over overconfident claims ("the court will certainly rule..."). It must prefer accurate statements of law over subtly incorrect ones. Alignment is how we encode these preferences. Without it, an instruction-following model might produce fluent, well-formatted answers that are confidently wrong -- the worst kind of failure for a legal tool.

## Notebooks

| Notebook | Topic |
|----------|-------|
| `01_reward_modeling.ipynb` | Build a preference dataset, implement a reward model with a Bradley-Terry loss, train it to score legal responses, and understand PPO conceptually. |
| `02_dpo.ipynb` | Implement DPO as a simpler alternative to PPO. Train with `trl.DPOTrainer`, compare aligned vs unaligned outputs, and explore the beta parameter. |

## Prerequisites

- **Module 06 (Supervised Fine-Tuning)** -- You need to understand SFT training, chat templates, and how models learn to follow instructions before learning how to align their preferences.
