# Module 08: Evaluation

This module covers LLM evaluation -- metrics, benchmarks, and evaluation methodology. Learners build an evaluation harness for legal AI that combines traditional NLP metrics, domain-specific metrics, and LLM-as-judge patterns into a systematic testing framework.

**CoCounsel context:** For legal AI, "works" means factually accurate, properly cited, and appropriately uncertain. Standard NLP metrics like BLEU and ROUGE measure surface-level text overlap, not correctness. A summary with fabricated citations can score high on ROUGE if it matches the reference structure. A factually wrong answer with good word overlap scores well on BLEU. Legal AI evaluation requires domain-specific metrics -- citation accuracy, hallucination detection, and uncertainty calibration -- layered on top of standard benchmarks.

The evaluation patterns built in this module feed directly into App Stage 5 (Evaluation and Guardrails), where they are integrated into the CoCounsel application as runtime quality checks.

## Notebooks

| Notebook | Topic |
|----------|-------|
| `01_metrics.ipynb` | Traditional NLP metrics (perplexity, BLEU, ROUGE), where they break down for legal AI, and domain-specific metrics: citation accuracy and hallucination detection. |
| `02_eval_harness.ipynb` | Build a systematic evaluation harness with legal test cases, automated scoring, LLM-as-judge patterns, and results visualization. |

## Prerequisites

- **Module 06 (Supervised Fine-Tuning)** -- You need a model (or at least an understanding of model outputs) to evaluate. The metrics here assume you have generated text to score.
