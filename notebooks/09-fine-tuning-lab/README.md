# Module 09: Fine-Tuning Lab

**Optional -- requires GPU**

This is a self-guided lab for QLoRA fine-tuning of a larger model on legal data. Learners work through checkpoints at their own pace. Each checkpoint builds on the previous one, but you can stop at any checkpoint and resume later.

QLoRA (Quantized Low-Rank Adaptation) combines 4-bit quantization with LoRA adapters to fine-tune large models on consumer hardware. You will apply this technique to a 7B-parameter model, adapting it to the legal domain using instruction data from Module 06.

## Prerequisites

- **Modules 06-07**: You should be comfortable with instruction dataset construction (Module 06) and alignment training concepts (Module 07) before starting this lab.
- **Module 08** (recommended): The evaluation checkpoint uses a simplified version of the harness from Module 08.

## Requirements

- **NVIDIA GPU with at least 16 GB VRAM** (e.g., RTX 4090, A100, or cloud GPU)
- CUDA toolkit and drivers compatible with PyTorch
- Python packages: `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, `accelerate`

This lab will **not** run on CPU-only machines or Apple Silicon. If you do not have access to a suitable GPU, consider using a cloud provider (e.g., Lambda Labs, RunPod, or Google Colab Pro with an A100 runtime).

## Checkpoints

| # | Checkpoint | What you will do |
|---|-----------|-----------------|
| 1 | Setup and Quantization | Load a 7B model in 4-bit precision, measure GPU memory |
| 2 | LoRA Configuration | Attach low-rank adapters, inspect trainable parameter count |
| 3 | Training | Fine-tune on legal instruction data with SFTTrainer |
| 4 | Evaluation and Merging | Compare base vs fine-tuned outputs, merge adapters |
| 5 | Experiments | Vary rank, target modules, and epochs; record results |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_qlora_finetuning.ipynb` | Complete QLoRA fine-tuning lab (all 5 checkpoints) |
