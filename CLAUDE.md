# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A collection of neural network implementations focused on Transformer-based models. Currently contains two projects: a GPT-2 training implementation and a Shakespeare text generator.

## Running GPT-2 Training

```bash
cd gpt2
bash run.sh          # Auto-detects GPUs: single-GPU or multi-GPU via torchrun
```

`run.sh` sets `HF_ENDPOINT` for Hugging Face mirroring and `OMP_NUM_THREADS=8`. It calls `python3 gpt2.py` for single-GPU or `torchrun --nproc_per_node=$N gpt2.py` for multi-GPU.

### Dependencies

```bash
pip install transformers datasets numpy
# PyTorch with CUDA must be pre-installed
```

## Architecture

### `gpt2/gpt2.py`

Single-file training script (~346 lines) implementing GPT-2 from scratch:

- **Model**: 12-layer decoder-only transformer, 768d, 12 heads, 1024 context, vocab=50257 (tied input/output embeddings)
- **Data**: Streams from Hugging Face FineWeb (10BT sample) via `FineWebDataset` (IterableDataset)
- **Training loop**:
  - Gradient accumulation targeting 2^19 tokens per optimizer step
  - Cosine LR schedule with warmup (1% of ~76k total steps), peak LR 6e-4
  - AdamW with fused kernels, bfloat16 autocast, `torch.compile`
  - DDP support — check `ddp` flag and `dist.init_process_group` for distributed logic
  - Checkpoint saves to `./checkpoint.pt`

### `gpt2/auxiliary.py`

Standalone `generate()` function — temperature scaling + top-k (k=40) sampling. Import separately for inference.

### `shakespeare/net.ipynb`

Educational decoder-only transformer trained on tiny Shakespeare (Karpathy's char-rnn dataset). Self-contained notebook.

## Key Design Notes

- Batch size is estimated dynamically based on available GPU memory (see `estimate_batch_size()`)
- Logits tensor is the memory bottleneck at large batch sizes (B×T×50257 elements)
- `benchmark.md` tracks throughput history (v0 baseline 5.6k tok/s → H100 67k tok/s)
