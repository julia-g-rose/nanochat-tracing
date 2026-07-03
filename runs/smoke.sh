#!/bin/bash

# Quick END-TO-END smoke test of the FULL nanochat pipeline:
#   tokenizer -> base pretrain -> base eval -> SFT -> chat eval
# ...but with tiny settings (depth=4 model, ~30 steps) so it finishes in a
# few minutes instead of hours. Use this to validate the CoreWeave setup and
# the whole training path. The resulting model is NOT useful — it's a smoke
# test, not a real run. For a real model use runs/speedrun.sh.
#
# Assumes dependencies are already installed (true in the Docker image, where
# /opt/venv is on PATH). To run on a bare node, first set up the venv the same
# way runs/speedrun.sh does (uv venv && uv sync --extra gpu && source .venv/...).

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# wandb run name. "dummy" (the default) disables wandb logging.
WANDB_RUN="${WANDB_RUN:-dummy}"
# GPUs per node.
NPROC="${NPROC_PER_NODE:-8}"

# -----------------------------------------------------------------------------
# Tokenizer
# 4 shards is plenty and gives base_train the >=2 parquet files it needs for the
# train/val split. Train the tokenizer on a small slice for speed (the real run
# uses 2B chars).
python -m nanochat.dataset -n 4
python -m scripts.tok_train --max-chars=200000000
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) — tiny: depth 4, 30 steps.
# total_batch_size must be a multiple of device_batch_size*max_seq_len*NPROC
# (= 1*512*8 = 4096), so grad-accum = 1 here.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
  --depth=4 --max-seq-len=512 --device-batch-size=1 --total-batch-size=4096 \
  --num-iterations=30 --eval-every=-1 --core-metric-every=-1 \
  --sample-every=-1 --eval-tokens=4096 --grad-metrics-every=5 --fp8 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Base eval: CORE + BPB + sampling. num_fewshot is now capped to the available
# pool in core_eval, so a small --max-per-task degrades gracefully (fewer-shot)
# instead of crashing. This also exercises the CORE per-example wandb table.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- \
  --eval core,bpb,sample --max-per-task=16 --split-tokens=16384 --device-batch-size=8 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# SFT — teach chat special tokens etc. Small identity dataset + few steps.
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- \
  --num-iterations=30 --eval-every=-1 --chatcore-every=-1 \
  --mmlu-epochs=1 --gsm8k-epochs=1 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Chat eval on the SFT model — only a few problems per task.
# --max-problems=16 (x6 tasks, gathered across ranks) gives a richer eval table.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- \
  -i sft --max-problems=16 --run=$WANDB_RUN

echo "✅ Smoke test complete: full pipeline ran end-to-end."
