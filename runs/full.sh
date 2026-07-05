#!/bin/bash

# Full (real-scale) nanochat training run: tokenizer -> base pretrain -> base
# eval -> SFT -> chat eval, at real settings. Unlike runs/smoke.sh this trains a
# usable model. Container-ready (assumes deps already installed, as in the Docker
# image); to run on a bare node, set up the venv first like runs/speedrun.sh does.
#
# Configure via env vars:
#   DEPTH            model depth (default 24; use 12 for a smaller/cheaper run)
#   NPROC_PER_NODE   GPUs to use (default 8)
#   NUM_SHARDS       pretraining shards to download (default 170; ~24 is plenty for d12)
#   WANDB_RUN        run name ("dummy" disables wandb logging)

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

WANDB_RUN="${WANDB_RUN:-dummy}"
NPROC="${NPROC_PER_NODE:-8}"
DEPTH="${DEPTH:-24}"
NUM_SHARDS="${NUM_SHARDS:-170}"

# -----------------------------------------------------------------------------
# Tokenizer + data
# Download the first 8 shards for tokenizer training, kick off the rest in the
# background while the tokenizer trains.
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n "$NUM_SHARDS" &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# -----------------------------------------------------------------------------
# Base pretrain. --target-param-data-ratio=8 sets a real (slightly-undertrained,
# GPT-2-beating) token budget; --fp8 uses torchao float8 on H100.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
  --depth=$DEPTH --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN

# Base eval: full CORE + BPB + samples (no --max-per-task cap -> full test sets).
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- \
  --device-batch-size=16 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# SFT + chat eval
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- \
  --device-batch-size=16 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- \
  -i sft --run=$WANDB_RUN

echo "✅ Full run complete (depth=$DEPTH, nproc=$NPROC)."
