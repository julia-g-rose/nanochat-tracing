#!/usr/bin/env bash

# Seeded depth-24 replicate at the ratio 20-24 compute frontier recommended in:
# https://wandb.ai/wandb/nanochat-with-aria/reports/Nanochat-with-ARIA-%E2%80%94-Project-Summary-(September-2026)--VmlldzoxNzkxNzI1Nw==
#
# The first trial defaults to ratio 20 and seed 43. After reviewing it, run the
# matched ratio-24 trial with RATIO=24 and/or add seeds 44+ for replication.
# Prerequisite: the existing tokenizer and ClimbMix shards must be available
# under NANOCHAT_BASE_DIR. This entrypoint runs pretraining only.

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC="${NPROC_PER_NODE:-1}"
RATIO="${RATIO:-20}"
SEED="${SEED:-43}"

case "$RATIO" in
  20|24) ;;
  *) echo "RATIO must be 20 or 24 (got: $RATIO)" >&2; exit 2 ;;
esac

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer (got: $SEED)" >&2
  exit 2
fi

WANDB_RUN="${WANDB_RUN:-aria-expC-d24-ratio${RATIO}-seed${SEED}}"
MODEL_TAG="${MODEL_TAG:-expC-d24-ratio${RATIO}-seed${SEED}}"

TRAIN_ARGS=(
  --run="$WANDB_RUN"
  --seed="$SEED"
  --device-type=cuda
  --fp8
  --fp8-recipe=tensorwise
  --depth=24
  --aspect-ratio=64
  --head-dim=128
  --max-seq-len=2048
  --window-pattern=SSSL
  --target-param-data-ratio="$RATIO"
  --device-batch-size=16
  --total-batch-size=-1
  --embedding-lr=0.3
  --matrix-lr=0.02
  --scalar-lr=0.5
  --unembedding-lr=0.008
  --weight-decay=0.28
  --warmup-steps=40
  --warmdown-ratio=0.65
  --final-lr-frac=0.05
  --eval-every=250
  --eval-tokens=41943040
  --core-metric-every=2000
  --core-metric-max-per-task=500
  --sample-every=2000
  --save-every=-1
  --grad-metrics-every=100
  --model-tag="$MODEL_TAG"
)

if [[ "$NPROC" == "1" ]]; then
  python -m scripts.base_train "${TRAIN_ARGS[@]}"
else
  torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- "${TRAIN_ARGS[@]}"
fi
