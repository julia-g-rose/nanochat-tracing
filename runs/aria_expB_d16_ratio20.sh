#!/usr/bin/env bash

# Depth-16, 20:1 token-to-scaling-parameter experiment proposed in:
# https://wandb.ai/wandb/nanochat-with-aria/reports/Nanochat-Scaling-Proposal-%E2%80%94-Depth-16-at-Ratio-20--VmlldzoxNzk4MDAyNg==
#
# Prerequisites: the tokenizer and ClimbMix shards used by the existing W&B
# runs are available under NANOCHAT_BASE_DIR. This script only runs pretraining.
# The successful comparison runs used one GPU, so NPROC_PER_NODE defaults to 1;
# set it higher when launching on a multi-GPU node.

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC="${NPROC_PER_NODE:-1}"
WANDB_RUN="${WANDB_RUN:-aria-expB-d16-ratio20}"
MODEL_TAG="${MODEL_TAG:-expB-d16-ratio20}"

TRAIN_ARGS=(
  --run="$WANDB_RUN"
  --device-type=cuda
  --fp8
  --fp8-recipe=tensorwise
  --depth=16
  --aspect-ratio=64
  --head-dim=128
  --max-seq-len=2048
  --window-pattern=SSSL
  --target-param-data-ratio=20
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