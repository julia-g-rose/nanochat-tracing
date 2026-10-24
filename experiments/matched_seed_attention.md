# D12 matched-seed standard vs gated attention (draft)

## Status and scope

**Review only. No training is submitted by this PR or by its plan generator.**
Proposed runner: one H200 on W&B Launch queue `aria-h200-autoresearch`.
Proposed budget: six new full runs, approximately 9–10 H200 training-hours;
allow up to 12 H200-hours including setup/evaluation, subject to fresh approval.
No replacement trials, automatic extensions, merge, or promotion is authorized.
The generator does not implement a queue submitter or budget watchdog. An
approved execution must enforce the wall-clock cap and stop/cancel remaining
work when the cap is reached. If interrupted, report partial evidence rather
than replacing runs or treating partial checkpoints as final results.

This branch includes the unmerged opt-in gate implementation from PR #4 and
the seed CLI from `aria/ratio28-seed43-replication`, applied on current master.
It preserves master's unrelated files. Review the new initialization fix and
experiment files separately from these prerequisite commits.

## Motivation (observed historical results, not new measurements)

Project: https://wandb.ai/wandb/nanochat-with-aria

| Historical run | BPB (lower better) | CORE (higher better) |
| --- | ---: | ---: |
| Standard ratio 28, original seed 42: `ftvu5hjb` | 0.8183553 | 0.1955940 |
| Standard ratio 28, seed 43: `wkcel0bq` | 0.8224290 | 0.1628849 |
| Gated ratio 28, original seed 42: `iqmdwe3k` | 0.8208257 | 0.1731872 |

The observed repeat gap of 0.00407 BPB exceeds the original standard-vs-gated
gap of 0.00247 BPB. Code revisions also differ, so this is not a pure estimate
of seed variance. Existing gated runs had tiny token-budget differences and
gate initialization changed subsequent shared-weight random draws. These
runs motivate the experiment but are excluded from its primary analysis.

## Hypothesis and controls

At fixed training tokens, query-dependent headwise sigmoid gating after SDPA
improves mean final validation BPB across matched seeds without unacceptable
CORE degradation or runtime overhead.

Three pairs: seeds **42, 43, 44**, each with attention gating off and on. Use
six fresh runs from **one immutable commit of this branch**. Vary only the
gate inside each pair; run/model names necessarily differ. Alternate arm
order (baseline first for 42/44, gated first for 43).

| Setting | Fixed value |
| --- | --- |
| Model | 12 layers, width 768, 6 attention/KV heads, head dim 128 |
| Sequence/window | 2048 tokens, SSSL |
| Horizon | **5,880 optimizer steps** (explicit override, no resume) |
| Global batch | **524,288 tokens/step** (16 examples/device, accumulation 16, one GPU) |
| Training tokens | **3,082,813,440 per run** |
| Precision | BF16 compute with tensorwise FP8 training, same FA3 backend |
| Learning rates | matrix 0.02, embedding 0.3, unembedding 0.008, scalar 0.5 |
| Weight decay | 0.28; verify effective value matches across arms |
| Schedule | warmup 40, warmdown ratio 0.65, final LR fraction 0.05 |
| Validation | 41,943,040 tokens; every 250 steps and final |
| CORE | identical task examples, max 500/task; every 2,000 steps and final |
| Samples/gradient diagnostics | every 2,000 / 100 steps |
| Hardware/software | same one-H200 runner and pinned environment |

Keep `target_param_data_ratio=28` for the existing optimizer-scaling formulas,
but explicitly set steps and batch size. Do not let added gate parameters
increase the number of training steps. This is **equal-token**, not equal-FLOP
or equal-wall-clock comparison; report the treatment overhead.

### Initialization fix

The original `init_weights` initialized each gate between shared projections
and the MLP, shifting later random draws. Move all new gate draws after all
shared random draws and put them in `torch.random.fork_rng`, covering CPU and
the gate's CUDA device. The query-projection uniform distribution is retained;
the exact historical gate values intentionally change. Both arms are built on
meta then materialized, matching `base_train`.

Tests assert exact equality of **every shared parameter** and the next RNG
draw, gate repeatability/non-degeneracy, seed sensitivity, and finite backward
gradients. CUDA equality has its own test and must pass on the runner.
FP8/FA3 training is not promised bitwise deterministic; paired initialization
controls a confound, not every source of numerical variation.

## Before authorizing any full run

1. Approve queue, six-run/12-H200-hour bounds, and thresholds below.
2. Pin the final source SHA and container/environment digest. Verify a clean
   checkout; set `NANOCHAT_SOURCE_COMMIT` to that SHA for config provenance.
3. Resolve immutable versions/digests of existing collections
   `nanochat-climbmix-cache-170shards` and
   `nanochat-tokenizer-climbmix-170shards`. Hydrate the same versions into the
   runtime cache for all runs. Do not use moving `:latest` aliases during the
   campaign. Record the ordered shard list and hashes, tokenizer hash, CORE
   bundle/config hashes, library versions, and GPU/backend in run metadata.
   The plan generator itself neither resolves nor hydrates assets.
4. Run the CPU tests below and the CUDA initialization test on H200. Check
   that the actual first few training batches have identical token hashes
   across arms for each seed; this data check is pending, not covered by the
   synthetic unit tests. Keep validation/task ordering identical too.
5. In a separately approved tiny GPU shakeout, verify FP8/FA3, finite loss and
   gradients, cache availability, final BPB/CORE logging, and model save.
   Do not call a two-step result scientific evidence. Count GPU preflight
   against the campaign cap; do not silently create additional W&B runs.
6. Disable fragile optional EvalTable instrumentation for this experiment;
   this branch uses the established scalar/table path, not PR #6. Abort on
   missing assets, invalid logging, or a protocol mismatch before spending a
   full run. Never silently change dependencies or fall back to another backend.

## Inspect the plan (safe; creates no W&B runs)

```bash
python -m unittest discover -s tests -p 'test_matched_seed_*.py' -v
python experiments/matched_seed_attention.py plan
```

The JSON prints all six explicit commands and does **not** execute them.
After approval and preflight, submit those commands on the agreed runner with
`WANDB_ENTITY=wandb`, `WANDB_PROJECT=nanochat-with-aria`, one GPU, a pinned
`NANOCHAT_BASE_DIR`, and `NANOCHAT_SOURCE_COMMIT` set to the reviewed SHA.
The W&B group is `matched-seed-attention-d12-ratio28-v1`; `pair_id` is logged.
ARIA thread/turn provenance is preserved when those environment variables are
provided. Keep job/artifact lineage for the source, environment, and assets.

The existing image job fetches an old commit at runtime. Do **not** blindly
relaunch it: update its entrypoint to the reviewed SHA and the explicit plan
arguments, then read back the submitted job/config before claiming it matches.

## Predeclared analysis (proposed thresholds; not yet approved)

For each seed, `delta_bpb = gated_final_bpb - baseline_final_bpb`.
Use the final 5,880-step checkpoint, never independently selected best checkpoints.

A **promising screen** requires all of:

- Mean paired BPB delta <= **-0.002**, with at least **2/3** seed pairs improving.
- Mean paired CORE delta >= **-0.005 absolute** (not a percentage).
- Aggregate gated training time / aggregate standard training time <= **1.05**.

The historical absolute CORE=0.17 threshold is not used here: a paired relative
guardrail is proposed because the seed-43 baseline itself was below 0.17.
Report all absolute CORE scores too; this change requires reviewer approval.
Failure to meet the rule means **not promoted**, not proof the gate is harmful.
Report paired deltas, mean/sample SD, final BPB/CORE, training time, end-to-end
time, total FLOPs, throughput and gate mean/std/quantiles/layer means. Three
pairs are exploratory screening evidence, not a definitive significance claim.

Export exactly six protocol-verified final summary records into a JSON array.
Required fields per row: `seed`, `arm` (`baseline` or `gated`), `state`, `step`,
`val/bpb`, `core_metric`, `total_training_time`. Then:

```bash
python experiments/matched_seed_attention.py analyze results.json
```

The analyzer rejects failed, incomplete, duplicate, missing or nonfinite trials.
It calculates the prespecified paired deltas and guardrails. Verify source,
asset hashes, batch size, precision and evaluation protocol **before** export;
the analyzer does not independently query or audit W&B provenance.

## Validation scope for this draft

- Sandbox checks: **7 new CPU tests passed; 1 CUDA test skipped**. Four
  existing gate regression functions also passed via direct invocation
  (`pytest` is not installed in the sandbox). Syntax and whitespace checks passed.
- A separate CPU comparison against current master confirmed that gate-off
  weights and the next RNG draw match exactly for seeds 42, 43 and 44.
  Sandbox torch is 2.12.1+cpu, not the runner's pinned torch 2.9.1.
- Tiny CPU tests and syntax checks can run without a GPU or model downloads.
- CUDA, pinned torch 2.9.1, FP8/FA3, real data/evaluation and end-to-end Launch
  execution require the runner; no results from those checks are claimed here.
- No training, checkpoint promotion, PR merge or GPU spend is part of drafting
  this change. Launch approval is a separate action.