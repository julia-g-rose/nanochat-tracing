"""Review-only plan and paired-results analysis. Never launches training.

Print six commands: python experiments/matched_seed_attention.py plan
Analyze exported final summaries: python experiments/matched_seed_attention.py analyze results.json
"""
import argparse
import json
import math
import shlex
import statistics
import sys

SEEDS = (42, 43, 44)
STEPS = 5880
TOKENS_PER_STEP = 524288
GROUP = "matched-seed-attention-d12-ratio28-v1"


def plan():
    trials = []
    for seed in SEEDS:
        # Alternate treatment order to reduce a systematic time/order bias.
        for arm in (("baseline", "gated") if seed != 43 else ("gated", "baseline")):
            name = f"{GROUP}-s{seed}-{arm}"
            command = [sys.executable, "-m", "scripts.base_train",
                       "--run", name, "--model-tag", name,
                       "--experiment-group", GROUP, "--pair-id", f"seed-{seed}",
                       "--seed", str(seed), "--device-type", "cuda",
                       "--depth", "12", "--aspect-ratio", "64", "--head-dim", "128",
                       "--max-seq-len", "2048", "--window-pattern", "SSSL",
                       "--num-iterations", str(STEPS), "--target-param-data-ratio", "28",
                       "--total-batch-size", str(TOKENS_PER_STEP),
                       "--device-batch-size", "16", "--matrix-lr", "0.02",
                       "--embedding-lr", "0.3", "--unembedding-lr", "0.008",
                       "--scalar-lr", "0.5", "--weight-decay", "0.28",
                       "--warmup-steps", "40", "--warmdown-ratio", "0.65",
                       "--final-lr-frac", "0.05", "--fp8", "--fp8-recipe", "tensorwise",
                       "--eval-every", "250", "--eval-tokens", "41943040",
                       "--core-metric-every", "2000", "--core-metric-max-per-task", "500",
                       "--sample-every", "2000", "--grad-metrics-every", "100",
                       "--resume-from-step", "-1"]
            if arm == "gated":
                command.append("--attention-gate")
            trials.append({"seed": seed, "arm": arm, "pair_id": f"seed-{seed}",
                           "training_tokens": STEPS * TOKENS_PER_STEP,
                           "command": shlex.join(command)})
    return {"status": "review_only_not_authorized_to_run", "project": "wandb/nanochat-with-aria",
            "proposed_queue": "aria-h200-autoresearch", "max_full_runs": 6,
            "proposed_campaign_gpu_hours": 12, "trials": trials}


def analyze(rows):
    """Input: six finished final-summary records; reject partial/duplicate pairs.

    Each row: seed, arm, state, step, val/bpb, core_metric, total_training_time.
    Caller must verify source/assets/protocol using the checklist before export.
    """
    indexed = {}
    for row in rows:
        key = (row["seed"], row["arm"])
        if key in indexed or key[0] not in SEEDS or key[1] not in ("baseline", "gated"):
            raise ValueError(f"Unexpected or duplicate trial: {key}")
        if row["state"] != "finished" or row["step"] != STEPS:
            raise ValueError(f"Trial is not a completed {STEPS}-step run: {key}")
        for metric in ("val/bpb", "core_metric", "total_training_time"):
            value = row[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {metric}: {key}")
        if row["total_training_time"] <= 0 or row["val/bpb"] <= 0:
            raise ValueError(f"Invalid positive metric: {key}")
        indexed[key] = row
    if len(indexed) != 6:
        raise ValueError("Exactly three complete matched pairs are required")
    pairs = []
    for seed in SEEDS:
        b, g = indexed[seed, "baseline"], indexed[seed, "gated"]
        pairs.append({"seed": seed, "delta_bpb": g["val/bpb"] - b["val/bpb"],
                      "delta_core": g["core_metric"] - b["core_metric"],
                      "time_overhead": g["total_training_time"] / b["total_training_time"] - 1})
    mean_bpb = statistics.mean(p["delta_bpb"] for p in pairs)
    mean_core = statistics.mean(p["delta_core"] for p in pairs)
    time_overhead = (sum(indexed[s, "gated"]["total_training_time"] for s in SEEDS) /
                     sum(indexed[s, "baseline"]["total_training_time"] for s in SEEDS) - 1)
    wins = sum(p["delta_bpb"] < 0 for p in pairs)
    promising = mean_bpb <= -0.002 and wins >= 2 and mean_core >= -0.005 and time_overhead <= 0.05
    return {"pairs": pairs, "mean_delta_bpb": mean_bpb,
            "sd_delta_bpb": statistics.stdev(p["delta_bpb"] for p in pairs),
            "mean_delta_core": mean_core, "aggregate_training_time_overhead": time_overhead,
            "gated_wins": wins, "decision": "promising_screen" if promising else "not_promoted",
            "caveat": "Three pairs are a screen, not definitive statistical evidence. Inspect each pair."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan")
    sub.add_parser("analyze").add_argument("results")
    args = parser.parse_args()
    if args.action == "plan":
        result = plan()
    else:
        with open(args.results) as f:
            result = analyze(json.load(f))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()