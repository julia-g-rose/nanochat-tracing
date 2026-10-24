"""Standard-library tests for the review-only plan and decision calculation."""
import copy
import shlex
import unittest

from experiments.matched_seed_attention import STEPS, TOKENS_PER_STEP, analyze, plan


def fixture():
    # Synthetic fixtures only: not measured experiment results.
    return [dict(seed=seed, arm=arm, state="finished", step=STEPS,
                 **{"val/bpb": 0.82 if arm == "baseline" else 0.817,
                    "core_metric": 0.17, "total_training_time": 5500 if arm == "baseline" else 5650})
            for seed in (42, 43, 44) for arm in ("baseline", "gated")]


class MatchedSeedProtocol(unittest.TestCase):
    def test_six_explicit_equal_token_budgets(self):
        trials = plan()["trials"]
        self.assertEqual(len(trials), 6)
        self.assertEqual({(t["seed"], t["arm"]) for t in trials},
                         {(s, a) for s in (42, 43, 44) for a in ("baseline", "gated")})
        for t in trials:
            cmd = shlex.split(t["command"])
            self.assertEqual(cmd[cmd.index("--num-iterations") + 1], str(STEPS))
            self.assertEqual(cmd[cmd.index("--total-batch-size") + 1], str(TOKENS_PER_STEP))
            self.assertEqual("--attention-gate" in cmd, t["arm"] == "gated")
            self.assertEqual(t["training_tokens"], 3082813440)

    def test_promising_screen_and_time_guardrail(self):
        rows = fixture()
        self.assertEqual(analyze(rows)["decision"], "promising_screen")
        for row in rows:
            if row["arm"] == "gated":
                row["total_training_time"] = 6000
        self.assertEqual(analyze(rows)["decision"], "not_promoted")

    def test_core_guardrail(self):
        rows = fixture()
        for row in rows:
            if row["arm"] == "gated":
                row["core_metric"] = 0.15
        self.assertEqual(analyze(rows)["decision"], "not_promoted")

    def test_reject_incomplete_duplicate_nonfinite_or_failed(self):
        invalid = [fixture()[:-1], fixture() + [fixture()[0]]]
        for field, value in (("state", "failed"), ("step", 2000), ("val/bpb", float("nan")),
                             ("total_training_time", 0)):
            rows = copy.deepcopy(fixture())
            rows[0][field] = value
            invalid.append(rows)
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                analyze(rows)


if __name__ == "__main__":
    unittest.main()