"""CPU-sized reproducibility checks; also runnable without pytest.

    python -m unittest discover -s tests -p test_matched_seed_initialization.py -v
"""
import unittest

import torch

from nanochat.gpt import GPT, GPTConfig


def build(seed, gated, device="cpu"):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2,
                       n_head=4, n_kv_head=4, n_embd=32,
                       window_pattern="L", attention_gate=gated)
    # Mirror base_train: constructors run on meta and consume no real RNG.
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    return model


class MatchedSeedInitialization(unittest.TestCase):
    def check_pair(self, device):
        for seed in (42, 43, 44):
            with self.subTest(seed=seed, device=device):
                baseline = build(seed, False, device)
                baseline_next = torch.rand(8, device=device)
                gated = build(seed, True, device)
                gated_next = torch.rand(8, device=device)
                treatment = dict(gated.named_parameters())
                for name, weight in baseline.named_parameters():
                    self.assertTrue(torch.equal(weight, treatment[name]), name)
                self.assertTrue(torch.equal(baseline_next, gated_next))
                gates = [b.attn.sdpa_gate.weight for b in gated.transformer.h]
                for gate in gates:
                    self.assertTrue(torch.isfinite(gate).all().item())
                    self.assertGreater(torch.count_nonzero(gate).item(), 0)
                self.assertFalse(torch.equal(gates[0], gates[1]))

    def test_shared_weights_and_next_rng_draw_match_cpu(self):
        self.check_pair("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA runner")
    def test_shared_weights_and_next_rng_draw_match_cuda(self):
        self.check_pair("cuda")

    def test_seed_changes_initialization_and_gate_is_repeatable(self):
        first = build(42, True)
        repeat = build(42, True)
        other = build(43, True)
        for (name, a), (_, b) in zip(first.named_parameters(), repeat.named_parameters()):
            self.assertTrue(torch.equal(a, b), name)
        self.assertFalse(torch.equal(first.transformer.wte.weight, other.transformer.wte.weight))
        self.assertFalse(torch.equal(first.transformer.h[0].attn.sdpa_gate.weight,
                                     other.transformer.h[0].attn.sdpa_gate.weight))

    def test_gate_forward_backward_is_finite(self):
        model = build(42, True)
        tokens = torch.arange(10).reshape(2, 5)
        loss = model(tokens, targets=(tokens + 1) % 64)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for block in model.transformer.h:
            grad = block.attn.sdpa_gate.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all().item())


if __name__ == "__main__":
    unittest.main()