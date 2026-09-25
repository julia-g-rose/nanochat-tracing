"""Unit tests for Peri-LN branch-output normalization."""

import torch
import torch.nn as nn

from nanochat.gpt import Block, GPT, GPTConfig, norm


class _ConstantBranch(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, *args):
        return torch.full_like(x, self.value)


def _config(peri_ln):
    return GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
        peri_ln=peri_ln,
    )


def test_fixed_peri_ln_normalizes_each_branch_before_residual_add():
    block = Block(_config("fixed"), layer_idx=0)
    block.attn = _ConstantBranch(2.0)
    block.mlp = _ConstantBranch(3.0)
    x = torch.randn(2, 5, 32)

    actual = block(x, ve=None, cos_sin=None, window_size=None, kv_cache=None)
    after_attn = x + norm(torch.full_like(x, 2.0))
    expected = after_attn + norm(torch.full_like(x, 3.0))
    assert torch.allclose(actual, expected)


def test_learned_peri_ln_gain_is_trainable_and_initialized_to_one():
    config = _config("learned")
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()

    for block in model.transformer.h:
        assert torch.allclose(block.peri_ln_attn_gain, torch.ones(config.n_embd))
        assert torch.allclose(block.peri_ln_mlp_gain, torch.ones(config.n_embd))
        assert block.peri_ln_attn_gain.requires_grad
        assert block.peri_ln_mlp_gain.requires_grad

    counts = model.num_scaling_params()
    assert counts["peri_ln"] == 2 * config.n_layer * config.n_embd
    assert counts["total"] == sum(parameter.numel() for parameter in model.parameters())


def test_peri_ln_is_opt_in():
    block = Block(_config("none"), layer_idx=0)
    assert block.peri_ln_attn_gain is None
    assert block.peri_ln_mlp_gain is None


def test_unknown_peri_ln_mode_is_rejected():
    try:
        Block(_config("other"), layer_idx=0)
    except ValueError as exc:
        assert "Unknown peri_ln mode" in str(exc)
    else:
        raise AssertionError("invalid Peri-LN mode should fail")