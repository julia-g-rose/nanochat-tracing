"""Unit tests for query-dependent, head-specific SDPA output gating."""

import torch

import nanochat.gpt as gpt_module
from nanochat.gpt import CausalSelfAttention, GPT, GPTConfig


def _tiny_config():
    return GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
    )


def test_gate_is_applied_per_query_and_head(monkeypatch):
    config = _tiny_config()
    attention = CausalSelfAttention(config, layer_idx=0)
    torch.nn.init.zeros_(attention.sdpa_gate.weight)
    torch.nn.init.eye_(attention.c_proj.weight)

    def fake_sdpa(q, _k, _v, causal, window_size):
        assert causal
        assert window_size == (-1, 0)
        return torch.ones_like(q)

    monkeypatch.setattr(gpt_module.flash_attn, "flash_attn_func", fake_sdpa)
    x = torch.randn(2, 5, config.n_embd, requires_grad=True)
    half_dim = config.n_embd // config.n_head // 2
    cos_sin = (
        torch.ones(1, 5, 1, half_dim),
        torch.zeros(1, 5, 1, half_dim),
    )

    output = attention(x, ve=None, cos_sin=cos_sin, window_size=(-1, 0), kv_cache=None)
    torch.testing.assert_close(output, torch.full_like(output, 0.5))
    output.sum().backward()
    assert attention.sdpa_gate.weight.grad is not None
    assert attention.sdpa_gate.weight.grad.abs().sum() > 0


def test_gate_runs_through_real_sdpa():
    config = _tiny_config()
    attention = CausalSelfAttention(config, layer_idx=0)
    torch.nn.init.zeros_(attention.sdpa_gate.weight)
    x = torch.randn(2, 5, config.n_embd, requires_grad=True)
    half_dim = config.n_embd // config.n_head // 2
    cos_sin = (
        torch.ones(1, 5, 1, half_dim),
        torch.zeros(1, 5, 1, half_dim),
    )

    output = attention(x, ve=None, cos_sin=cos_sin, window_size=(-1, 0), kv_cache=None)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert attention.sdpa_gate.weight.grad is not None
    assert torch.isfinite(attention.sdpa_gate.weight.grad).all()


def test_model_initializes_neutral_gates_and_counts_parameters():
    config = _tiny_config()
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()

    for block in model.transformer.h:
        torch.testing.assert_close(
            block.attn.sdpa_gate.weight,
            torch.zeros_like(block.attn.sdpa_gate.weight),
        )
        gates = torch.sigmoid(block.attn.sdpa_gate(torch.randn(2, 3, config.n_embd)))
        torch.testing.assert_close(gates, torch.full_like(gates, 0.5))

    counts = model.num_scaling_params()
    assert counts["total"] == sum(parameter.numel() for parameter in model.parameters())