"""Stand-ins for the two NVIDIA training libraries the Cosmos code imports, so neither has
to be installed for single-GPU inference.

Megatron-Core: only `megatron.core.parallel_state` (is model / context parallelism set
up?) and the `ModelParallelConfig` dataclass (a field of the config) are used on the
inference path. The stand-ins answer "not initialised, one GPU" (upstream behaves the
same when run on one GPU without torchrun).

TransformerEngine: plain PyTorch for the three pieces the Cosmos DiT uses at inference
time (cosmos_predict1/diffusion/module/attention.py):

    te.pytorch.RMSNorm                          per-head q/k normalisation
    transformer_engine.pytorch.attention.apply_rotary_pos_emb   3D RoPE (non-interleaved)
    transformer_engine.pytorch.attention.DotProductAttention    softmax attention, no mask

Same maths as TransformerEngine 1.12 (the version upstream pins); attention runs on
torch.nn.functional.scaled_dot_product_attention (FlashAttention-2 / memory-efficient
kernels on the RTX 4090). Installing the real TransformerEngine would mean compiling it
against CUDA 12 headers, which this machine's glibc cannot do. No FP8, no context
parallelism (single GPU inference only).

install() must run before anything from cosmos_predict1 is imported.

The real megatron-core pip package is not installed on purpose: it imports
TransformerEngine itself and pulls in its training stack (nltk, sentencepiece, ...).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight, computed in float32 (TE's kernel does too)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, zero_centered_gamma: bool = False, **_):
        super().__init__()
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        w = self.weight.float() + 1 if self.zero_centered_gamma else self.weight.float()
        return (y * w).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, tensor_format: str = "sbhd",
                         fused: bool = False, cu_seqlens=None, interleaved: bool = False, **_) -> torch.Tensor:
    """TE's non-interleaved RoPE. freqs: [s, 1, 1, d] angles; only the first d channels rotate."""
    if interleaved or cu_seqlens is not None:
        raise NotImplementedError("te_shim: only non-interleaved, non-packed RoPE is implemented")
    if tensor_format not in ("sbhd", "bshd"):
        raise NotImplementedError(f"te_shim: tensor_format {tensor_format}")
    seq_len = t.shape[0] if tensor_format == "sbhd" else t.shape[1]
    freqs = freqs[:seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)
    rot_dim = freqs.shape[-1]
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # The fused TE kernel computes in float32 and casts the result back.
    tf = t_rot.float()
    out = tf * torch.cos(freqs.float()) + _rotate_half(tf) * torch.sin(freqs.float())
    return torch.cat((out.to(t.dtype), t_pass), dim=-1)


class DotProductAttention(nn.Module):
    """softmax(q k^T / sqrt(d)) v, no mask, no dropout. Output [s, b, h*d] for "sbhd"."""

    def __init__(self, num_attention_heads: int, kv_channels: int, qkv_format: str = "sbhd",
                 attention_dropout: float = 0.0, attn_mask_type: str = "no_mask", **_):
        super().__init__()
        if attn_mask_type != "no_mask" or attention_dropout:
            raise NotImplementedError("te_shim: only mask-free attention without dropout is implemented")
        if qkv_format not in ("sbhd", "bshd"):
            raise NotImplementedError(f"te_shim: qkv_format {qkv_format}")
        self.qkv_format = qkv_format
        self.cp_group = None
        self.cp_ranks = None

    def set_context_parallel_group(self, *args, **kwargs):
        raise NotImplementedError("te_shim: context parallelism is not supported")

    def forward(self, q, k, v, core_attention_bias_type: str = "no_bias", core_attention_bias=None, **_):
        if core_attention_bias is not None:
            raise NotImplementedError("te_shim: attention bias")
        if self.qkv_format == "sbhd":  # [s, b, h, d] -> [b, h, s, d]
            q, k, v = (x.permute(1, 2, 0, 3) for x in (q, k, v))
        else:  # [b, s, h, d]
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)  # [b, h, s, d]
        if self.qkv_format == "sbhd":
            return out.permute(2, 0, 1, 3).flatten(2)  # [s, b, h*d]
        return out.transpose(1, 2).flatten(2)  # [b, s, h*d]


@dataclass
class ModelParallelConfig:
    """The fields of megatron.core.ModelParallelConfig that cosmos_predict1's Config carries (one GPU)."""

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False


def _megatron_parallel_state() -> types.ModuleType:
    ps = types.ModuleType("megatron.core.parallel_state")
    ps.is_initialized = lambda: False
    ps.model_parallel_is_initialized = lambda: False
    for name in ("tensor_model", "context", "data", "pipeline_model", "expert_model"):
        setattr(ps, f"get_{name}_parallel_world_size", lambda: 1)
        setattr(ps, f"get_{name}_parallel_rank", lambda: 0)
    for name in ("tensor_model", "context", "data", "model", "pipeline_model"):
        setattr(ps, f"get_{name}_parallel_group", lambda *a, **k: None)
    return ps


def install() -> None:
    """Register `transformer_engine` (+ .pytorch, .pytorch.attention) and `megatron.core.parallel_state`."""
    if "transformer_engine" in sys.modules:
        return
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    core.parallel_state = _megatron_parallel_state()
    core.ModelParallelConfig = ModelParallelConfig
    megatron.core = core
    megatron.__path__ = []
    core.__path__ = []
    sys.modules.update({"megatron": megatron, "megatron.core": core, "megatron.core.parallel_state": core.parallel_state})
    te = types.ModuleType("transformer_engine")
    te_pt = types.ModuleType("transformer_engine.pytorch")
    te_attn = types.ModuleType("transformer_engine.pytorch.attention")
    te_attn.DotProductAttention = DotProductAttention
    te_attn.apply_rotary_pos_emb = apply_rotary_pos_emb
    te_pt.RMSNorm = RMSNorm
    te_pt.DotProductAttention = DotProductAttention
    te_pt.attention = te_attn
    te.pytorch = te_pt
    te.__version__ = "0+lab2shot.shim"
    te.__path__ = []  # a package, so "import transformer_engine.pytorch" works
    te_pt.__path__ = []
    sys.modules.update({
        "transformer_engine": te,
        "transformer_engine.pytorch": te_pt,
        "transformer_engine.pytorch.attention": te_attn,
    })
