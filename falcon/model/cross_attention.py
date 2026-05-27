"""Gated cross-attention bottleneck for fusing face and body features.

Current implementation (``CrossBottleneckAttn``):
- Query from face attending to body key/value, and vice versa.
- Gated fusion via sigmoid-weighted blending.
- Single step (not iterative).

See ``falcon_improvement_proposals.md`` for a proposed bi-directional
iterative variant (``BidirectionalCrossAttn``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.layers.bottleneck_attn import PosEmbedRel
from timm.layers.helpers import make_divisible
from timm.layers.mlp import Mlp
from timm.layers.trace_utils import _assert
from timm.layers.weight_init import trunc_normal_

__all__ = ["CrossBottleneckAttn"]


class CrossBottleneckAttn(nn.Module):
    """Gated cross-attention between face and body feature tensors.

    Splits the input channel-wise in half (face / body), computes
    cross-attention in both directions, and blends the results with
    a learnable sigmoid gate.

    Args:
        dim: Input channel dimension.
        dim_out: Output channel dimension (defaults to *dim*).
        feat_size: Spatial (height, width) of the feature map.
        stride: Spatial downsampling factor (2 or 1).
        num_heads: Number of attention heads.
        dim_head: Per-head dimension for Q/K (defaults to
            ``make_divisible(dim_out * qk_ratio, 8) // num_heads``).
        qk_ratio: Ratio of Q/K dimension relative to *dim_out*.
        qkv_bias: Whether to use bias in QKV projections.
        scale_pos_embed: Whether to scale position embeddings.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        feat_size: tuple[int, int] | None = None,
        stride: int = 1,
        num_heads: int = 4,
        dim_head: int | None = None,
        qk_ratio: float = 1.0,
        qkv_bias: bool = False,
        scale_pos_embed: bool = False,
    ):
        super().__init__()
        assert feat_size is not None
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0

        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(
            dim_out * qk_ratio, divisor=8
        ) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk ** -0.5
        self.scale_pos_embed = scale_pos_embed

        self.qkv_f = nn.Conv2d(
            dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias
        )
        self.qkv_p = nn.Conv2d(
            dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias
        )

        self.pos_embed = PosEmbedRel(
            feat_size, dim_head=self.dim_head_qk, scale=self.scale
        )

        self.gate_conv = nn.Conv2d(
            self.dim_out_v * 2, self.dim_out_v, kernel_size=1
        )

        self.norm = nn.LayerNorm([self.dim_out_v, *feat_size])
        mlp_ratio = 4
        self.mlp = Mlp(
            in_features=self.dim_out_v,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            out_features=dim_out,
            drop=0,
            use_conv=True,
        )
        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        """Initialise weights with truncated normal."""
        trunc_normal_(self.qkv_f.weight, std=self.qkv_f.weight.shape[1] ** -0.5)
        trunc_normal_(self.qkv_p.weight, std=self.qkv_p.weight.shape[1] ** -0.5)
        trunc_normal_(self.gate_conv.weight, std=0.02)
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def get_qkv(
        self, x: torch.Tensor, qvk_conv: nn.Conv2d
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input through *qvk_conv* and split into Q, K, V tensors."""
        B, C, H, W = x.shape
        x = qvk_conv(x)
        q, k, v = torch.split(
            x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1
        )
        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)
        v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        return q, k, v

    def apply_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        H: int,
        W: int,
        dropout: nn.Module | None = None,
    ) -> torch.Tensor:
        """Apply attention with optional position embeddings."""
        if self.scale_pos_embed:
            attn = (q @ k + self.pos_embed(q)) * self.scale
        else:
            attn = (q @ k) * self.scale + self.pos_embed(q)
        attn = attn.softmax(dim=-1)
        if dropout:
            attn = dropout(attn)
        out = attn @ v
        out = out.transpose(-1, -2).reshape(B, self.dim_out_v, H, W)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor shaped (B, 2 * C, H, W) where the first half is
                face features and the second half is body features.

        Returns:
            Fused tensor of shape (B, dim_out, H_out, W_out).
        """
        B, C, H, W = x.shape
        dim = int(C / 2)
        x1 = x[:, :dim, :, :]
        x2 = x[:, dim:, :, :]

        _assert(H == self.pos_embed.height, "Height mismatch with pos_embed")
        _assert(W == self.pos_embed.width, "Width mismatch with pos_embed")

        q_f, k_f, v_f = self.get_qkv(x1, self.qkv_f)
        q_p, k_p, v_p = self.get_qkv(x2, self.qkv_p)

        out_f = self.apply_attn(q_f, k_p, v_p, B, H, W)
        out_p = self.apply_attn(q_p, k_f, v_f, B, H, W)

        gate_input = torch.cat((out_f, out_p), dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))
        x_pf = gate * out_f + (1 - gate) * out_p

        x_pf = self.norm(x_pf)
        x_pf = self.mlp(x_pf)
        out = self.pool(x_pf)
        return out
