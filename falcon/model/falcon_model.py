"""Falcon model definition — multi-input transformer for age & gender.

Extends timm's VOLO with a ``PatchEmbed`` that supports dual-branch
(face + body) input and gated cross-attention fusion.  The head produces
both a gender logit vector and an age distribution over 101 bins.

See ``falcon_improvement_proposals.md`` for planned upgrades:
- DINOv2 / ConvNeXt-ViT hybrid backbone (Sprint 3).
- Multi-task auxiliary heads (Sprint 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from falcon.model.cross_attention import CrossBottleneckAttn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model
from timm.models.volo import VOLO

__all__ = [
    "FalconModel",
    "PatchEmbed",
    "get_output_size",
    "get_output_size_module",
    "falcon_d1_224",
    "falcon_d1_384",
    "falcon_d2_224",
    "falcon_d2_384",
    "falcon_d3_224",
    "falcon_d3_448",
    "falcon_d4_224",
    "falcon_d4_448",
    "falcon_d5_224",
    "falcon_d5_448",
    "falcon_d5_512",
]


def _cfg(url: str = "", **kwargs):
    """Helper to build a default config dict for model registries."""
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.96,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": None,
        "classifier": ("head", "aux_head"),
        **kwargs,
    }


default_cfgs = {
    "falcon_d1_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar",
        crop_pct=0.96,
    ),
    "falcon_d1_384": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d1_384_85.2.pth.tar",
        crop_pct=1.0,
        input_size=(3, 384, 384),
    ),
    "falcon_d2_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d2_224_85.2.pth.tar",
        crop_pct=0.96,
    ),
    "falcon_d2_384": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d2_384_86.0.pth.tar",
        crop_pct=1.0,
        input_size=(3, 384, 384),
    ),
    "falcon_d3_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d3_224_85.4.pth.tar",
        crop_pct=0.96,
    ),
    "falcon_d3_448": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d3_448_86.3.pth.tar",
        crop_pct=1.0,
        input_size=(3, 448, 448),
    ),
    "falcon_d4_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d4_224_85.7.pth.tar",
        crop_pct=0.96,
    ),
    "falcon_d4_448": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d4_448_86.79.pth.tar",
        crop_pct=1.15,
        input_size=(3, 448, 448),
    ),
    "falcon_d5_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d5_224_86.10.pth.tar",
        crop_pct=0.96,
    ),
    "falcon_d5_448": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d5_448_87.0.pth.tar",
        crop_pct=1.15,
        input_size=(3, 448, 448),
    ),
    "falcon_d5_512": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d5_512_87.07.pth.tar",
        crop_pct=1.15,
        input_size=(3, 512, 512),
    ),
}


def get_output_size(input_shape: list[int], conv_layer: nn.Conv2d) -> list[int]:
    """Compute the spatial output size after a conv layer.

    Args:
        input_shape: ``[height, width]`` of the input.
        conv_layer: The convolutional layer.

    Returns:
        ``[height, width]`` of the output.
    """
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    output_size = [
        (
            (input_shape[i] + 2 * padding[i]
             - dilation[i] * (kernel_size[i] - 1) - 1)
            // stride[i]
        )
        + 1
        for i in range(2)
    ]
    return output_size


def get_output_size_module(input_size: int, stem: nn.Sequential) -> list[int]:
    """Compute spatial output size after a sequential stem of conv layers.

    Args:
        input_size: Input spatial size (assumed square).
        stem: Sequential of conv modules.

    Returns:
        ``[height, width]`` after all conv layers.
    """
    output_size = [input_size, input_size]
    for module in stem:
        if isinstance(module, nn.Conv2d):
            output_size = [
                (
                    output_size[i] + 2 * module.padding[i]
                    - module.dilation[i] * (module.kernel_size[i] - 1)
                    - 1
                )
                // module.stride[i]
                + 1
                for i in range(2)
            ]
    return output_size


class PatchEmbed(nn.Module):
    """Overlapping patch embedding with optional dual-branch support.

    When ``in_chans == 6``, two separate stems project face (first 3 ch)
    and body (last 3 ch) features, fused via ``CrossBottleneckAttn``.
    """

    def __init__(
        self,
        img_size: int = 224,
        stem_conv: bool = False,
        stem_stride: int = 1,
        patch_size: int = 8,
        in_chans: int = 3,
        hidden_dim: int = 64,
        embed_dim: int = 384,
    ):
        super().__init__()
        assert patch_size in [4, 8, 16]
        assert in_chans in [3, 6]
        self.with_persons_model = in_chans == 6
        self.use_cross_attn = True

        if stem_conv:
            if not self.with_persons_model:
                self.conv = self.create_stem(stem_stride, in_chans, hidden_dim)
            else:
                self.conv = True
                self.conv1 = self.create_stem(stem_stride, 3, hidden_dim)
                self.conv2 = self.create_stem(stem_stride, 3, hidden_dim)
        else:
            self.conv = None

        overlap_kw = dict(
            kernel_size=7, stride=patch_size // stem_stride, padding=3
        )

        if self.with_persons_model:
            self.proj1 = nn.Conv2d(hidden_dim, embed_dim, **overlap_kw)
            self.proj2 = nn.Conv2d(hidden_dim, embed_dim, **overlap_kw)
            stem_out_shape = get_output_size_module(img_size, self.conv1)
            self.proj_output_size = get_output_size(
                stem_out_shape, self.proj1
            )
            self.map = CrossBottleneckAttn(
                embed_dim,
                dim_out=embed_dim,
                num_heads=1,
                feat_size=self.proj_output_size,
            )
        else:
            self.proj = nn.Conv2d(hidden_dim, embed_dim, **overlap_kw)

        self.patch_dim = img_size // patch_size
        self.num_patches = self.patch_dim ** 2

    @staticmethod
    def create_stem(
        stem_stride: int, in_chans: int, hidden_dim: int
    ) -> nn.Sequential:
        """Build the initial convolutional stem.

        Args:
            stem_stride: Stride for the first convolution.
            in_chans: Input channels (3 for face or body).
            hidden_dim: Hidden dimension.

        Returns:
            A ``nn.Sequential`` of Conv→BN→ReLU blocks.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_chans,
                hidden_dim,
                kernel_size=7,
                stride=stem_stride,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        For face+body mode (``in_chans == 6``), returns fused features.
        Otherwise returns single-branch patch embeddings.
        """
        if self.conv is not None:
            if self.with_persons_model:
                x1 = x[:, :3]
                x2 = x[:, 3:]
                x1 = self.conv1(x1)
                x1 = self.proj1(x1)
                x2 = self.conv2(x2)
                x2 = self.proj2(x2)
                x = torch.cat([x1, x2], dim=1)
                x = self.map(x)
            else:
                x = self.conv(x)
                x = self.proj(x)
        return x


class FalconModel(VOLO):
    """Falcon multi-input model extending VOLO.

    Adds age/gender prediction heads and a custom ``PatchEmbed`` that
    supports dual-branch face+body fusion.

    Model variants (``falcon_d1`` … ``falcon_d5``) differ in depth,
    embedding dimension, and head count.
    """

    def __init__(
        self,
        layers,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        patch_size=8,
        stem_hidden_dim=64,
        embed_dims=None,
        num_heads=None,
        downsamples=(True, False, False, False),
        outlook_attention=(True, False, False, False),
        mlp_ratio=3.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        post_layers=("ca", "ca"),
        use_aux_head=True,
        use_mix_token=False,
        pooling_scale=2,
    ):
        super().__init__(
            layers,
            img_size,
            in_chans,
            num_classes,
            global_pool,
            patch_size,
            stem_hidden_dim,
            embed_dims,
            num_heads,
            downsamples,
            outlook_attention,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            post_layers,
            use_aux_head,
            use_mix_token,
            pooling_scale,
        )

        im_size = img_size[0] if isinstance(img_size, tuple) else img_size
        self.patch_embed = PatchEmbed(
            img_size=im_size,
            stem_conv=True,
            stem_stride=2,
            patch_size=patch_size,
            in_chans=in_chans,
            hidden_dim=stem_hidden_dim,
            embed_dim=embed_dims[0],
        )

        self.num_age_bins = 101
        self.age_bins = torch.arange(self.num_age_bins, dtype=torch.float32)
        self.gender_head: nn.Linear | None = None
        self.age_head: nn.Linear | None = None

        if num_classes > 101:
            self.gender_head = nn.Linear(embed_dims[-1], 2)
            self.age_head = nn.Linear(embed_dims[-1], self.num_age_bins)

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone (patch embed + VOLO stages)."""
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        x = self.forward_tokens(x)
        if self.post_network is not None:
            x = self.forward_cls(x)
        x = self.norm(x)
        return x

    def forward_head(
        self, x, pre_logits: bool = False, targets=None, epoch=None
    ):
        """Apply the age/gender head.

        If *pre_logits* is ``True``, returns the pooled feature vector
        before the final linear projection.
        """
        if self.global_pool == "avg":
            out = x.mean(dim=1)
        elif self.global_pool == "token":
            out = x[:, 0]
        else:
            out = x
        if pre_logits:
            return out

        features = out
        fds_enabled = hasattr(self, "_fds_forward")
        if fds_enabled:
            features = self._fds_forward(features, targets, epoch)

        if self.age_head is not None and self.gender_head is not None:
            age_out = self.age_head(features)
            gender_out = self.gender_head(features)
            out = torch.cat([gender_out, age_out], dim=-1)
        elif hasattr(self, "head"):
            out = self.head(features)
            if self.aux_head is not None:
                aux = self.aux_head(x[:, 1:])
                out = out + 0.5 * aux.max(1)[0]
        return (out, features) if (fds_enabled and self.training) else out

    def forward(self, x, targets=None, epoch=None):
        """End-to-end forward: features → head."""
        x = self.forward_features(x)
        x = self.forward_head(x, targets=targets, epoch=epoch)
        return x


# ---------------------------------------------------------------------------
# Model registration helpers
# ---------------------------------------------------------------------------


def _create_falcon(variant: str, pretrained: bool = False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )
    return build_model_with_cfg(FalconModel, variant, pretrained, **kwargs)


@register_model
def falcon_d1_224(pretrained=False, **kwargs):
    model_args = dict(
        layers=(4, 4, 8, 2),
        embed_dims=(192, 384, 384, 384),
        num_heads=(6, 12, 12, 12),
        **kwargs,
    )
    return _create_falcon("falcon_d1_224", pretrained=pretrained, **model_args)


@register_model
def falcon_d1_384(pretrained=False, **kwargs):
    model_args = dict(
        layers=(4, 4, 8, 2),
        embed_dims=(192, 384, 384, 384),
        num_heads=(6, 12, 12, 12),
        **kwargs,
    )
    return _create_falcon("falcon_d1_384", pretrained=pretrained, **model_args)


@register_model
def falcon_d2_224(pretrained=False, **kwargs):
    model_args = dict(
        layers=(6, 4, 10, 4),
        embed_dims=(256, 512, 512, 512),
        num_heads=(8, 16, 16, 16),
        **kwargs,
    )
    return _create_falcon("falcon_d2_224", pretrained=pretrained, **model_args)


@register_model
def falcon_d2_384(pretrained=False, **kwargs):
    model_args = dict(
        layers=(6, 4, 10, 4),
        embed_dims=(256, 512, 512, 512),
        num_heads=(8, 16, 16, 16),
        **kwargs,
    )
    return _create_falcon("falcon_d2_384", pretrained=pretrained, **model_args)


@register_model
def falcon_d3_224(pretrained=False, **kwargs):
    model_args = dict(
        layers=(8, 8, 16, 4),
        embed_dims=(256, 512, 512, 512),
        num_heads=(8, 16, 16, 16),
        **kwargs,
    )
    return _create_falcon("falcon_d3_224", pretrained=pretrained, **model_args)


@register_model
def falcon_d3_448(pretrained=False, **kwargs):
    model_args = dict(
        layers=(8, 8, 16, 4),
        embed_dims=(256, 512, 512, 512),
        num_heads=(8, 16, 16, 16),
        **kwargs,
    )
    return _create_falcon("falcon_d3_448", pretrained=pretrained, **model_args)


@register_model
def falcon_d4_224(pretrained=False, **kwargs):
    model_args = dict(
        layers=(8, 8, 16, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        **kwargs,
    )
    return _create_falcon("falcon_d4_224", pretrained=pretrained, **model_args)


@register_model
def falcon_d4_448(pretrained=False, **kwargs):
    model_args = dict(
        layers=(8, 8, 16, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        **kwargs,
    )
    return _create_falcon("falcon_d4_448", pretrained=pretrained, **model_args)


@register_model
def falcon_d5_224(pretrained=False, **kwargs):
    model_args = dict(
        layers=(12, 12, 20, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        mlp_ratio=4,
        stem_hidden_dim=128,
        **kwargs,
    )
    return _create_falcon("falcon_d5_224", pretrained=pretrained, **model_args)


@register_model
def falcon_d5_448(pretrained=False, **kwargs):
    model_args = dict(
        layers=(12, 12, 20, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        mlp_ratio=4,
        stem_hidden_dim=128,
        **kwargs,
    )
    return _create_falcon("falcon_d5_448", pretrained=pretrained, **model_args)


@register_model
def falcon_d5_512(pretrained=False, **kwargs):
    model_args = dict(
        layers=(12, 12, 20, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        mlp_ratio=4,
        stem_hidden_dim=128,
        **kwargs,
    )
    return _create_falcon("falcon_d5_512", pretrained=pretrained, **model_args)
