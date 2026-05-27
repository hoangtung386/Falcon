"""Model creation and checkpoint loading utilities.

Provides:
- ``create_model``: Build a Falcon model from a name + optional checkpoint.
- ``load_checkpoint``: Load state dict with optional key remapping / filtering.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import timm
from falcon.model.falcon_model import *  # noqa: F401, F403
from timm.layers import set_layer_config
from timm.models._factory import parse_model_name
from timm.models._helpers import load_state_dict
from timm.models._hub import load_model_config_from_hf
from timm.models._pretrained import PretrainedCfg
from timm.models._registry import (
    is_model,
    model_entrypoint,
    split_model_name_tag,
)

__all__ = ["create_model", "load_checkpoint"]


def load_checkpoint(
    model,
    checkpoint_path: str,
    use_ema: bool = True,
    strict: bool = True,
    remap: bool = False,
    filter_keys: list[str] | None = None,
    state_dict_map: dict[str, str] | None = None,
):
    """Load a PyTorch checkpoint into a model.

    Supports:
    - ``.npz`` / ``.npy`` via ``model.load_pretrained()``.
    - ``.pth`` / ``.pth.tar`` via standard ``load_state_dict``.
    - Optional key filtering and remapping.

    Args:
        model: The model module to load into.
        checkpoint_path: Path to the checkpoint file.
        use_ema: Whether to load the EMA copy of weights.
        strict: Whether to enforce strict key matching.
        remap: Unused (kept for backwards compatibility).
        filter_keys: Keys containing any of these substrings will be removed.
        state_dict_map: Mapping ``{target_key: source_substring}`` for renaming.

    Returns:
        ``IncompatibleKeys`` namedtuple from ``load_state_dict``.
    """
    ext = os.path.splitext(checkpoint_path)[-1].lower()
    if ext in (".npz", ".npy"):
        if hasattr(model, "load_pretrained"):
            timm.models._model_builder.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError("Model cannot load numpy checkpoint")
        return

    state_dict = load_state_dict(checkpoint_path, use_ema)

    if filter_keys:
        for sd_key in list(state_dict.keys()):
            for filter_key in filter_keys:
                if filter_key in sd_key and sd_key in state_dict:
                    del state_dict[sd_key]

    rep = []
    if state_dict_map is not None:
        for state_k in list(state_dict.keys()):
            for target_k, target_v in state_dict_map.items():
                if target_v in state_k:
                    target_name = state_k.replace(target_v, target_k)
                    state_dict[target_name] = state_dict[state_k]
                    rep.append(state_k)
        for r in rep:
            if r in state_dict:
                del state_dict[r]

    incompatible_keys = model.load_state_dict(
        state_dict,
        strict=strict if filter_keys is None else False,
    )
    return incompatible_keys


def create_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    checkpoint_path: str = "",
    scriptable: Optional[bool] = None,
    exportable: Optional[bool] = None,
    no_jit: Optional[bool] = None,
    filter_keys=None,
    state_dict_map=None,
    **kwargs,
):
    """Create a Falcon model by name, optionally loading a checkpoint.

    Args:
        model_name: Name registered via ``@register_model`` (e.g.
            ``"falcon_d1_224"``).
        pretrained: Whether to load pretrained weights from the registry.
        pretrained_cfg: Custom pretrained config.
        pretrained_cfg_overlay: Overlay config on top of the default.
        checkpoint_path: Path to a local checkpoint.
        scriptable: Whether to make model scriptable (TorchScript).
        exportable: Whether to prepare for export.
        no_jit: Disable JIT.
        filter_keys: Forwarded to ``load_checkpoint``.
        state_dict_map: Forwarded to ``load_checkpoint``.

    Returns:
        The constructed model (with loaded weights if checkpoint provided).
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == "hf-hub":
        assert not pretrained_cfg
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_name)
        if not pretrained_cfg:
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError(f"Unknown model ({model_name})")

    create_fn = model_entrypoint(model_name)
    with set_layer_config(
        scriptable=scriptable, exportable=exportable, no_jit=no_jit
    ):
        model = create_fn(
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            **kwargs,
        )

    if checkpoint_path:
        load_checkpoint(
            model,
            checkpoint_path,
            filter_keys=filter_keys,
            state_dict_map=state_dict_map,
        )

    return model
