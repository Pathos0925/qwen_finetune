from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModelForImageTextToText, PretrainedConfig

from model.modeling_cls import (
    Qwen2VLForSequenceClassification,
    Qwen2_5_VLForSequenceClassification,
    Qwen3VLForSequenceClassification,
    Qwen3_5ForSequenceClassification,
    Qwen3_5MoeForSequenceClassification,
)
from train.monkey_patch_forward import (
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen3_5_moe_with_mixed_modality_forward,
    replace_qwen3_5_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward,
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen_2_with_mixed_modality_forward,
)
from train.monkey_patch_loop import (
    LOOP_ADAPTER_ATTR,
    replace_qwen3_5_text_with_looped_forward,
    replace_qwen3_vl_text_with_looped_forward,
)
from train.monkey_patch_vision import replace_qwen2_5_vision
from model.looplm import LoopAdapter

_GENERATION_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
}

_PATCHERS = {
    "qwen2_vl": (replace_qwen_2_with_mixed_modality_forward,),
    "qwen2_5_vl": (
        replace_qwen2_5_with_mixed_modality_forward,
        replace_qwen2_5_vision,
    ),
    "qwen3_5": (replace_qwen3_5_with_mixed_modality_forward,),
    "qwen3_5_moe": (replace_qwen3_5_moe_with_mixed_modality_forward,),
    "qwen3_vl": (replace_qwen3_with_mixed_modality_forward,),
    "qwen3_vl_moe": (replace_qwen3_vl_moe_with_mixed_modality_forward,),
}

_SEQUENCE_CLASSIFICATION_MODEL_CLS = {
    "qwen2_vl": Qwen2VLForSequenceClassification,
    "qwen2_5_vl": Qwen2_5_VLForSequenceClassification,
    "qwen3_5": Qwen3_5ForSequenceClassification,
    "qwen3_5_moe": Qwen3_5MoeForSequenceClassification,
    "qwen3_vl": Qwen3VLForSequenceClassification,
}


def get_qwen_vl_generation_backbone(model):
    if not hasattr(model, "model"):
        raise TypeError(f"Unsupported generation model wrapper: {type(model)!r}")
    return model.model


def apply_qwen_vl_monkey_patches(model_type: str) -> str:
    try:
        patchers = _PATCHERS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_PATCHERS))
        raise ValueError(f"Unsupported Qwen-VL model_type: {model_type}. Supported: {supported}") from exc

    for patcher in patchers:
        patcher()

    return model_type


def load_qwen_vl_generation_model(
    model_name_or_path: str, *, config: PretrainedConfig | None = None, **kwargs: Any
):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type not in _GENERATION_MODEL_TYPES:
        supported = ", ".join(sorted(_GENERATION_MODEL_TYPES))
        raise ValueError(
            f"Unsupported Qwen-VL generation model_type: {config.model_type}. Supported: {supported}"
        )

    apply_qwen_vl_monkey_patches(config.model_type)
    return AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        config=config,
        **kwargs,
    )


_LOOP_PATCHERS = {
    "qwen3_5": replace_qwen3_5_text_with_looped_forward,
    "qwen3_vl": replace_qwen3_vl_text_with_looped_forward,
}


def install_loop_adapter(
    model,
    *,
    t_max: int,
    use_inter_norm: bool = True,
    kv_cache_strategy: str = "last",
):
    """Graft a LoopAdapter onto a loaded Qwen-VL model.

    Must be called *before* PEFT wraps the model so the gate's nn.Linear
    isn't mistaken for a LoRA target.
    """
    config = model.config
    if config.model_type not in _LOOP_PATCHERS:
        raise ValueError(
            f"loop_enable is only supported for model_type in "
            f"{sorted(_LOOP_PATCHERS)}; got {config.model_type!r}."
        )

    # Install the looped forward on the inner text model for this backbone.
    _LOOP_PATCHERS[config.model_type]()

    text_model = model.model.language_model
    text_config = text_model.config

    # Stamp loop knobs on the inner text config (the patched forward reads from it).
    text_config.loop_t_max = int(t_max)
    text_config.loop_kv_cache_strategy = kv_cache_strategy
    # Also stamp on the outer config so it survives `config.to_json_file`.
    config.loop_t_max = int(t_max)
    config.loop_kv_cache_strategy = kv_cache_strategy
    config.loop_enable = True

    d_model = text_config.hidden_size
    eps = getattr(text_config, "rms_norm_eps", 1e-6)
    adapter = LoopAdapter(d_model=d_model, eps=eps, use_inter_norm=use_inter_norm)
    adapter.to(dtype=next(text_model.parameters()).dtype, device=next(text_model.parameters()).device)
    text_model.add_module(LOOP_ADAPTER_ATTR, adapter)
    return adapter


def get_qwen_vl_sequence_classification_model_cls(model_type: str):
    try:
        return _SEQUENCE_CLASSIFICATION_MODEL_CLS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_SEQUENCE_CLASSIFICATION_MODEL_CLS))
        raise ValueError(
            f"Unsupported Qwen-VL sequence classification model_type: {model_type}. Supported: {supported}"
        ) from exc


def load_qwen_vl_sequence_classification_model(
    model_name_or_path: str, *, config: PretrainedConfig | None = None, **kwargs: Any
):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    apply_qwen_vl_monkey_patches(config.model_type)
    model_cls = get_qwen_vl_sequence_classification_model_cls(config.model_type)
    return model_cls.from_pretrained(
        model_name_or_path,
        config=config,
        **kwargs,
    )
