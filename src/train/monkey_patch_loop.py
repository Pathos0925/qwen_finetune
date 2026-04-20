"""Monkey-patch the Qwen3.5 text model so its layer stack runs T_max times.

The patched forward keeps all the standard setup (embeddings, masks, RoPE)
exactly as upstream, then wraps the per-layer loop in an outer T_max loop.
After each pass it normalizes via the backbone's `self.norm` to get h^(t),
queries the LoopAdapter for λ_t, and (optionally) re-normalizes the residual
stream via an inter-loop RMSNorm before the next pass.

Per-step hidden states and λ_t values are stashed on a side-channel
attribute `self._last_loop_state` so the trainer's compute_loss can read
them without us having to also patch Qwen3_5Model.forward and
Qwen3_5ForConditionalGeneration.forward.

Stable assumptions (will break loudly if violated):
  - The backbone has `self.embed_tokens`, `self.layers`, `self.norm`,
    `self.rotary_emb`, and a `Qwen3_5DynamicCache`-style cache contract.
  - Position ids and masks do not depend on hidden state and can be reused
    unchanged across loop iterations.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen3_5_mod
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5TextModel,
)
from transformers.masking_utils import create_causal_mask


# Sentinel attribute names. Use leading underscore so they don't collide with
# anything PEFT/HF might want to wrap.
LOOP_STATE_ATTR = "_last_loop_state"
LOOP_ADAPTER_ATTR = "_loop_adapter"


def _looped_qwen3_5_text_forward(
    self: Qwen3_5TextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Qwen3_5ModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = Qwen3_5DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    causal_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    adapter = getattr(self, LOOP_ADAPTER_ATTR, None)
    t_max = int(getattr(self.config, "loop_t_max", 1))
    if adapter is None or t_max <= 1:
        # Loop disabled: behave exactly like upstream.
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_mask = (
                linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            )
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        if hasattr(self, LOOP_STATE_ATTR):
            setattr(self, LOOP_STATE_ATTR, None)
        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    if use_cache:
        # Re-running the same layers across t loop iterations would append
        # to per-layer KV / SSM caches t times, which is incorrect. The
        # inference cache-sharing strategy is deferred to a later milestone.
        raise NotImplementedError(
            "LoopLM forward currently supports use_cache=False only. "
            "Inference (use_cache=True) requires a per-loop cache wrapper."
        )

    per_step_hidden_states: List[torch.Tensor] = []
    per_step_lambdas: List[torch.Tensor] = []

    for t in range(t_max):
        h = hidden_states
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_mask = (
                linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            )
            h = decoder_layer(
                h,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                **kwargs,
            )

        h_post_norm = self.norm(h)
        per_step_hidden_states.append(h_post_norm)
        per_step_lambdas.append(adapter.lambda_at(h_post_norm))

        if t < t_max - 1:
            hidden_states = adapter.between_loops(h)
        else:
            hidden_states = h_post_norm

    setattr(
        self,
        LOOP_STATE_ATTR,
        {
            "per_step_hidden_states": per_step_hidden_states,
            "per_step_lambdas": per_step_lambdas,
        },
    )

    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def replace_qwen3_5_text_with_looped_forward() -> None:
    """Install the looped forward on Qwen3_5TextModel."""
    _qwen3_5_mod.Qwen3_5TextModel.forward = _looped_qwen3_5_text_forward


def get_loop_state(text_model) -> Optional[dict]:
    """Read and clear the per-step state stashed by the most recent forward."""
    state = getattr(text_model, LOOP_STATE_ATTR, None)
    return state
