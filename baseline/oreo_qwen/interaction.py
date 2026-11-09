"""Trainable attention adapters surrounding each Knowledge Interaction Layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Qwen3Config
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from .contracts import KGInputBatch
from .reasoner import KGReasonerConfig


class GroupedQueryCrossAttention(nn.Module):
    """Qwen-shaped grouped-query attention with an explicit boolean mask."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_dropout: float,
        attention_bias: bool,
    ) -> None:
        super().__init__()
        if num_attention_heads % num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads."
            )
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads * head_dim,
            bias=attention_bias,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        query_states: Tensor,
        memory_states: Tensor,
        allowed_attention: Tensor,
        *,
        past_key_values: Cache | None = None,
        cache_layer_index: int | None = None,
    ) -> Tensor:
        batch_size, query_length, _ = query_states.shape
        memory_length = memory_states.shape[1]

        query = self.q_proj(query_states).view(
            batch_size,
            query_length,
            self.num_attention_heads,
            self.head_dim,
        )
        query = self.q_norm(query).transpose(1, 2)

        key = self.k_proj(memory_states).view(
            batch_size,
            memory_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        key = self.k_norm(key).transpose(1, 2)
        value = self.v_proj(memory_states).view(
            batch_size,
            memory_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        if past_key_values is not None:
            if cache_layer_index is None:
                raise ValueError("cache_layer_index is required when using a cache.")
            key, value = past_key_values.update(
                key,
                value,
                cache_layer_index,
            )

        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        expected_mask_shape = (batch_size, query_length, key.shape[-2])
        if tuple(allowed_attention.shape) != expected_mask_shape:
            raise ValueError(
                "allowed_attention must have shape "
                f"{expected_mask_shape}, got {tuple(allowed_attention.shape)}."
            )

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        allowed = allowed_attention[:, None, :, :]
        has_memory = allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probabilities = torch.where(
            has_memory,
            probabilities,
            torch.zeros_like(probabilities),
        )
        probabilities = F.dropout(
            probabilities,
            p=self.attention_dropout,
            training=self.training,
        )

        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).reshape(
            batch_size,
            query_length,
            self.num_attention_heads * self.head_dim,
        )
        output = self.o_proj(output)
        # A biased output projection must not move a residual stream whose
        # query has no eligible memory. This preserves exact identity for
        # tokens with no strictly preceding T_ENT marker.
        return output * allowed_attention.any(dim=-1, keepdim=True).to(output.dtype)


class MarkerContextualizationLayer(nn.Module):
    """Let Oreo marker queries attend bidirectionally to prompt-only context."""

    def __init__(self, config: KGReasonerConfig, qwen_config: Qwen3Config) -> None:
        super().__init__()
        self.query_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.memory_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryCrossAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=qwen_config.num_attention_heads,
            num_key_value_heads=qwen_config.num_key_value_heads,
            head_dim=qwen_config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_dropout=qwen_config.attention_dropout,
            attention_bias=qwen_config.attention_bias,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: Tensor,
        kg: KGInputBatch,
        prompt_mask: Tensor,
    ) -> Tensor:
        marker_positions = torch.stack(
            (kg.kg_src_positions, kg.kg_rel_positions, kg.kg_tgt_positions),
            dim=1,
        )
        marker_batches = kg.kg_mention_batch[:, None].expand_as(marker_positions)
        marker_states = hidden_states[marker_batches, marker_positions]
        prompt_states = hidden_states.index_select(0, kg.kg_mention_batch)
        allowed = prompt_mask.index_select(0, kg.kg_mention_batch)
        allowed = allowed[:, None, :].expand(-1, marker_positions.shape[1], -1)

        delta = self.attention(
            self.query_norm(marker_states),
            self.memory_norm(prompt_states),
            allowed,
        )
        updated = marker_states + torch.tanh(self.gate) * delta

        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * sequence_length, hidden_size)
        flat_markers = marker_batches * sequence_length + marker_positions
        return flat_hidden.index_copy(
            0,
            flat_markers.reshape(-1),
            updated.reshape(-1, hidden_size),
        ).reshape_as(hidden_states)


class CausalTentBroadcastLayer(nn.Module):
    """Add attention over strictly preceding post-KIL target-marker states."""

    def __init__(
        self,
        config: KGReasonerConfig,
        qwen_config: Qwen3Config,
        *,
        cache_layer_index: int,
    ) -> None:
        super().__init__()
        self.cache_layer_index = cache_layer_index
        self.query_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.memory_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryCrossAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=qwen_config.num_attention_heads,
            num_key_value_heads=qwen_config.num_key_value_heads,
            head_dim=qwen_config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_dropout=qwen_config.attention_dropout,
            attention_bias=qwen_config.attention_bias,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: Tensor,
        kg: KGInputBatch,
        *,
        past_seen_tokens: int,
        past_key_values: Cache | None,
        use_cache: bool,
    ) -> Tensor:
        batch_size, query_length, _ = hidden_states.shape
        cache = past_key_values if use_cache else None
        cached_length = _cache_layer_length(cache, self.cache_layer_index)
        if cache is not None and cached_length != past_seen_tokens:
            raise ValueError(
                "The TENT-attention cache is inconsistent with the Qwen cache: "
                f"expected {past_seen_tokens} cached tokens, found {cached_length}."
            )

        key_length = cached_length + query_length
        tent_keys = torch.zeros(
            (batch_size, key_length),
            dtype=torch.bool,
            device=hidden_states.device,
        )
        valid_tents = kg.kg_tgt_positions < key_length
        tent_keys[
            kg.kg_mention_batch[valid_tents],
            kg.kg_tgt_positions[valid_tents],
        ] = True

        query_positions = (
            torch.arange(query_length, device=hidden_states.device)
            + past_seen_tokens
        )
        key_positions = torch.arange(key_length, device=hidden_states.device)
        strictly_preceding = key_positions[None, :] < query_positions[:, None]
        allowed = tent_keys[:, None, :] & strictly_preceding[None, :, :]

        delta = self.attention(
            self.query_norm(hidden_states),
            self.memory_norm(hidden_states),
            allowed,
            past_key_values=cache,
            cache_layer_index=self.cache_layer_index if cache is not None else None,
        )
        return hidden_states + torch.tanh(self.gate) * delta


def ensure_tent_cache_layers(cache: Cache, required_layers: int) -> None:
    """Extend a dynamic cache with slots for the post-KIL attention adapters."""

    if not isinstance(cache, DynamicCache):
        return
    while len(cache.layers) < required_layers:
        cache.layers.append(DynamicLayer())


def _cache_layer_length(cache: Cache | None, layer_index: int) -> int:
    if cache is None:
        return 0
    layers = getattr(cache, "layers", None)
    if layers is None or len(layers) <= layer_index:
        return 0
    return layers[layer_index].get_seq_length()


__all__ = [
    "CausalTentBroadcastLayer",
    "GroupedQueryCrossAttention",
    "MarkerContextualizationLayer",
    "ensure_tent_cache_layers",
]
