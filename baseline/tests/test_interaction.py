from __future__ import annotations

import unittest

import torch
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache

from oreo_qwen.contracts import KGInputBatch
from oreo_qwen.interaction import (
    CausalTentBroadcastLayer,
    MarkerContextualizationLayer,
    ensure_tent_cache_layers,
)
from oreo_qwen.reasoner import KGReasonerConfig


def make_configs(*, attention_bias: bool = False) -> tuple[KGReasonerConfig, Qwen3Config]:
    reasoner = KGReasonerConfig(
        hidden_size=16,
        num_relations=2,
        num_kil_steps=1,
        knowledge_embedding_dim=4,
        adapter_bottleneck_dim=8,
    )
    qwen = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        attention_bias=attention_bias,
        attention_dropout=0.0,
    )
    return reasoner, qwen


def make_kg() -> KGInputBatch:
    return KGInputBatch(
        kg_mention_batch=torch.tensor([0]),
        kg_rel_positions=torch.tensor([2]),
        kg_tgt_positions=torch.tensor([3]),
        kg_node_embeddings=torch.randn(2, 4),
        kg_relation_embeddings=torch.randn(2, 4),
        kg_node_batch=torch.tensor([0, 0]),
        kg_edge_src=torch.tensor([0]),
        kg_edge_dst=torch.tensor([1]),
        kg_edge_type=torch.tensor([0]),
        kg_initial_node_indices=torch.tensor([0]),
    )


class MarkerContextualizationLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        reasoner, qwen = make_configs()
        self.layer = MarkerContextualizationLayer(reasoner, qwen).eval()
        self.layer.gate.data.fill_(1.0)
        self.hidden = torch.randn(1, 6, 16)
        self.kg = make_kg()
        # Position 5 is teacher-forced assistant content and is deliberately
        # excluded from the memory visible to the marker queries.
        self.prompt_mask = torch.tensor([[True, True, True, True, True, False]])

    def test_only_markers_are_updated_and_answer_is_not_visible(self) -> None:
        output = self.layer(self.hidden, self.kg, self.prompt_mask)
        marker_positions = torch.tensor([2, 3])
        nonmarker_positions = torch.tensor([0, 1, 4, 5])

        torch.testing.assert_close(
            output[:, nonmarker_positions],
            self.hidden[:, nonmarker_positions],
            atol=0,
            rtol=0,
        )
        self.assertGreater(
            (output[:, marker_positions] - self.hidden[:, marker_positions])
            .abs()
            .sum()
            .item(),
            0.0,
        )

        changed_answer = self.hidden.clone()
        changed_answer[:, 5] = 1_000 * torch.randn_like(changed_answer[:, 5])
        changed_output = self.layer(changed_answer, self.kg, self.prompt_mask)
        torch.testing.assert_close(
            changed_output[:, marker_positions],
            output[:, marker_positions],
            atol=0,
            rtol=0,
        )

    def test_markers_can_read_later_prompt_context(self) -> None:
        output = self.layer(self.hidden, self.kg, self.prompt_mask)
        changed_prompt = self.hidden.clone()
        changed_prompt[:, 4] += 10.0
        changed_output = self.layer(changed_prompt, self.kg, self.prompt_mask)

        self.assertGreater(
            (changed_output[:, 2:4] - output[:, 2:4]).abs().sum().item(),
            0.0,
        )


class CausalTentBroadcastLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        reasoner, qwen = make_configs(attention_bias=True)
        self.qwen_config = qwen
        self.layer = CausalTentBroadcastLayer(
            reasoner,
            qwen,
            cache_layer_index=qwen.num_hidden_layers,
        ).eval()
        self.layer.gate.data.fill_(1.0)
        self.layer.attention.o_proj.bias.data.fill_(0.5)
        self.hidden = torch.randn(1, 6, 16)
        self.kg = make_kg()

    def test_only_tokens_after_tent_receive_an_update(self) -> None:
        output = self.layer(
            self.hidden,
            self.kg,
            past_seen_tokens=0,
            past_key_values=None,
            use_cache=False,
        )

        # T_ENT is at position 3. Strict precedence means positions 0..3 have
        # no eligible memory and remain bit-for-bit identical.
        torch.testing.assert_close(output[:, :4], self.hidden[:, :4], atol=0, rtol=0)
        self.assertGreater((output[:, 4:] - self.hidden[:, 4:]).abs().sum().item(), 0.0)

    def test_non_tent_states_are_never_used_as_memory(self) -> None:
        output = self.layer(
            self.hidden,
            self.kg,
            past_seen_tokens=0,
            past_key_values=None,
            use_cache=False,
        )
        changed_non_tent = self.hidden.clone()
        changed_non_tent[:, 0] += 100.0
        changed_output = self.layer(
            changed_non_tent,
            self.kg,
            past_seen_tokens=0,
            past_key_values=None,
            use_cache=False,
        )
        torch.testing.assert_close(changed_output[:, 5], output[:, 5], atol=0, rtol=0)

    def test_cached_decode_matches_full_sequence_attention(self) -> None:
        full = self.layer(
            self.hidden,
            self.kg,
            past_seen_tokens=0,
            past_key_values=None,
            use_cache=False,
        )

        cache = DynamicCache(config=self.qwen_config)
        ensure_tent_cache_layers(cache, self.qwen_config.num_hidden_layers + 1)
        self.layer(
            self.hidden[:, :5],
            self.kg,
            past_seen_tokens=0,
            past_key_values=cache,
            use_cache=True,
        )
        decoded = self.layer(
            self.hidden[:, 5:],
            self.kg,
            past_seen_tokens=5,
            past_key_values=cache,
            use_cache=True,
        )
        torch.testing.assert_close(decoded, full[:, 5:], atol=1e-6, rtol=1e-6)

    def test_uses_qwen_grouped_query_head_layout_and_is_trainable(self) -> None:
        attention = self.layer.attention
        self.assertEqual(attention.num_attention_heads, 4)
        self.assertEqual(attention.num_key_value_heads, 2)
        self.assertEqual(attention.head_dim, 4)

        two_tent_kg = KGInputBatch(
            kg_mention_batch=torch.tensor([0, 0]),
            kg_rel_positions=torch.tensor([0, 2]),
            kg_tgt_positions=torch.tensor([1, 3]),
            kg_node_embeddings=torch.randn(2, 4),
            kg_relation_embeddings=torch.randn(2, 4),
            kg_node_batch=torch.tensor([0, 0]),
            kg_edge_src=torch.tensor([0]),
            kg_edge_dst=torch.tensor([1]),
            kg_edge_type=torch.tensor([0]),
            kg_initial_node_indices=torch.tensor([0, 0]),
        )
        output = self.layer(
            self.hidden.requires_grad_(),
            two_tent_kg,
            past_seen_tokens=0,
            past_key_values=None,
            use_cache=False,
        )
        output[:, 4:].sum().backward()
        self.assertGreater(attention.q_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(attention.k_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(attention.v_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(attention.o_proj.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
