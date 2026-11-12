from __future__ import annotations

import unittest

import torch

from oreo_qwen.contracts import KGInputBatch
from oreo_qwen.reasoner import (
    GlobalInverseDegreeWalker,
    KGReasoner,
    KGReasonerConfig,
)


def make_config() -> KGReasonerConfig:
    return KGReasonerConfig(
        hidden_size=8,
        num_relations=2,
        num_kil_steps=1,
        knowledge_embedding_dim=4,
        adapter_bottleneck_dim=6,
    )


def make_kg(*, supervised: bool = True) -> KGInputBatch:
    optional = {}
    if supervised:
        optional = {
            "kg_relation_targets": torch.tensor([[[1.0, 0], [1.0, 0]]]),
            "kg_relation_target_mask": torch.tensor([[True, False]]),
        }
    return KGInputBatch(
        kg_mention_batch=torch.tensor([0, 1]),
        kg_rel_positions=torch.tensor([1, 1]),
        kg_tgt_positions=torch.tensor([2, 2]),
        kg_node_embeddings=torch.tensor(
            [
                [1.0, 0, 0, 0],
                [0, 1.0, 0, 0],
                [0, 0, 1.0, 0],
                [1.0, 1.0, 0, 0],
                [0, 1.0, 1.0, 0],
                [0, 0, 1.0, 1.0],
            ]
        ),
        kg_relation_embeddings=torch.tensor(
            [[1.0, 0, 0, 0], [0, 1.0, 0, 0]]
        ),
        kg_node_batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        kg_edge_src=torch.tensor([0, 1, 3, 4]),
        kg_edge_dst=torch.tensor([1, 2, 4, 5]),
        kg_edge_type=torch.tensor([0, 1, 0, 1]),
        kg_initial_node_indices=torch.tensor([0, 3]),
        **optional,
    )


class GlobalInverseDegreeWalkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.walker = GlobalInverseDegreeWalker()

    def test_single_relation_walk(self) -> None:
        previous = torch.tensor([[1.0, 0, 0]])
        result = self.walker(
            previous,
            torch.tensor([[1.0, 0]]),
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
        )
        torch.testing.assert_close(result, torch.tensor([[0.0, 1.0, 0.0]]))

    def test_global_normalization_and_inverse_degree(self) -> None:
        previous = torch.tensor([[0.75, 0.25, 0, 0]])
        result = self.walker(
            previous,
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([0, 0, 1]),
            torch.tensor([2, 3, 3]),
            torch.tensor([0, 1, 0]),
        )
        torch.testing.assert_close(result, torch.tensor([[0, 0, 0.375, 0.625]]))

    def test_zero_score_and_empty_edges_retain_state(self) -> None:
        previous = torch.tensor([[1.0, 0]])
        zero = self.walker(
            previous,
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([0]),
        )
        empty = self.walker(
            previous,
            torch.tensor([[1.0, 0]]),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
        torch.testing.assert_close(zero, previous)
        torch.testing.assert_close(empty, previous)


class KGReasonerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.config = make_config()
        self.reasoner = KGReasoner(self.config)
        self.hidden = torch.randn(2, 4, 8)
        self.kg = make_kg()

    def test_contains_no_trainable_entity_or_relation_table(self) -> None:
        parameter_names = dict(self.reasoner.named_parameters())
        self.assertFalse(any("entity_memory" in name for name in parameter_names))
        self.assertFalse(any("relation_memory" in name for name in parameter_names))
        self.assertFalse(any("embedding.weight" in name for name in parameter_names))

    def test_external_hard_initialization(self) -> None:
        state = self.reasoner.initialize(self.hidden, self.kg)
        expected = torch.tensor(
            [[1.0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0]]
        )
        torch.testing.assert_close(state.entity_probs, expected)
        torch.testing.assert_close(state.local_entity_embeddings.norm(dim=-1), torch.ones(6))

    def test_forward_shapes_loss_and_frozen_inputs(self) -> None:
        self.kg.kg_node_embeddings.requires_grad_(True)
        self.kg.kg_relation_embeddings.requires_grad_(True)
        state = self.reasoner.initialize(self.hidden, self.kg)
        output = self.reasoner(self.hidden, self.kg, state, 0)
        self.assertEqual(output.hidden_states.shape, self.hidden.shape)
        self.assertEqual(output.relation_probs.shape, (2, 2))
        self.assertEqual(output.retrieved_entity_embeddings.shape, (2, 4))
        self.assertEqual(output.projected_entity_values.shape, (2, 8))
        self.assertGreater(output.relation_loss.item(), 0.0)

        output.hidden_states.sum().backward()
        step = self.reasoner.steps[0]
        self.assertGreater(
            step.relation_router.query_projection.input.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            step.value_projection.projection.input.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertIsNone(self.kg.kg_node_embeddings.grad)
        self.assertIsNone(self.kg.kg_relation_embeddings.grad)

    def test_unsupervised_relation_loss_is_differentiable_zero(self) -> None:
        kg = make_kg(supervised=False)
        state = self.reasoner.initialize(self.hidden, kg)
        output = self.reasoner(self.hidden, kg, state, 0)
        self.assertEqual(output.relation_loss.item(), 0.0)
        self.assertTrue(output.relation_loss.requires_grad)

    def test_soft_initialization(self) -> None:
        kg = make_kg()
        kg.kg_initial_node_indices = None
        kg.kg_initial_entity_probs = torch.tensor(
            [[0.5, 0.5, 0, 0, 0, 0], [0, 0, 0, 0.25, 0.75, 0]]
        )
        state = self.reasoner.initialize(self.hidden, kg)
        torch.testing.assert_close(state.entity_probs, kg.kg_initial_entity_probs)


if __name__ == "__main__":
    unittest.main()
