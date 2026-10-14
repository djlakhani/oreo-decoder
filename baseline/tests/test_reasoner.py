from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from oreo_qwen.contracts import KGInputBatch
from oreo_qwen.reasoner import (
    GlobalInverseDegreeWalker,
    KGReasoner,
    KGReasonerConfig,
)


def make_config() -> KGReasonerConfig:
    return KGReasonerConfig(
        hidden_size=8,
        num_entities=32,
        num_relations=2,
        num_kil_steps=2,
        entity_embedding_dim=4,
        relation_embedding_dim=6,
    )


def make_kg(
    *,
    initialization: str = "hard",
    include_supervision: bool = True,
) -> KGInputBatch:
    optional: dict[str, torch.Tensor] = {}
    if initialization == "hard":
        optional["kg_initial_node_indices"] = torch.tensor([0, 3])
    elif initialization == "soft":
        optional["kg_initial_entity_probs"] = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
    elif initialization != "learned":
        raise ValueError(f"Unknown initialization mode: {initialization}")

    if include_supervision:
        optional.update(
            kg_entity_link_targets=torch.tensor([0, 3]),
            kg_entity_link_mask=torch.tensor([True, False]),
            kg_relation_targets=torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ]
            ),
            kg_relation_target_mask=torch.tensor(
                [[True, False], [False, True]],
            ),
        )

    return KGInputBatch(
        kg_mention_batch=torch.tensor([0, 1]),
        kg_src_positions=torch.tensor([0, 0]),
        kg_rel_positions=torch.tensor([1, 1]),
        kg_tgt_positions=torch.tensor([2, 2]),
        kg_node_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
        kg_node_batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        kg_edge_src=torch.tensor([0, 1, 3, 4]),
        kg_edge_dst=torch.tensor([1, 2, 4, 5]),
        kg_edge_type=torch.tensor([0, 1, 0, 1]),
        **optional,
    )


class GlobalInverseDegreeWalkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.walker = GlobalInverseDegreeWalker()

    def test_fixed_inverse_degree_and_global_normalization(self) -> None:
        previous = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])
        relation_probs = torch.tensor([[1.0]])
        edge_src = torch.tensor([0, 0, 1])
        edge_dst = torch.tensor([2, 3, 4])
        edge_type = torch.tensor([0, 0, 0])

        actual = self.walker(
            previous,
            relation_probs,
            edge_src,
            edge_dst,
            edge_type,
        )

        # Scores are .5/2, .5/2, and .5/1 before global normalization.
        expected = torch.tensor([[0.0, 0.0, 0.25, 0.25, 0.5]])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(dim=-1), torch.ones(1))

    def test_relation_probabilities_reweight_edges(self) -> None:
        previous = torch.tensor([[1.0, 0.0, 0.0]])
        relation_probs = torch.tensor([[0.8, 0.2]])

        actual = self.walker(
            previous,
            relation_probs,
            torch.tensor([0, 0]),
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
        )

        torch.testing.assert_close(actual, torch.tensor([[0.0, 0.8, 0.2]]))

    def test_partial_dead_end_mass_is_discarded_and_renormalized(self) -> None:
        previous = torch.tensor([[0.5, 0.5, 0.0]])
        actual = self.walker(
            previous,
            torch.tensor([[1.0]]),
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([0]),
        )
        torch.testing.assert_close(actual, torch.tensor([[0.0, 0.0, 1.0]]))

    def test_all_zero_edge_score_retains_previous_state(self) -> None:
        previous = torch.tensor([[1.0, 0.0, 0.0]])
        actual = self.walker(
            previous,
            torch.tensor([[1.0]]),
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([0]),
        )
        torch.testing.assert_close(actual, previous)

    def test_empty_edge_list_retains_previous_state(self) -> None:
        previous = torch.tensor([[0.3, 0.7]])
        actual = self.walker(
            previous,
            torch.tensor([[1.0]]),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
        self.assertIs(actual, previous)

    def test_disjoint_components_do_not_mix(self) -> None:
        previous = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        actual = self.walker(
            previous,
            torch.ones(2, 1),
            torch.tensor([0, 3]),
            torch.tensor([1, 4]),
            torch.tensor([0, 0]),
        )
        expected = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        torch.testing.assert_close(actual, expected)


class KGReasonerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = make_config()
        self.reasoner = KGReasoner(self.config)
        self.hidden_states = torch.randn(2, 4, self.config.hidden_size)

    def test_config_exposes_matching_contract_spec(self) -> None:
        spec = self.config.contract_spec
        self.assertEqual(spec.num_entities, self.config.num_entities)
        self.assertEqual(spec.num_relations, self.config.num_relations)
        self.assertEqual(spec.num_kil_steps, self.config.num_kil_steps)

    def test_hard_initialization_creates_one_hot_state(self) -> None:
        output = self.reasoner.initialize(self.hidden_states, make_kg(initialization="hard"))
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        torch.testing.assert_close(output.state.entity_probs, expected)
        self.assertIsNotNone(output.entity_logits)

    def test_soft_initialization_is_used_without_modification(self) -> None:
        kg = make_kg(initialization="soft", include_supervision=False)
        output = self.reasoner.initialize(self.hidden_states, kg)
        torch.testing.assert_close(
            output.state.entity_probs,
            kg.kg_initial_entity_probs,
        )
        self.assertIsNone(output.entity_logits)

    def test_soft_initialization_is_exactly_component_isolated(self) -> None:
        kg = make_kg(initialization="soft", include_supervision=False)
        kg.kg_initial_entity_probs[0, 0] -= 1e-6
        kg.kg_initial_entity_probs[0, 3] = 1e-6
        output = self.reasoner.initialize(self.hidden_states, kg)
        torch.testing.assert_close(
            output.state.entity_probs[0, 3:],
            torch.zeros(3),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            output.state.entity_probs.sum(dim=-1),
            torch.ones(2),
        )

    def test_learned_initialization_is_component_masked(self) -> None:
        kg = make_kg(initialization="learned", include_supervision=False)
        output = self.reasoner.initialize(self.hidden_states, kg)

        self.assertIsNotNone(output.entity_logits)
        torch.testing.assert_close(
            output.state.entity_probs.sum(dim=-1),
            torch.ones(2),
        )
        torch.testing.assert_close(
            output.state.entity_probs[0, 3:],
            torch.zeros(3),
        )
        torch.testing.assert_close(
            output.state.entity_probs[1, :3],
            torch.zeros(3),
        )

    def test_entity_link_loss_respects_mask(self) -> None:
        kg = make_kg(initialization="hard", include_supervision=True)
        output = self.reasoner.initialize(self.hidden_states, kg)
        expected = F.cross_entropy(
            output.entity_logits[:1],
            kg.kg_entity_link_targets[:1],
        )
        torch.testing.assert_close(output.entity_link_loss, expected)

    def test_step_walks_aggregates_raw_entities_and_only_updates_targets(self) -> None:
        kg = make_kg(initialization="hard", include_supervision=True)
        initialization = self.reasoner.initialize(self.hidden_states, kg)
        output = self.reasoner(
            self.hidden_states,
            kg,
            initialization.state,
            step_index=0,
        )

        expected_state = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        torch.testing.assert_close(output.state.entity_probs, expected_state)

        local_embeddings = self.reasoner.entity_memory(kg.kg_node_ids)
        expected_retrieved = local_embeddings.index_select(0, torch.tensor([1, 4]))
        torch.testing.assert_close(
            output.retrieved_entity_embeddings,
            expected_retrieved,
        )

        changed = torch.zeros(2, 4, dtype=torch.bool)
        changed[kg.kg_mention_batch, kg.kg_tgt_positions] = True
        torch.testing.assert_close(
            output.hidden_states[~changed],
            self.hidden_states[~changed],
        )
        self.assertFalse(
            torch.allclose(
                output.hidden_states[changed],
                self.hidden_states[changed],
            )
        )

    def test_second_step_reaches_two_hop_entities(self) -> None:
        kg = make_kg(initialization="hard", include_supervision=True)
        initialization = self.reasoner.initialize(self.hidden_states, kg)
        first = self.reasoner(self.hidden_states, kg, initialization.state, 0)
        second = self.reasoner(first.hidden_states, kg, first.state, 1)
        expected = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        torch.testing.assert_close(second.state.entity_probs, expected)

    def test_relation_loss_respects_step_and_mention_masks(self) -> None:
        kg = make_kg(initialization="hard", include_supervision=True)
        initialization = self.reasoner.initialize(self.hidden_states, kg)
        first = self.reasoner(self.hidden_states, kg, initialization.state, 0)
        first_expected = F.cross_entropy(
            first.relation_logits[:1],
            torch.tensor([0]),
        )
        torch.testing.assert_close(first.relation_loss, first_expected)

        second = self.reasoner(first.hidden_states, kg, first.state, 1)
        second_expected = F.cross_entropy(
            second.relation_logits[1:],
            torch.tensor([1]),
        )
        torch.testing.assert_close(second.relation_loss, second_expected)

    def test_step_parameters_are_distinct_but_relation_memory_is_shared(self) -> None:
        first, second = self.reasoner.steps
        self.assertIsNot(first.relation_router, second.relation_router)
        self.assertIsNot(first.value_projection, second.value_projection)
        self.assertIsNot(first.fusion, second.fusion)
        self.assertEqual(
            sum(1 for name, _ in self.reasoner.named_parameters() if name == "relation_memory.embedding.weight"),
            1,
        )

    def test_missing_auxiliary_targets_produce_differentiable_zero_losses(self) -> None:
        hidden_states = self.hidden_states.clone().requires_grad_(True)
        kg = make_kg(initialization="hard", include_supervision=False)
        initialization = self.reasoner.initialize(hidden_states, kg)
        step = self.reasoner(hidden_states, kg, initialization.state, 0)

        self.assertTrue(initialization.entity_link_loss.requires_grad)
        self.assertTrue(step.relation_loss.requires_grad)
        self.assertEqual(initialization.entity_link_loss.item(), 0.0)
        self.assertEqual(step.relation_loss.item(), 0.0)

    def test_gradients_reach_all_reasoner_subsystems_and_hidden_states(self) -> None:
        hidden_states = self.hidden_states.clone().requires_grad_(True)
        kg = make_kg(initialization="hard", include_supervision=True)
        initialization = self.reasoner.initialize(hidden_states, kg)
        first = self.reasoner(hidden_states, kg, initialization.state, 0)
        second = self.reasoner(first.hidden_states, kg, first.state, 1)

        loss = (
            initialization.entity_link_loss
            + first.relation_loss
            + second.relation_loss
            + second.hidden_states.square().mean()
        )
        loss.backward()

        self.assertIsNotNone(hidden_states.grad)
        self.assertGreater(hidden_states.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.reasoner.entity_memory.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.reasoner.relation_memory.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            self.reasoner.entity_linker.query_projection.input.weight.grad.abs().sum().item(),
            0.0,
        )
        for step in self.reasoner.steps:
            self.assertGreater(
                step.relation_router.query_projection.input.weight.grad.abs().sum().item(),
                0.0,
            )
            self.assertGreater(
                step.value_projection.projection.input.weight.grad.abs().sum().item(),
                0.0,
            )
            self.assertGreater(step.fusion.norm.weight.grad.abs().sum().item(), 0.0)

    def test_invalid_step_index_is_rejected(self) -> None:
        kg = make_kg()
        initialization = self.reasoner.initialize(self.hidden_states, kg)
        with self.assertRaises(IndexError):
            self.reasoner(
                self.hidden_states,
                kg,
                initialization.state,
                self.config.num_kil_steps,
            )


if __name__ == "__main__":
    unittest.main()
