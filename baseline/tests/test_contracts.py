from __future__ import annotations

import unittest

import torch

from oreo_qwen.contracts import (
    ContractError,
    KGContractSpec,
    KGInputBatch,
    OreoQwenBatch,
)
from oreo_qwen.tokens import OreoSpecialTokenIds


SPECIAL_IDS = OreoSpecialTokenIds(relation=11, target_entity=12)
SPEC = KGContractSpec(
    num_relations=2,
    num_kil_steps=1,
    knowledge_embedding_dim=4,
)


def make_valid_batch(*, use_soft_initialization: bool = False) -> OreoQwenBatch:
    input_ids = torch.tensor(
        [[20, 11, 12, 30, 50, 40, 2, 0], [21, 11, 12, 31, 51, 41, 2, 0]]
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0]]
    )
    labels = torch.tensor(
        [
            [-100, -100, -100, -100, -100, 40, 2, -100],
            [-100, -100, -100, -100, -100, 41, 2, -100],
        ]
    )
    initial: dict[str, torch.Tensor]
    if use_soft_initialization:
        initial = {
            "kg_initial_entity_probs": torch.tensor(
                [[1.0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0]]
            )
        }
    else:
        initial = {"kg_initial_node_indices": torch.tensor([0, 3])}
    kg = KGInputBatch(
        kg_mention_batch=torch.tensor([0, 1]),
        kg_rel_positions=torch.tensor([1, 1]),
        kg_tgt_positions=torch.tensor([2, 2]),
        kg_node_embeddings=torch.arange(24, dtype=torch.float32).reshape(6, 4),
        kg_relation_embeddings=torch.tensor(
            [[1.0, 0, 0, 0], [0, 1.0, 0, 0]]
        ),
        kg_node_batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        kg_edge_src=torch.tensor([0, 1, 3, 4]),
        kg_edge_dst=torch.tensor([1, 2, 4, 5]),
        kg_edge_type=torch.tensor([0, 1, 0, 1]),
        kg_relation_targets=torch.tensor([[[1.0, 0], [1.0, 0]]]),
        kg_relation_target_mask=torch.tensor([[True, True]]),
        **initial,
    )
    return OreoQwenBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        assistant_content_start_positions=torch.tensor([5, 5]),
        kg=kg,
    )


class OreoQwenBatchTests(unittest.TestCase):
    def validate(self, batch: OreoQwenBatch) -> None:
        batch.validate(spec=SPEC, special_token_ids=SPECIAL_IDS)

    def test_valid_frozen_embedding_contract(self) -> None:
        batch = make_valid_batch()
        self.validate(batch)
        kwargs = batch.as_model_kwargs()
        self.assertIn("kg_node_embeddings", kwargs)
        self.assertIn("kg_relation_embeddings", kwargs)
        self.assertNotIn("kg_node_ids", kwargs)
        self.assertNotIn("kg_src_positions", kwargs)
        self.assertNotIn("kg_entity_link_targets", kwargs)

    def test_valid_soft_initialization(self) -> None:
        self.validate(make_valid_batch(use_soft_initialization=True))

    def test_external_initialization_is_required(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_initial_node_indices = None
        with self.assertRaisesRegex(ContractError, "exactly one"):
            self.validate(batch)

    def test_hard_and_soft_initialization_are_mutually_exclusive(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_initial_entity_probs = torch.tensor(
            [[1.0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0]]
        )
        with self.assertRaisesRegex(ContractError, "exactly one"):
            self.validate(batch)

    def test_embedding_dimensions_are_validated(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_node_embeddings = torch.zeros(6, 3)
        with self.assertRaisesRegex(ContractError, "shape"):
            self.validate(batch)

    def test_embeddings_must_be_finite_floating_tensors(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_node_embeddings[0, 0] = torch.nan
        with self.assertRaisesRegex(ContractError, "finite"):
            self.validate(batch)
        batch = make_valid_batch()
        batch.kg.kg_relation_embeddings = torch.ones(2, 4, dtype=torch.long)
        with self.assertRaisesRegex(ContractError, "floating"):
            self.validate(batch)

    def test_cross_component_edge_is_rejected(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_edge_dst[0] = 4
        with self.assertRaisesRegex(ContractError, "cross batch-example"):
            self.validate(batch)

    def test_hard_initial_node_must_belong_to_mention(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_initial_node_indices[0] = 3
        with self.assertRaisesRegex(ContractError, "owned by another"):
            self.validate(batch)

    def test_soft_state_cannot_cross_components(self) -> None:
        batch = make_valid_batch(use_soft_initialization=True)
        batch.kg.kg_initial_entity_probs[0, 0] = 0.5
        batch.kg.kg_initial_entity_probs[0, 3] = 0.5
        with self.assertRaisesRegex(ContractError, "outside a mention"):
            self.validate(batch)

    def test_marker_position_must_contain_expected_token(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_rel_positions[0] = 0
        with self.assertRaisesRegex(ContractError, "token ID"):
            self.validate(batch)

    def test_marker_may_not_point_to_padding_or_answer(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_tgt_positions[0] = 7
        with self.assertRaisesRegex(ContractError, "padding"):
            self.validate(batch)
        batch = make_valid_batch()
        batch.kg.kg_tgt_positions[0] = 5
        with self.assertRaisesRegex(ContractError, "before assistant"):
            self.validate(batch)

    def test_relation_target_shape_tracks_kil_count(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_relation_targets = torch.zeros(2, 2, 2)
        with self.assertRaisesRegex(ContractError, "shape"):
            self.validate(batch)

    def test_empty_edge_list_is_valid(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_edge_src = torch.empty(0, dtype=torch.long)
        batch.kg.kg_edge_dst = torch.empty(0, dtype=torch.long)
        batch.kg.kg_edge_type = torch.empty(0, dtype=torch.long)
        self.validate(batch)

    def test_graph_free_raw_qwen_batch_is_valid(self) -> None:
        source = make_valid_batch()
        OreoQwenBatch(
            input_ids=source.input_ids,
            attention_mask=source.attention_mask,
            labels=source.labels,
            assistant_content_start_positions=source.assistant_content_start_positions,
        ).validate(special_token_ids=SPECIAL_IDS)


class ContractSpecTests(unittest.TestCase):
    def test_dimensions_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            KGContractSpec(
                num_relations=0, num_kil_steps=1, knowledge_embedding_dim=4
            )


if __name__ == "__main__":
    unittest.main()
