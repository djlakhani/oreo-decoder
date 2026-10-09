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


SPECIAL_IDS = OreoSpecialTokenIds(
    source_entity=10,
    relation=11,
    target_entity=12,
    reconstruct=14,
    answer=13,
    masks=tuple(range(100, 116)),
)
SPEC = KGContractSpec(num_entities=1_000, num_relations=2, num_kil_steps=2)


def make_valid_batch(*, use_soft_initialization: bool = False) -> OreoQwenBatch:
    input_ids = torch.tensor(
        [
            [10, 20, 11, 12, 30, 13, 40, 2, 0],
            [10, 21, 11, 12, 31, 13, 41, 2, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0]],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [-100, -100, -100, -100, -100, -100, 40, 2, -100],
            [-100, -100, -100, -100, -100, -100, 41, 2, -100],
        ],
        dtype=torch.long,
    )

    initial_kwargs = {}
    if use_soft_initialization:
        initial_kwargs["kg_initial_entity_probs"] = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
    else:
        initial_kwargs["kg_initial_node_indices"] = torch.tensor([0, 3])

    kg = KGInputBatch(
        kg_mention_batch=torch.tensor([0, 1]),
        kg_src_positions=torch.tensor([0, 0]),
        kg_rel_positions=torch.tensor([2, 2]),
        kg_tgt_positions=torch.tensor([3, 3]),
        kg_node_ids=torch.tensor([101, 205, 310, 402, 510, 611]),
        kg_node_batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        kg_edge_src=torch.tensor([0, 1, 3, 4]),
        kg_edge_dst=torch.tensor([1, 2, 4, 5]),
        kg_edge_type=torch.tensor([0, 1, 0, 1]),
        kg_entity_link_targets=torch.tensor([0, 3]),
        kg_entity_link_mask=torch.tensor([True, True]),
        kg_relation_targets=torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        ),
        kg_relation_target_mask=torch.tensor(
            [[True, True], [True, True]],
        ),
        **initial_kwargs,
    )
    return OreoQwenBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        kg=kg,
    )


class OreoQwenBatchTests(unittest.TestCase):
    def validate(self, batch: OreoQwenBatch) -> None:
        batch.validate(
            spec=SPEC,
            special_token_ids=SPECIAL_IDS,
            expected_delimiter_token_id=SPECIAL_IDS.answer,
        )

    def test_valid_hard_initialization_and_flattened_kwargs(self) -> None:
        batch = make_valid_batch()
        self.validate(batch)

        kwargs = batch.as_model_kwargs()
        self.assertIn("input_ids", kwargs)
        self.assertIn("kg_node_batch", kwargs)
        self.assertIn("kg_initial_node_indices", kwargs)
        self.assertNotIn("kg_initial_entity_probs", kwargs)
        self.assertNotIn("kg", kwargs)

    def test_valid_soft_initialization(self) -> None:
        self.validate(make_valid_batch(use_soft_initialization=True))

    def test_missing_initialization_uses_learned_predictor_contract(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_initial_node_indices = None
        self.validate(batch)

    def test_graph_free_raw_qwen_batch_is_valid(self) -> None:
        source = make_valid_batch()
        batch = OreoQwenBatch(
            input_ids=source.input_ids,
            attention_mask=source.attention_mask,
            labels=source.labels,
        )
        batch.validate(
            special_token_ids=SPECIAL_IDS,
            expected_delimiter_token_id=SPECIAL_IDS.answer,
        )

    def test_reconstruction_suffix_must_start_with_first_sentinel(self) -> None:
        batch = make_valid_batch()
        batch.input_ids[:, 5] = SPECIAL_IDS.reconstruct
        batch.input_ids[:, 6] = SPECIAL_IDS.masks[0]
        batch.labels[:, 6] = SPECIAL_IDS.masks[0]
        batch.validate(
            spec=SPEC,
            special_token_ids=SPECIAL_IDS,
            expected_delimiter_token_id=SPECIAL_IDS.reconstruct,
        )

        batch.input_ids[0, 6] = SPECIAL_IDS.masks[1]
        batch.labels[0, 6] = SPECIAL_IDS.masks[1]
        with self.assertRaisesRegex(ContractError, "must begin"):
            batch.validate(
                spec=SPEC,
                special_token_ids=SPECIAL_IDS,
                expected_delimiter_token_id=SPECIAL_IDS.reconstruct,
            )

    def test_cross_component_edge_is_rejected(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_edge_dst[0] = 4
        with self.assertRaisesRegex(ContractError, "cross batch-example components"):
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
        with self.assertRaisesRegex(ContractError, "outside a mention's component"):
            self.validate(batch)

    def test_hard_and_soft_initialization_are_mutually_exclusive(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_initial_entity_probs = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        with self.assertRaisesRegex(ContractError, "either"):
            self.validate(batch)

    def test_marker_position_must_contain_expected_token(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_rel_positions[0] = 1
        with self.assertRaisesRegex(ContractError, "token ID"):
            self.validate(batch)

    def test_marker_position_may_not_point_to_padding(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_tgt_positions[0] = 8
        with self.assertRaisesRegex(ContractError, "padding position"):
            self.validate(batch)

    def test_duplicate_fusion_target_is_rejected(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_mention_batch[1] = 0
        batch.kg.kg_initial_node_indices[1] = 0
        batch.kg.kg_entity_link_targets[1] = 0
        with self.assertRaisesRegex(ContractError, "must be unique"):
            self.validate(batch)

    def test_prompt_label_is_rejected(self) -> None:
        batch = make_valid_batch()
        batch.labels[0, 4] = batch.input_ids[0, 4]
        with self.assertRaisesRegex(ContractError, "Prompt and delimiter"):
            self.validate(batch)

    def test_ignored_suffix_label_is_rejected(self) -> None:
        batch = make_valid_batch()
        batch.labels[0, 6] = -100
        with self.assertRaisesRegex(ContractError, "Every active suffix"):
            self.validate(batch)

    def test_relation_target_shape_tracks_kil_count(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_relation_targets = batch.kg.kg_relation_targets[:1]
        with self.assertRaisesRegex(ContractError, "must have shape"):
            self.validate(batch)

    def test_auxiliary_target_and_mask_must_be_paired(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_entity_link_mask = None
        with self.assertRaisesRegex(ContractError, "must be supplied together"):
            self.validate(batch)

    def test_empty_edge_list_is_valid(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_edge_src = torch.empty(0, dtype=torch.long)
        batch.kg.kg_edge_dst = torch.empty(0, dtype=torch.long)
        batch.kg.kg_edge_type = torch.empty(0, dtype=torch.long)
        self.validate(batch)

    def test_input_ids_and_inputs_embeds_are_mutually_exclusive(self) -> None:
        batch = make_valid_batch()
        batch.inputs_embeds = torch.zeros(2, 9, 8)
        with self.assertRaisesRegex(ContractError, "exactly one"):
            self.validate(batch)

    def test_invalid_integer_dtype_is_rejected(self) -> None:
        batch = make_valid_batch()
        batch.kg.kg_node_ids = batch.kg.kg_node_ids.to(torch.int32)
        with self.assertRaisesRegex(ContractError, "torch.long"):
            self.validate(batch)


class ContractSpecTests(unittest.TestCase):
    def test_dimensions_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            KGContractSpec(num_entities=0, num_relations=2, num_kil_steps=2)


if __name__ == "__main__":
    unittest.main()
