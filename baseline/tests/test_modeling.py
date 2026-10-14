from __future__ import annotations

import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from oreo_qwen import (
    KGInputBatch,
    OreoQwen3Config,
    OreoQwen3ForCausalLM,
    register_oreo_tokens_and_resize,
)


def make_config(
    *,
    num_hidden_layers: int = 4,
    kil_layer_boundaries: tuple[int, ...] = (1, 3),
    output_kg_diagnostics: bool = False,
) -> OreoQwen3Config:
    return OreoQwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=None,
        tie_word_embeddings=True,
        num_entities=32,
        num_relations=2,
        kil_layer_boundaries=kil_layer_boundaries,
        entity_embedding_dim=4,
        relation_embedding_dim=6,
        output_kg_diagnostics=output_kg_diagnostics,
    )


def make_kg(*, num_steps: int = 2, supervision: bool = True) -> KGInputBatch:
    optional: dict[str, torch.Tensor] = {
        "kg_initial_node_indices": torch.tensor([0, 3]),
    }
    if supervision:
        relation_targets = torch.zeros(num_steps, 2, 2)
        for step in range(num_steps):
            relation_targets[step, :, step % 2] = 1.0
        optional.update(
            kg_entity_link_targets=torch.tensor([0, 3]),
            kg_entity_link_mask=torch.tensor([True, True]),
            kg_relation_targets=relation_targets,
            kg_relation_target_mask=torch.ones(num_steps, 2, dtype=torch.bool),
        )
    return KGInputBatch(
        kg_mention_batch=torch.tensor([0, 1]),
        kg_src_positions=torch.tensor([0, 0]),
        kg_rel_positions=torch.tensor([2, 2]),
        kg_tgt_positions=torch.tensor([3, 3]),
        kg_node_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
        kg_node_batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        kg_edge_src=torch.tensor([0, 1, 3, 4]),
        kg_edge_dst=torch.tensor([1, 2, 4, 5]),
        kg_edge_type=torch.tensor([0, 1, 0, 1]),
        **optional,
    )


def make_language_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23],
        ]
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :5] = -100
    return input_ids, attention_mask, labels


def make_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {"<unk>": 0, "<eos>": 1}
    vocabulary.update({f"token_{index}": index for index in range(2, 64)})
    backend = Tokenizer(
        WordLevel(
            vocab=vocabulary,
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
    )


class OreoQwen3ConfigTests(unittest.TestCase):
    def test_official_depth_defaults_to_nine_ten_nine_groups(self) -> None:
        config = make_config(num_hidden_layers=28, kil_layer_boundaries=(9, 19))
        self.assertEqual(config.kil_layer_boundaries, [9, 19])
        self.assertEqual(config.num_kil_steps, 2)
        self.assertEqual(config.reasoner_config.num_kil_steps, 2)

        automatic = OreoQwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
        self.assertEqual(automatic.kil_layer_boundaries, [9, 19])

    def test_boundaries_are_completed_layer_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            make_config(kil_layer_boundaries=(3, 1))
        with self.assertRaisesRegex(ValueError, "num_hidden_layers - 1"):
            make_config(kil_layer_boundaries=(4,))

    def test_native_qwen_config_conversion_preserves_architecture(self) -> None:
        native = Qwen3Config(
            vocab_size=71,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=8,
        )
        config = OreoQwen3Config.from_qwen3_config(
            native,
            num_entities=20,
            num_relations=5,
            kil_layer_boundaries=[1],
        )
        self.assertEqual(config.vocab_size, native.vocab_size)
        self.assertEqual(config.hidden_size, native.hidden_size)
        self.assertEqual(config.num_hidden_layers, native.num_hidden_layers)
        self.assertEqual(config.num_entities, 20)


class OreoQwen3ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.config = make_config()
        self.model = OreoQwen3ForCausalLM(self.config)
        self.input_ids, self.attention_mask, self.labels = make_language_batch()
        self.kg = make_kg()

    def test_two_kils_run_at_configured_boundaries(self) -> None:
        events: list[str] = []
        hooks = []
        for index, layer in enumerate(self.model.model.layers):
            hooks.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, _output, index=index: events.append(
                        f"layer-{index + 1}"
                    )
                )
            )
        hooks.append(
            self.model.model.reasoner.register_forward_hook(
                lambda _module, _inputs, _output: events.append("kil")
            )
        )
        try:
            self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                use_cache=False,
                **self.kg.as_model_kwargs(),
            )
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(
            events,
            ["layer-1", "kil", "layer-2", "layer-3", "kil", "layer-4"],
        )

    def test_forward_exposes_losses_and_diagnostics(self) -> None:
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            use_cache=False,
            output_kg_diagnostics=True,
            **self.kg.as_model_kwargs(),
        )

        self.assertEqual(output.logits.shape, (2, 7, self.config.vocab_size))
        self.assertIsNotNone(output.lm_loss)
        self.assertIsNotNone(output.entity_link_loss)
        self.assertIsNotNone(output.relation_loss)
        torch.testing.assert_close(
            output.loss,
            output.lm_loss + output.entity_link_loss + output.relation_loss,
        )
        self.assertEqual(len(output.entity_states), 3)  # initial, one-hop, two-hop
        self.assertEqual(len(output.relation_probs), 2)
        self.assertEqual(output.entity_states[-1].shape, (2, 6))
        self.assertEqual(output.relation_probs[-1].shape, (2, 2))

    def test_auxiliary_weights_control_total_loss(self) -> None:
        self.model.config.entity_link_loss_weight = 0.25
        self.model.config.relation_loss_weight = 2.0
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            use_cache=False,
            **self.kg.as_model_kwargs(),
        )
        expected = (
            output.lm_loss
            + 0.25 * output.entity_link_loss
            + 2.0 * output.relation_loss
        )
        torch.testing.assert_close(output.loss, expected)

    def test_unsupervised_kg_still_receives_lm_gradients(self) -> None:
        kg = make_kg(supervision=False)
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            use_cache=False,
            **kg.as_model_kwargs(),
        )
        torch.testing.assert_close(output.loss, output.lm_loss)
        output.loss.backward()
        self.assertGreater(
            self.model.model.reasoner.entity_memory.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_gradients_reach_qwen_reasoner_memories_and_marker_embeddings(self) -> None:
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            use_cache=False,
            **self.kg.as_model_kwargs(),
        )
        output.loss.backward()

        self.assertGreater(
            self.model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum().item(),
            0.0,
        )
        reasoner = self.model.model.reasoner
        self.assertGreater(reasoner.entity_memory.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(reasoner.relation_memory.weight.grad.abs().sum().item(), 0.0)
        for step in reasoner.steps:
            self.assertGreater(
                step.relation_router.query_projection.input.weight.grad.abs().sum().item(),
                0.0,
            )
            self.assertGreater(
                step.value_projection.projection.input.weight.grad.abs().sum().item(),
                0.0,
            )

        # Token 10 occupies a source-marker position and participates in the
        # supervised entity-linking path.
        self.assertGreater(
            self.model.model.embed_tokens.weight.grad[10].abs().sum().item(),
            0.0,
        )

    def test_graph_free_path_matches_native_qwen(self) -> None:
        native_config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=None,
            tie_word_embeddings=True,
        )
        oreo_config = OreoQwen3Config.from_qwen3_config(
            native_config,
            num_entities=8,
            num_relations=2,
            entity_embedding_dim=4,
            kil_layer_boundaries=[1],
        )
        native = Qwen3ForCausalLM(native_config).eval()
        oreo = OreoQwen3ForCausalLM(oreo_config).eval()
        incompatible = oreo.load_state_dict(native.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(
            all(key.startswith("model.reasoner.") for key in incompatible.missing_keys)
        )

        with torch.no_grad():
            native_logits = native(self.input_ids, use_cache=False).logits
            oreo_logits = oreo(self.input_ids, use_cache=False).logits
        torch.testing.assert_close(oreo_logits, native_logits, atol=0, rtol=0)

    def test_native_qwen_checkpoint_factory_loads_base_weights(self) -> None:
        native_config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=None,
            tie_word_embeddings=True,
        )
        native = Qwen3ForCausalLM(native_config).eval()
        with tempfile.TemporaryDirectory() as directory:
            native.save_pretrained(directory)
            oreo = OreoQwen3ForCausalLM.from_qwen3_pretrained(
                directory,
                oreo_config_overrides={
                    "num_entities": 8,
                    "num_relations": 2,
                    "entity_embedding_dim": 4,
                    "kil_layer_boundaries": [1],
                },
            ).eval()

        with torch.no_grad():
            native_logits = native(self.input_ids, use_cache=False).logits
            oreo_logits = oreo(self.input_ids, use_cache=False).logits
        torch.testing.assert_close(oreo_logits, native_logits, atol=0, rtol=0)
        self.assertEqual(oreo.model.reasoner.entity_memory.weight.shape, (8, 4))

    def test_partial_flat_graph_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "partial KG batch"):
            self.model(
                input_ids=self.input_ids,
                kg_node_ids=self.kg.kg_node_ids,
            )

    def test_output_hidden_states_and_cache_follow_hugging_face_api(self) -> None:
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            use_cache=True,
            output_hidden_states=True,
            **self.kg.as_model_kwargs(),
        )
        self.assertIsNotNone(output.past_key_values)
        self.assertEqual(output.past_key_values.get_seq_length(), 7)
        self.assertEqual(len(output.hidden_states), self.config.num_hidden_layers + 1)

    def test_save_load_round_trip_preserves_outputs(self) -> None:
        self.model.eval()
        with torch.no_grad():
            expected = self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                use_cache=False,
                output_kg_diagnostics=True,
                **self.kg.as_model_kwargs(),
            )
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            restored = OreoQwen3ForCausalLM.from_pretrained(directory).eval()
            with torch.no_grad():
                actual = restored(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    use_cache=False,
                    output_kg_diagnostics=True,
                    **self.kg.as_model_kwargs(),
                )

        self.assertIsInstance(restored.config, OreoQwen3Config)
        self.assertEqual(restored.config.kil_layer_boundaries, [1, 3])
        torch.testing.assert_close(actual.logits, expected.logits)
        torch.testing.assert_close(actual.entity_states[-1], expected.entity_states[-1])
        torch.testing.assert_close(actual.relation_probs[-1], expected.relation_probs[-1])

    def test_cached_greedy_generation_runs_kil_only_during_prefill(self) -> None:
        model = OreoQwen3ForCausalLM(
            make_config(num_hidden_layers=2, kil_layer_boundaries=(1,))
        ).eval()
        kg = make_kg(num_steps=1, supervision=False)
        calls: list[None] = []
        hook = model.model.reasoner.register_forward_hook(
            lambda _module, _inputs, _output: calls.append(None)
        )
        try:
            generated = model.generate(
                self.input_ids[:, :5],
                attention_mask=self.attention_mask[:, :5],
                max_new_tokens=2,
                do_sample=False,
                **kg.as_model_kwargs(),
            )
        finally:
            hook.remove()

        self.assertEqual(generated.shape, (2, 7))
        self.assertEqual(len(calls), 1)

    def test_token_registration_resizes_tied_embeddings_and_updates_config(self) -> None:
        tokenizer = make_tokenizer()
        model = OreoQwen3ForCausalLM(
            OreoQwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                num_entities=8,
                num_relations=2,
                kil_layer_boundaries=[1],
                entity_embedding_dim=4,
                tie_word_embeddings=True,
            )
        )
        token_ids = register_oreo_tokens_and_resize(model, tokenizer)

        self.assertEqual(model.model.embed_tokens.num_embeddings, len(tokenizer))
        self.assertEqual(model.lm_head.out_features, len(tokenizer))
        self.assertEqual(model.config.vocab_size, len(tokenizer))
        self.assertEqual(model.config.oreo_special_token_ids, token_ids)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

        input_ids = torch.tensor(
            [[
                token_ids.source_entity,
                2,
                token_ids.relation,
                token_ids.target_entity,
                token_ids.reconstruct,
                token_ids.masks[0],
                3,
            ]]
        )
        labels = input_ids.clone()
        labels[:, :5] = -100
        kg = KGInputBatch(
            kg_mention_batch=torch.tensor([0]),
            kg_src_positions=torch.tensor([0]),
            kg_rel_positions=torch.tensor([2]),
            kg_tgt_positions=torch.tensor([3]),
            kg_node_ids=torch.tensor([1, 2]),
            kg_node_batch=torch.tensor([0, 0]),
            kg_edge_src=torch.tensor([0]),
            kg_edge_dst=torch.tensor([1]),
            kg_edge_type=torch.tensor([0]),
            kg_initial_node_indices=torch.tensor([0]),
            kg_entity_link_targets=torch.tensor([0]),
            kg_entity_link_mask=torch.tensor([True]),
            kg_relation_targets=torch.tensor([[[1.0, 0.0]]]),
            kg_relation_target_mask=torch.tensor([[True]]),
        )
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            use_cache=False,
            **kg.as_model_kwargs(),
        )
        output.loss.backward()
        self.assertGreater(
            model.model.embed_tokens.weight.grad[token_ids.source_entity]
            .abs()
            .sum()
            .item(),
            0.0,
        )

    def test_native_checkpoint_factory_requires_kg_vocabulary_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_entities"):
            OreoQwen3ForCausalLM.from_qwen3_pretrained(
                "unused",
                oreo_config_overrides={},
                qwen_config=Qwen3Config(),
                local_files_only=True,
            )

    def test_native_qwen_control_can_register_the_same_special_tokens(self) -> None:
        tokenizer = make_tokenizer()
        native = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                tie_word_embeddings=True,
            )
        )
        token_ids = register_oreo_tokens_and_resize(native, tokenizer)
        self.assertEqual(native.model.embed_tokens.num_embeddings, len(tokenizer))
        self.assertEqual(
            native.config.oreo_source_entity_token_id,
            token_ids.source_entity,
        )


if __name__ == "__main__":
    unittest.main()
