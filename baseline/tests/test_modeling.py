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
    num_hidden_layers: int = 2,
    kil_layer_boundaries: tuple[int, ...] = (1,),
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
        num_relations=2,
        kil_layer_boundaries=kil_layer_boundaries,
        knowledge_embedding_dim=4,
        adapter_bottleneck_dim=8,
        output_kg_diagnostics=output_kg_diagnostics,
    )


def make_kg(*, num_steps: int = 1, supervision: bool = True) -> KGInputBatch:
    optional: dict[str, torch.Tensor] = {
        "kg_initial_node_indices": torch.tensor([0, 3]),
    }
    if supervision:
        relation_targets = torch.zeros(num_steps, 2, 2)
        for step in range(num_steps):
            relation_targets[step, :, step % 2] = 1.0
        optional.update(
            kg_relation_targets=relation_targets,
            kg_relation_target_mask=torch.ones(num_steps, 2, dtype=torch.bool),
        )
    return KGInputBatch(
        kg_mention_batch=torch.tensor([0, 1]),
        kg_rel_positions=torch.tensor([2, 2]),
        kg_tgt_positions=torch.tensor([3, 3]),
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
    def test_qwen3_4b_defaults_to_one_late_kil(self) -> None:
        config = make_config(num_hidden_layers=36, kil_layer_boundaries=(34,))
        self.assertEqual(config.kil_layer_boundaries, [34])
        self.assertEqual(config.num_kil_steps, 1)
        self.assertEqual(config.reasoner_config.num_kil_steps, 1)

        automatic = OreoQwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=36,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
        self.assertEqual(automatic.kil_layer_boundaries, [34])

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
            num_relations=5,
            kil_layer_boundaries=[1],
        )
        self.assertEqual(config.vocab_size, native.vocab_size)
        self.assertEqual(config.hidden_size, native.hidden_size)
        self.assertEqual(config.num_hidden_layers, native.num_hidden_layers)
        self.assertEqual(config.knowledge_embedding_dim, 384)


class OreoQwen3ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.config = make_config()
        self.model = OreoQwen3ForCausalLM(self.config)
        self.input_ids, self.attention_mask, self.labels = make_language_batch()
        self.kg = make_kg()

    def test_single_kil_runs_at_configured_boundary(self) -> None:
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
        for layer in self.model.model.pre_kil_layers:
            hooks.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, _output: events.append("pre-kil")
                )
            )
        for layer in self.model.model.post_kil_layers:
            hooks.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, _output: events.append("post-kil")
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
            [
                "layer-1",
                "pre-kil",
                "kil",
                "post-kil",
                "layer-2",
            ],
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
        self.assertIsNotNone(output.relation_loss)
        torch.testing.assert_close(
            output.loss,
            output.lm_loss + output.relation_loss,
        )
        self.assertEqual(len(output.entity_states), 2)  # initial, one-hop
        self.assertEqual(len(output.relation_probs), 1)
        self.assertEqual(output.entity_states[-1].shape, (2, 6))
        self.assertEqual(output.relation_probs[-1].shape, (2, 2))

    def test_auxiliary_weights_control_total_loss(self) -> None:
        self.model.config.relation_loss_weight = 2.0
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            use_cache=False,
            **self.kg.as_model_kwargs(),
        )
        expected = (
            output.lm_loss + 2.0 * output.relation_loss
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
            self.model.model.reasoner.steps[0]
            .value_projection.projection.output.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_gradients_cross_frozen_qwen_and_reach_oreo_modules(self) -> None:
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            use_cache=False,
            **self.kg.as_model_kwargs(),
        )
        output.loss.backward()

        self.assertIsNone(
            self.model.model.layers[0].self_attn.q_proj.weight.grad
        )
        self.assertFalse(
            self.model.model.layers[0].self_attn.q_proj.weight.requires_grad
        )
        reasoner = self.model.model.reasoner
        for step in reasoner.steps:
            self.assertGreater(
                step.relation_router.query_projection.input.weight.grad.abs().sum().item(),
                0.0,
            )
            self.assertGreater(
                step.value_projection.projection.input.weight.grad.abs().sum().item(),
                0.0,
            )

        # The new attention branches are zero-gated for checkpoint stability.
        # Their gates must still receive a first-step gradient so the branches
        # can open and their projections can train on subsequent steps.
        for layer in (
            *self.model.model.pre_kil_layers,
            *self.model.model.post_kil_layers,
        ):
            self.assertIsNotNone(layer.gate.grad)
            self.assertGreater(layer.gate.grad.abs().item(), 0.0)

        self.assertIsNone(self.model.model.embed_tokens.weight.grad)
        self.assertIsNone(self.model.lm_head.weight.grad)

    def test_only_oreo_parameters_are_optimizer_visible(self) -> None:
        trainable = {
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        allowed = (
            "model.reasoner.",
            "model.pre_kil_layers.",
            "model.post_kil_layers.",
        )
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name == "model.oreo_token_deltas" or name.startswith(allowed)
                for name in trainable
            )
        )
        self.assertEqual(
            {name for name, _ in self.model.oreo_named_parameters()},
            trainable,
        )
        self.assertEqual(
            {id(parameter) for parameter in self.model.oreo_parameters()},
            {
                id(parameter)
                for parameter in self.model.parameters()
                if parameter.requires_grad
            },
        )
        self.assertFalse(self.model.model.embed_tokens.weight.requires_grad)
        self.assertFalse(self.model.model.norm.weight.requires_grad)
        self.assertFalse(self.model.lm_head.weight.requires_grad)

    def test_gradient_checkpointing_preserves_frozen_backbone_flow(self) -> None:
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            use_cache=False,
            **self.kg.as_model_kwargs(),
        )
        output.loss.backward()
        gradient = (
            self.model.model.reasoner.steps[0]
            .relation_router.query_projection.output.weight.grad
        )
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertIsNone(
            self.model.model.layers[-1].self_attn.q_proj.weight.grad
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
            num_relations=2,
            knowledge_embedding_dim=4,
            adapter_bottleneck_dim=8,
            kil_layer_boundaries=[1],
        )
        native = Qwen3ForCausalLM(native_config).eval()
        oreo = OreoQwen3ForCausalLM(oreo_config).eval()
        incompatible = oreo.load_state_dict(native.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        new_module_prefixes = (
            "model.oreo_token_deltas",
            "model.reasoner.",
            "model.pre_kil_layers.",
            "model.post_kil_layers.",
        )
        self.assertTrue(
            all(key.startswith(new_module_prefixes) for key in incompatible.missing_keys)
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
                    "num_relations": 2,
                    "knowledge_embedding_dim": 4,
                    "adapter_bottleneck_dim": 8,
                    "kil_layer_boundaries": [1],
                },
            ).eval()

        with torch.no_grad():
            native_logits = native(self.input_ids, use_cache=False).logits
            oreo_logits = oreo(self.input_ids, use_cache=False).logits
        torch.testing.assert_close(oreo_logits, native_logits, atol=0, rtol=0)
        self.assertEqual(
            oreo.model.reasoner.steps[0].value_projection.projection.input.in_features,
            4,
        )

    def test_partial_flat_graph_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "partial KG batch"):
            self.model(
                input_ids=self.input_ids,
                kg_node_embeddings=self.kg.kg_node_embeddings,
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
        self.assertEqual(restored.config.kil_layer_boundaries, [1])
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
                num_relations=2,
                kil_layer_boundaries=[1],
                knowledge_embedding_dim=4,
                adapter_bottleneck_dim=8,
                tie_word_embeddings=True,
            )
        )
        token_ids = register_oreo_tokens_and_resize(model, tokenizer)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        self.assertEqual(model.model.embed_tokens.num_embeddings, len(tokenizer))
        self.assertEqual(model.lm_head.out_features, len(tokenizer))
        self.assertEqual(model.config.vocab_size, len(tokenizer))
        self.assertEqual(model.config.oreo_special_token_ids, token_ids)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

        input_ids = torch.tensor(
            [[
                2,
                token_ids.relation,
                token_ids.target_entity,
                4,
                5,
                3,
            ]]
        )
        labels = input_ids.clone()
        labels[:, :3] = -100
        kg = KGInputBatch(
            kg_mention_batch=torch.tensor([0]),
            kg_rel_positions=torch.tensor([1]),
            kg_tgt_positions=torch.tensor([2]),
            kg_node_embeddings=torch.randn(2, 4),
            kg_relation_embeddings=torch.randn(2, 4),
            kg_node_batch=torch.tensor([0, 0]),
            kg_edge_src=torch.tensor([0]),
            kg_edge_dst=torch.tensor([1]),
            kg_edge_type=torch.tensor([0]),
            kg_initial_node_indices=torch.tensor([0]),
            kg_relation_targets=torch.tensor([[[1.0, 0.0]]]),
            kg_relation_target_mask=torch.tensor([[True]]),
        )
        frozen_before = (
            model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        )
        delta_before = model.model.oreo_token_deltas.detach().clone()
        optimizer = torch.optim.SGD(model.oreo_parameters(), lr=0.1)
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            use_cache=False,
            **kg.as_model_kwargs(),
        )
        output.loss.backward()
        self.assertGreater(
            model.model.oreo_token_deltas.grad[0].abs().sum().item(),
            0.0,
        )
        self.assertIsNone(model.model.embed_tokens.weight.grad)
        self.assertIsNone(model.lm_head.weight.grad)
        self.assertFalse(model.model.embed_tokens.weight.requires_grad)
        self.assertFalse(model.lm_head.weight.requires_grad)
        optimizer.step()
        torch.testing.assert_close(
            model.model.layers[0].self_attn.q_proj.weight,
            frozen_before,
            atol=0,
            rtol=0,
        )
        self.assertFalse(
            torch.equal(model.model.oreo_token_deltas.detach(), delta_before)
        )

    def test_native_checkpoint_factory_requires_relation_vocabulary_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_relations"):
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
        self.assertEqual(native.config.oreo_relation_token_id, token_ids.relation)


if __name__ == "__main__":
    unittest.main()
