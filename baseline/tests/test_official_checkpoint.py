from __future__ import annotations

import os
import unittest

import torch

from oreo_qwen import OreoQwen3ForCausalLM


@unittest.skipUnless(
    os.environ.get("OREO_RUN_OFFICIAL_SMOKE") == "1",
    "Set OREO_RUN_OFFICIAL_SMOKE=1 to download and test Qwen3-4B.",
)
class OfficialQwenCheckpointSmokeTest(unittest.TestCase):
    def test_loads_qwen3_4b_weights_and_runs_without_graph(self) -> None:
        model = OreoQwen3ForCausalLM.from_qwen3_pretrained(
            "Qwen/Qwen3-4B",
            oreo_config_overrides={
                "num_relations": 4,
                "knowledge_embedding_dim": 8,
                "adapter_bottleneck_dim": 8,
            },
            dtype=torch.bfloat16,
        ).eval()
        with torch.no_grad():
            output = model(input_ids=torch.tensor([[1, 2]]), use_cache=False)
        self.assertEqual(output.logits.shape[:2], (1, 2))
        self.assertEqual(model.config.kil_layer_boundaries, [34])


if __name__ == "__main__":
    unittest.main()
