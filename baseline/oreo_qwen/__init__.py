"""Public interfaces for the Oreo-Qwen baseline."""

from .configuration import OreoQwen3Config
from .contracts import ContractError, KGContractSpec, KGInputBatch, OreoQwenBatch
from .modeling import (
    OreoBaseModelOutputWithPast,
    OreoCausalLMOutputWithPast,
    OreoQwen3ForCausalLM,
    OreoQwen3Model,
    register_oreo_tokens_and_resize,
)
from .interaction import (
    CausalTentBroadcastLayer,
    GroupedQueryCrossAttention,
    MarkerContextualizationLayer,
)
from .embedding import (
    FrozenKnowledgeEmbedder,
    PersistentEmbeddingCache,
    format_entity_text,
    format_relation_text,
)
from .data import (
    CollationError,
    DatasetError,
    KnowledgeEmbeddingRequest,
    OreoSemanticCollator,
    ReconstructionDatasetItem,
    SemanticKGInputBatch,
    SemanticOreoBatch,
    Wikidata5MReconstructionDataset,
)
from .reasoner import (
    BottleneckMLP,
    GlobalInverseDegreeWalker,
    KGReasoner,
    KGReasonerConfig,
    KGReasonerState,
    KGStepOutput,
    KnowledgeFusion,
    RelationRouter,
    ValueProjection,
)
from .tokens import (
    OREO_SPECIAL_TOKENS,
    REL_TOKEN,
    TARGET_ENTITY_TOKEN,
    OreoSpecialTokenIds,
    register_oreo_special_tokens,
)

__all__ = [
    "ContractError",
    "OreoQwen3Config",
    "KGContractSpec",
    "KGInputBatch",
    "OreoQwenBatch",
    "BottleneckMLP",
    "GlobalInverseDegreeWalker",
    "KGReasoner",
    "KGReasonerConfig",
    "KGReasonerState",
    "KGStepOutput",
    "KnowledgeFusion",
    "RelationRouter",
    "ValueProjection",
    "OreoBaseModelOutputWithPast",
    "OreoCausalLMOutputWithPast",
    "OreoQwen3ForCausalLM",
    "OreoQwen3Model",
    "CausalTentBroadcastLayer",
    "GroupedQueryCrossAttention",
    "MarkerContextualizationLayer",
    "FrozenKnowledgeEmbedder",
    "PersistentEmbeddingCache",
    "CollationError",
    "DatasetError",
    "KnowledgeEmbeddingRequest",
    "OreoSemanticCollator",
    "ReconstructionDatasetItem",
    "SemanticKGInputBatch",
    "SemanticOreoBatch",
    "Wikidata5MReconstructionDataset",
    "format_entity_text",
    "format_relation_text",
    "register_oreo_tokens_and_resize",
    "OREO_SPECIAL_TOKENS",
    "REL_TOKEN",
    "TARGET_ENTITY_TOKEN",
    "OreoSpecialTokenIds",
    "register_oreo_special_tokens",
]
