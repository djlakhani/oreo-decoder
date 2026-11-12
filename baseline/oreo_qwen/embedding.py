"""Frozen, on-demand knowledge embedding outside the Oreo-Qwen model graph."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class PersistentEmbeddingCache:
    """SQLite-backed CPU cache for restart-safe frozen knowledge vectors."""

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str,
        output_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.output_dim = output_dim
        self.dtype = dtype
        self._connection = sqlite3.connect(self.path, timeout=60.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (namespace, cache_key)
            )
            """
        )
        self._connection.commit()

    def get(self, key: Hashable, text: str) -> Tensor | None:
        encoded_key = _stable_cache_key(key)
        row = self._connection.execute(
            "SELECT text_sha256, vector FROM embeddings "
            "WHERE namespace=? AND cache_key=?",
            (self.namespace, encoded_key),
        ).fetchone()
        if row is None:
            return None
        expected_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if row[0] != expected_text_hash:
            raise ValueError("One persistent cache key was paired with conflicting text.")
        raw = bytearray(row[1])
        expected_bytes = self.output_dim * torch.empty((), dtype=self.dtype).element_size()
        if len(raw) != expected_bytes:
            raise ValueError("Persistent embedding cache contains a malformed vector.")
        return torch.frombuffer(raw, dtype=torch.uint8).view(self.dtype).clone()

    def put(self, key: Hashable, text: str, vector: Tensor) -> None:
        self.put_many(((key, text, vector),))

    def put_many(
        self, items: Iterable[tuple[Hashable, str, Tensor]]
    ) -> None:
        rows: list[tuple[str, str, str, bytes]] = []
        for key, text, vector in items:
            value = vector.detach().to(device="cpu", dtype=self.dtype).contiguous()
            if tuple(value.shape) != (self.output_dim,):
                raise ValueError(
                    f"cached vector must have shape ({self.output_dim},), "
                    f"got {tuple(value.shape)}"
                )
            rows.append(
                (
                    self.namespace,
                    _stable_cache_key(key),
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    value.view(torch.uint8).numpy().tobytes(),
                )
            )
        if not rows:
            return
        self._connection.executemany(
            "INSERT OR REPLACE INTO embeddings "
            "(namespace, cache_key, text_sha256, vector) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._connection.commit()

    @property
    def item_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE namespace=?", (self.namespace,)
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "PersistentEmbeddingCache":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class FrozenKnowledgeEmbedder:
    """Batch-encode cache misses with a frozen Qwen3 embedding checkpoint.

    The bounded hot cache is CPU-resident and an optional SQLite cache persists
    vectors across restarts. This helper is not an ``nn.Module`` and is absent
    from Oreo-Qwen checkpoints and optimizer parameter groups. Callers pass the
    resolved tensors to ``KGInputBatch``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        output_dim: int = 384,
        device: str | torch.device = "cpu",
        model_dtype: torch.dtype | str = "auto",
        cache_dtype: torch.dtype = torch.float16,
        cache_path: str | Path | None = None,
        max_memory_items: int | None = 100_000,
        max_length: int = 512,
        **from_pretrained_kwargs: Any,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if max_memory_items is not None and max_memory_items <= 0:
            raise ValueError("max_memory_items must be positive when supplied.")
        self.model_name = model_name
        self.output_dim = output_dim
        self.device = torch.device(device)
        self.cache_dtype = cache_dtype
        self.max_length = max_length
        self.max_memory_items = max_memory_items
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            **from_pretrained_kwargs,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            **from_pretrained_kwargs,
        ).to(self.device)
        hidden_size = int(self.model.config.hidden_size)
        if output_dim > hidden_size:
            raise ValueError(
                f"output_dim {output_dim} exceeds encoder width {hidden_size}."
            )
        self.model.eval()
        self.model.requires_grad_(False)
        self._cache: OrderedDict[Hashable, tuple[str, Tensor]] = OrderedDict()
        self.persistent_cache = (
            None
            if cache_path is None
            else PersistentEmbeddingCache(
                cache_path,
                namespace=f"{model_name}|dim={output_dim}|dtype={cache_dtype}",
                output_dim=output_dim,
                dtype=cache_dtype,
            )
        )

    @property
    def cached_items(self) -> int:
        return len(self._cache)

    @property
    def persistent_cached_items(self) -> int:
        cache = getattr(self, "persistent_cache", None)
        return 0 if cache is None else cache.item_count

    @torch.inference_mode()
    def resolve(
        self,
        keys: Sequence[Hashable],
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        output_device: str | torch.device | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Resolve keys in order, encoding each unique cache miss only once."""

        if len(keys) != len(texts):
            raise ValueError("keys and texts must have the same length.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        missing: dict[Hashable, str] = {}
        for key, text in zip(keys, texts, strict=True):
            cached = self._get_cached(key, text)
            if cached is not None and cached[0] != text:
                raise ValueError("One cache key was paired with conflicting text.")
            if cached is None:
                prior = missing.setdefault(key, text)
                if prior != text:
                    raise ValueError("One cache key was paired with conflicting text.")

        missing_items = list(missing.items())
        for start in range(0, len(missing_items), batch_size):
            chunk = missing_items[start : start + batch_size]
            encoded = self._encode_texts([text for _, text in chunk])
            resolved_chunk: list[tuple[Hashable, str, Tensor]] = []
            for (key, key_text), vector in zip(chunk, encoded, strict=True):
                cached_vector = vector.to(
                    device="cpu", dtype=self.cache_dtype
                ).contiguous()
                resolved_chunk.append((key, key_text, cached_vector))
                self._remember(key, key_text, cached_vector)
            persistent = getattr(self, "persistent_cache", None)
            if persistent is not None:
                persistent.put_many(resolved_chunk)

        if not keys:
            result = torch.empty((0, self.output_dim), dtype=self.cache_dtype)
        else:
            resolved = [self._get_cached(key, text) for key, text in zip(keys, texts, strict=True)]
            if any(item is None for item in resolved):
                raise RuntimeError("embedding cache lost a value during resolution")
            result = torch.stack([item[1] for item in resolved if item is not None])
        return result.to(
            device=output_device if output_device is not None else "cpu",
            dtype=output_dtype if output_dtype is not None else self.cache_dtype,
        )

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        persistent = getattr(self, "persistent_cache", None)
        if persistent is not None:
            persistent.close()

    def _get_cached(
        self, key: Hashable, text: str
    ) -> tuple[str, Tensor] | None:
        cached = self._cache.get(key)
        if cached is not None:
            if isinstance(self._cache, OrderedDict):
                self._cache.move_to_end(key)
            return cached
        persistent = getattr(self, "persistent_cache", None)
        if persistent is None:
            return None
        vector = persistent.get(key, text)
        if vector is None:
            return None
        self._remember(key, text, vector)
        return text, vector

    def _remember(self, key: Hashable, text: str, vector: Tensor) -> None:
        self._cache[key] = (text, vector)
        if isinstance(self._cache, OrderedDict):
            self._cache.move_to_end(key)
        limit = getattr(self, "max_memory_items", None)
        if limit is not None and len(self._cache) > limit:
            if getattr(self, "persistent_cache", None) is None:
                return
            self._cache.popitem(last=False)

    def _encode_texts(self, texts: Sequence[str]) -> Tensor:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**batch)
        pooled = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        return F.normalize(pooled[:, : self.output_dim].float(), dim=-1)


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Match Qwen3-Embedding's recommended last-token pooling."""

    if bool(torch.all(attention_mask[:, -1] == 1)):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(
        last_hidden_states.shape[0], device=last_hidden_states.device
    )
    return last_hidden_states[batch_indices, sequence_lengths]


def format_entity_text(
    label: str,
    *,
    description: str | None = None,
    aliases: Sequence[str] = (),
    types: Sequence[str] = (),
) -> str:
    fields = [f"Entity: {label.strip()}"]
    if description:
        fields.append(f"Description: {description.strip()}")
    if types:
        fields.append("Types: " + ", ".join(item.strip() for item in types if item.strip()))
    if aliases:
        fields.append(
            "Aliases: " + ", ".join(item.strip() for item in aliases if item.strip())
        )
    return "\n".join(fields)


def format_relation_text(label: str, *, description: str | None = None) -> str:
    fields = [f"Relation: {label.strip()}"]
    if description:
        fields.append(f"Description: {description.strip()}")
    return "\n".join(fields)


def _stable_cache_key(key: Hashable) -> str:
    serialized = json.dumps(key, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "FrozenKnowledgeEmbedder",
    "PersistentEmbeddingCache",
    "format_entity_text",
    "format_relation_text",
]
