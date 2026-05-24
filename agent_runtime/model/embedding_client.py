from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any


class EmbeddingClient:
    """Lightweight Ollama embedding client with an in-memory cache.

    Uses the /api/embed endpoint.  Cache is keyed by SHA-256 of the input
    text so that identical prompts are never re-embedded.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:4b",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._cache: dict[str, list[float]] = {}

    # ── public API ──────────────────────────────────────────────────

    def embed_single(self, text: str) -> list[float]:
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]
        vec = self._call_api([text])[0]
        self._cache[key] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed multiple texts.  Uncached texts are sent in one request."""
        result: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                result[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            vectors = self._call_api(uncached_texts)
            for j, vec in zip(uncached_indices, vectors):
                result[j] = vec
                key = self._cache_key(texts[j])
                self._cache[key] = vec

        return result  # type: ignore[return-value]

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding call failed: {exc}") from exc
        return raw.get("embeddings", [])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors of equal length."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
