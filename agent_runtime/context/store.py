from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from agent_runtime.model.embedding_client import EmbeddingClient

# Lazy FAISS import — falls back to linear scan if not installed
_faiss = None


def _get_faiss() -> Any:
    global _faiss
    if _faiss is None:
        try:
            import faiss  # type: ignore
            _faiss = faiss
        except ImportError:
            _faiss = False
    return _faiss


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def _normalize_for_dedup(text: str) -> str:
    """Normalize so that cosmetic differences (whitespace, line-endings, Unicode)
    map to the same hash.  Does NOT lowercase — code / proper nouns are case-sensitive."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_content(content: str, max_chunk_chars: int = 1500) -> list[str]:
    """Split agent output into reusable chunks at paragraph boundaries."""
    if len(content) <= max_chunk_chars:
        return [content]

    raw = re.split(r"\n\n+", content)
    chunks: list[str] = []
    buf = ""

    for para in raw:
        para = para.strip()
        if not para:
            continue
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if len(candidate) <= max_chunk_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if len(para) > max_chunk_chars:
                lines = para.split("\n")
                sub = ""
                for line in lines:
                    trial = (sub + "\n" + line).strip() if sub else line.strip()
                    if len(trial) <= max_chunk_chars:
                        sub = trial
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = line.strip()
                buf = sub if sub else ""
            else:
                buf = para

    if buf:
        chunks.append(buf)

    return chunks or [content]


@dataclass(slots=True)
class ContextSegment:
    segment_id: str
    kind: str
    content: str
    owner: str
    visibility: str = "shared"
    allowed_agents: list[str] = field(default_factory=list)
    content_hash: str = ""
    normalized_hash: str = ""
    compressed: bool = False
    created_at: float = field(default_factory=time.time)

    def as_record(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "owner": self.owner,
            "visibility": self.visibility,
            "allowed_agents": self.allowed_agents,
            "hash": self.content_hash,
            "normalized_hash": self.normalized_hash,
            "compressed": self.compressed,
            "tokens": estimate_tokens(self.content),
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ContextSnapshot:
    context_id: str
    segment_ids: list[str]
    parent_id: str | None = None
    owner: str = "runtime"
    created_at: float = field(default_factory=time.time)


class ContextStore:
    def __init__(
        self,
        compression_threshold_tokens: int = 5000,
        embedding_client: EmbeddingClient | None = None,
        semantic_threshold: float = 0.90,
        disable_reuse: bool = False,
    ) -> None:
        self.compression_threshold_tokens = compression_threshold_tokens
        self.embedding_client = embedding_client
        self.semantic_threshold = semantic_threshold
        self.disable_reuse = disable_reuse

        self.segments: dict[str, ContextSegment] = {}
        self.snapshots: dict[str, ContextSnapshot] = {}
        self.hash_index: dict[str, str] = {}             # exact SHA256 → seg_id
        self.normalized_index: dict[str, str] = {}       # normalized SHA256 → seg_id
        self.ref_counts: dict[str, int] = {}
        self._segment_seq = 0
        self._snapshot_seq = 0
        self.hits = 0           # exact byte-level dedup hits
        self.soft_hits = 0      # normalized-level dedup hits
        self.semantic_hits = 0  # embedding-level dedup hits
        self.misses = 0
        self.compressions = 0

        # ── semantic dedup state ──
        if self.embedding_client is not None:
            from agent_runtime.model.embedding_client import cosine_similarity
            self._cosine = cosine_similarity
        else:
            self._cosine = lambda _a, _b: 0.0  # noqa: E731

        # FAISS index (built lazily when embedding count exceeds threshold)
        self._faiss_index: Any = None
        self._faiss_seg_ids: list[str] = []
        self._faiss_dirty: bool = False
        self._faiss_threshold: int = 200

    # ── segment creation ────────────────────────────────────────────────

    def create_segment(
        self,
        kind: str,
        content: str,
        owner: str,
        visibility: str = "shared",
        allowed_agents: list[str] | None = None,
    ) -> str:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        norm_text = _normalize_for_dedup(content)
        norm_digest = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

        if self.disable_reuse:
            return self._store_new_segment(
                kind, content, owner, visibility, allowed_agents or [], digest, norm_digest,
            )

        # Level 1 ─ exact match
        if digest in self.hash_index:
            self.hits += 1
            seg_id = self.hash_index[digest]
            self.ref_counts[seg_id] = self.ref_counts.get(seg_id, 0) + 1
            return seg_id

        # Level 2 ─ normalized (soft) match
        if norm_digest in self.normalized_index and norm_digest != digest:
            self.soft_hits += 1

        # Level 3 ─ semantic (embedding) match
        if self.embedding_client is not None and self._cosine is not None:
            matched_id = self._semantic_lookup(content)
            if matched_id is not None:
                self.semantic_hits += 1
                self.ref_counts[matched_id] = self.ref_counts.get(matched_id, 0) + 1
                return matched_id

        return self._store_new_segment(
            kind, content, owner, visibility, allowed_agents or [], digest, norm_digest,
        )

    def create_segments_batch(
        self,
        kind_prefix: str,
        content: str,
        owner: str,
        visibility: str = "shared",
        allowed_agents: list[str] | None = None,
    ) -> list[str]:
        """Split content into reusable chunks and create one segment per chunk.

        When an embedding client is configured, all chunks that survive the
        first two dedup levels are embedded in a single batch call to reduce
        round-trips."""
        chunks = _chunk_content(content)
        allowed = allowed_agents or []
        ids: list[str] = []

        # Bypass all dedup when disabled
        if self.disable_reuse:
            for i, chunk in enumerate(chunks):
                kind = f"{kind_prefix}:c{i}" if len(chunks) > 1 else kind_prefix
                digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                norm_text = _normalize_for_dedup(chunk)
                norm_digest = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
                ids.append(self._store_new_segment(kind, chunk, owner, visibility, allowed, digest, norm_digest))
            return ids

        # First pass: exact + normalized dedup only (fast, no API calls)
        pending: list[tuple[int, str, str, str, str]] = []  # i, kind, chunk, digest, norm_digest
        for i, chunk in enumerate(chunks):
            kind = f"{kind_prefix}:c{i}" if len(chunks) > 1 else kind_prefix
            digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()

            if digest in self.hash_index:
                self.hits += 1
                seg_id = self.hash_index[digest]
                self.ref_counts[seg_id] = self.ref_counts.get(seg_id, 0) + 1
                ids.append(seg_id)
                continue

            norm_text = _normalize_for_dedup(chunk)
            norm_digest = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
            if norm_digest in self.normalized_index and norm_digest != digest:
                self.soft_hits += 1

            if self.embedding_client is not None and self._cosine is not None:
                pending.append((i, kind, chunk, digest, norm_digest))
            else:
                seg_id = self._store_new_segment(kind, chunk, owner, visibility, allowed, digest, norm_digest)
                ids.append(seg_id)

        if not pending:
            return ids

        # Second pass: batch-embed all pending chunks, then semantic lookup
        pending_texts = [item[2] for item in pending]
        try:
            embeddings = self.embedding_client.embed_batch(pending_texts)  # type: ignore[union-attr]
        except Exception:
            # Embedding service unavailable → store all as new segments
            embeddings = [None] * len(pending)

        for (i, kind, chunk, digest, norm_digest), emb in zip(pending, embeddings):
            if emb is not None:
                matched_id = self._semantic_lookup_with_embedding(emb)
                if matched_id is not None:
                    self.semantic_hits += 1
                    self.ref_counts[matched_id] = self.ref_counts.get(matched_id, 0) + 1
                    ids.insert(i, matched_id)
                    continue

            seg_id = self._store_new_segment(kind, chunk, owner, visibility, allowed, digest, norm_digest)
            # Store the pre-computed embedding
            if emb is not None and self.embedding_client is not None:
                self._store_embedding(seg_id, emb)
            ids.insert(i, seg_id)

        return ids

    def _store_new_segment(
        self,
        kind: str,
        content: str,
        owner: str,
        visibility: str,
        allowed_agents: list[str],
        digest: str,
        norm_digest: str,
    ) -> str:
        self.misses += 1
        self._segment_seq += 1
        seg_id = f"seg-{self._segment_seq:06d}"
        segment = ContextSegment(
            segment_id=seg_id,
            kind=kind,
            content=content,
            owner=owner,
            visibility=visibility,
            allowed_agents=allowed_agents,
            content_hash=digest,
            normalized_hash=norm_digest,
        )
        self.segments[seg_id] = segment
        self.hash_index[digest] = seg_id
        self.normalized_index[norm_digest] = seg_id
        self.ref_counts[seg_id] = 1
        return seg_id

    # ── semantic dedup ──────────────────────────────────────────────────

    def _semantic_lookup(self, text: str) -> str | None:
        """Embed *text* and search existing embeddings for the nearest match."""
        if not self.embedding_client:
            return None
        emb = self.embedding_client.embed_single(text)
        return self._semantic_lookup_with_embedding(emb)

    def _semantic_lookup_with_embedding(self, emb: list[float]) -> str | None:
        if not emb or not hasattr(self, '_cosine'):
            return None

        faiss = _get_faiss()
        vec = np.array(emb, dtype=np.float32)

        # ── FAISS path ──
        if faiss and self._faiss_index is not None:
            if self._faiss_dirty:
                self._rebuild_faiss_index(faiss)
            if len(self._faiss_seg_ids) > 0:
                distances, indices = self._faiss_index.search(vec.reshape(1, -1), 1)  # type: ignore[union-attr]
                if indices[0][0] >= 0:
                    best_score = float(distances[0][0])
                    best_idx = int(indices[0][0])
                    if 0 <= best_idx < len(self._faiss_seg_ids) and best_score >= self.semantic_threshold:
                        return self._faiss_seg_ids[best_idx]
            return None

        # ── Linear fallback ──
        if not hasattr(self, '_embeddings'):
            return None
        best_id, best_score = None, 0.0
        for seg_id, existing in self._embeddings.items():
            score = self._cosine(emb, existing)  # type: ignore[misc]
            if score > best_score:
                best_score, best_id = score, seg_id
        if best_score >= self.semantic_threshold and best_id is not None:
            return best_id
        return None

    def _store_embedding(self, seg_id: str, emb: list[float]) -> None:
        if not hasattr(self, '_embeddings'):
            self._embeddings: dict[str, list[float]] = {}
        self._embeddings[seg_id] = emb
        self._faiss_dirty = True

        faiss = _get_faiss()
        if faiss and len(self._embeddings) >= self._faiss_threshold:
            self._rebuild_faiss_index(faiss)

    def _rebuild_faiss_index(self, faiss_module: Any) -> None:
        if not hasattr(self, '_embeddings') or not self._embeddings:
            return
        seg_ids = list(self._embeddings.keys())
        vectors = np.array([self._embeddings[sid] for sid in seg_ids], dtype=np.float32)
        # Inner product is equivalent to cosine for normalized vectors
        dim = vectors.shape[1]
        index = faiss_module.IndexFlatIP(dim)
        index.add(vectors)
        self._faiss_index = index
        self._faiss_seg_ids = seg_ids
        self._faiss_dirty = False

    # ── snapshots ───────────────────────────────────────────────────────

    def create_snapshot(
        self,
        segment_ids: list[str],
        owner: str = "runtime",
        parent_id: str | None = None,
    ) -> str:
        self._snapshot_seq += 1
        context_id = f"ctx-{self._snapshot_seq:06d}"
        self.snapshots[context_id] = ContextSnapshot(
            context_id=context_id,
            segment_ids=list(segment_ids),
            parent_id=parent_id,
            owner=owner,
        )
        for segment_id in segment_ids:
            self.ref_counts[segment_id] = self.ref_counts.get(segment_id, 0) + 1
        return context_id

    def fork(
        self, parent_id: str, extra_segment_ids: list[str] | None = None, owner: str = "runtime",
    ) -> str:
        parent = self.snapshots[parent_id]
        return self.create_snapshot(
            parent.segment_ids + (extra_segment_ids or []), owner=owner, parent_id=parent_id,
        )

    # ── visibility & materialisation ────────────────────────────────────

    def visible_segments(self, context_id: str, agent_id: str) -> list[ContextSegment]:
        snapshot = self.snapshots[context_id]
        visible: list[ContextSegment] = []
        for segment_id in snapshot.segment_ids:
            segment = self.segments[segment_id]
            if segment.visibility == "public":
                visible.append(segment)
            elif segment.visibility == "shared":
                visible.append(segment)
            elif segment.owner == agent_id or agent_id in segment.allowed_agents:
                visible.append(segment)
        return visible

    def materialize(self, context_id: str, agent_id: str, token_budget: int) -> tuple[str, dict[str, Any]]:
        segments = self.visible_segments(context_id, agent_id)
        parts: list[str] = []
        total_tokens = 0
        for segment in segments:
            header = f"[{segment.kind}:{segment.segment_id}:owner={segment.owner}]"
            body = f"{header}\n{segment.content}".strip()
            tokens = estimate_tokens(body)
            if total_tokens + tokens > token_budget:
                remaining = max(0, token_budget - total_tokens)
                if remaining <= 20:
                    break
                body = self._truncate(body, remaining)
                tokens = estimate_tokens(body)
            parts.append(body)
            total_tokens += tokens
        return "\n\n".join(parts), {
            "segments": len(parts),
            "estimated_tokens": total_tokens,
            "context_id": context_id,
        }

    # ── compression ─────────────────────────────────────────────────────

    def maybe_compress(self, context_id: str, owner: str = "runtime") -> str:
        text, metrics = self.materialize(context_id, owner, token_budget=10**9)
        if metrics["estimated_tokens"] < self.compression_threshold_tokens:
            return context_id
        summary = self._extractive_summary(text, max_chars=self.compression_threshold_tokens * 2)
        raw = zlib.compress(summary.encode("utf-8"))
        compressed_text = zlib.decompress(raw).decode("utf-8")
        seg_id = self.create_segment("summary", compressed_text, owner=owner, visibility="shared")
        self.segments[seg_id].compressed = True
        self.compressions += 1
        return self.create_snapshot([seg_id], owner=owner, parent_id=context_id)

    # ── export & metrics ────────────────────────────────────────────────

    def export(self, path: Path) -> None:
        data = {
            "segments": [segment.as_record() for segment in self.segments.values()],
            "snapshots": {
                key: {
                    "context_id": value.context_id,
                    "segment_ids": value.segment_ids,
                    "parent_id": value.parent_id,
                    "owner": value.owner,
                    "created_at": value.created_at,
                }
                for key, value in self.snapshots.items()
            },
            "metrics": self.metrics(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def metrics(self) -> dict[str, Any]:
        total_refs = sum(self.ref_counts.values())
        total_dedup_attempts = self.hits + self.soft_hits + self.semantic_hits + self.misses
        all_hits = self.hits + self.soft_hits + self.semantic_hits
        return {
            "segments": len(self.segments),
            "snapshots": len(self.snapshots),
            "dedupe_hits": self.hits,
            "dedupe_soft_hits": self.soft_hits,
            "dedupe_semantic_hits": self.semantic_hits,
            "dedupe_misses": self.misses,
            "reuse_ratio": all_hits / max(1, total_dedup_attempts),
            "exact_reuse_ratio": self.hits / max(1, total_dedup_attempts),
            "soft_reuse_ratio": (self.hits + self.soft_hits) / max(1, total_dedup_attempts),
            "sharing_ratio": max(0, total_refs - len(self.segments)) / max(1, total_refs),
            "ref_counts": total_refs,
            "compressions": self.compressions,
        }

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _truncate(text: str, token_budget: int) -> str:
        char_budget = max(80, token_budget * 3)
        return text[:char_budget] + "\n[context truncated by runtime]"

    @staticmethod
    def _extractive_summary(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        return f"{head}\n\n[compressed middle by runtime]\n\n{tail}"
