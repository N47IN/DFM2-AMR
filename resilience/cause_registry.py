#!/usr/bin/env python3
"""
CauseRegistry
==============

Embeddings are the ground truth identity for each cause. Textual labels from
VLM/narration are just aliases that map back to a canonical vector ID.  When a
new embedding is registered we check cosine similarity against existing entries;
if the score exceeds the configured threshold the metadata is merged, otherwise
a brand new entry is created.
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def _hash_vec(norm_vec: np.ndarray) -> str:
    return hashlib.sha1(norm_vec.tobytes()).hexdigest()


@dataclass
class GPParams:
    lxy: Optional[float] = None
    lz: Optional[float] = None
    A: Optional[float] = None
    b: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    timestamp: Optional[float] = None
    buffer_id: Optional[str] = None


@dataclass
class CauseEntry:
    """
    Single cause entry indexed by embedding (vec_id).
    
    Contains all metadata for a cause:
    - names: List of textual aliases (e.g., ["loose cable", "cable dangling"])
    - color_rgb: Visualization color [R, G, B]
    - gp_params: GP fitting parameters (lxy, lz, A, b, metrics, etc.)
    - enhanced_embedding: Refined embedding if computed
    - stats: Detection counts, scores
    - metadata: Arbitrary key-value data
    - buffers: Associated buffer IDs
    - thresholds: Similarity/detection thresholds
    """
    vec_id: str  # Embedding hash (primary key)
    embedding: np.ndarray  # Normalized embedding vector
    names: List[str] = field(default_factory=list)  # Textual aliases
    source: str = "unknown"
    type: str = "dynamic"
    color_rgb: List[int] = field(default_factory=lambda: [255, 255, 255])
    thresholds: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=lambda: {"detections": 0, "last_score": 0.0})
    buffers: Dict[str, Any] = field(default_factory=lambda: {"all": []})
    enhanced_embedding: Optional[np.ndarray] = None
    gp_params: Optional[GPParams] = None
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["embedding"] = self.embedding.tolist()
        if self.enhanced_embedding is not None:
            data["enhanced_embedding"] = self.enhanced_embedding.tolist()
        if self.gp_params is not None:
            data["gp_params"] = asdict(self.gp_params)
        return data


class CauseRegistry:
    """
    Embedding-indexed registry for cause objects.
    
    Structure:
    - Entries are indexed by embedding hash (vec_id) in _entries dict
    - Each CauseEntry contains all metadata:
      * names: List of textual aliases (e.g., ["loose cable", "cable dangling"])
      * color_rgb: Visualization color
      * gp_params: GP fitting parameters
      * enhanced_embedding: Refined embedding if available
      * stats, metadata, buffers, thresholds: Additional properties
    
    When a new embedding is registered, cosine similarity is checked against
    existing entries. If similarity >= threshold, metadata is merged; otherwise
    a new entry is created.
    """
    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self.similarity_threshold = similarity_threshold
        # Primary storage: entries indexed by embedding hash (vec_id)
        self._entries: Dict[str, CauseEntry] = {}
        # Secondary index: name -> vec_id for quick name lookups
        self._name_to_vec: Dict[str, str] = {}
        # Vector matrix for fast cosine similarity search
        self._vector_matrix: Optional[np.ndarray] = None  # shape (N, D)
        self._vec_ids: List[str] = []

    # ---------------------------------------------------------------- embeddings
    def upsert_cause(
        self,
        name: str,
        embedding: np.ndarray,
        *,
        source: str = "unknown",
        type_: str = "dynamic",
        color_rgb: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CauseEntry:
        """Create or merge a cause using its embedding."""
        if embedding is None:
            raise ValueError("Embedding is required to register a cause")

        norm = _normalize(embedding)
        match_id, score = self._find_best_match(norm)

        if match_id is not None and score is not None and score >= self.similarity_threshold:
            entry = self._entries[match_id]
        else:
            vec_id = _hash_vec(norm)
            while vec_id in self._entries:
                vec_id = f"{vec_id}_dup"
            entry = CauseEntry(
                vec_id=vec_id,
                embedding=norm,
                source=source,
                type=type_,
                color_rgb=list(map(int, color_rgb or [255, 255, 255])),
            )
            self._entries[vec_id] = entry
            self._append_to_index(norm, vec_id)

        if name:
            self._register_name(entry, name)
        if metadata:
            entry.metadata.update(metadata)
        entry.source = entry.source or source
        entry.type = entry.type or type_
        entry.color_rgb = list(map(int, color_rgb or entry.color_rgb))
        entry.updated_ts = time.time()
        return entry

    def set_enhanced_embedding(
        self,
        name: str,
        embedding: np.ndarray,
        *,
        buffer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        entry = self._get_entry_by_name(name)
        if entry is None:
            return False
        entry.enhanced_embedding = _normalize(embedding)
        if buffer_id:
            self._append_buffer(entry, buffer_id)
        if metadata:
            entry.metadata.update(metadata)
        entry.updated_ts = time.time()
        return True

    # -------------------------------------------------------------------- stats
    def record_detection(self, name: str, score: float) -> bool:
        entry = self._get_entry_by_name(name)
        if entry is None:
            return False
        entry.stats["detections"] = int(entry.stats.get("detections", 0)) + 1
        entry.stats["last_score"] = float(score)
        entry.updated_ts = time.time()
        return True

    def set_threshold(self, name: str, key: str, value: float) -> bool:
        entry = self._get_entry_by_name(name)
        if entry is None:
            return False
        entry.thresholds[key] = float(value)
        entry.updated_ts = time.time()
        return True

    def set_metadata(self, name: str, data: Dict[str, Any]) -> bool:
        entry = self._get_entry_by_name(name)
        if entry is None:
            return False
        entry.metadata.update(data)
        entry.updated_ts = time.time()
        return True

    def set_gp_params(self, name: str, params: GPParams) -> bool:
        entry = self._get_entry_by_name(name)
        if entry is None:
            return False
        entry.gp_params = params
        if params.buffer_id:
            self._append_buffer(entry, params.buffer_id)
        entry.updated_ts = time.time()
        return True

    def has_name(self, name: str) -> bool:
        return self._name_key(name) in self._name_to_vec

    def get_entry_by_name(self, name: str) -> Optional[CauseEntry]:
        """Get entry by textual alias (name)."""
        return self._get_entry_by_name(name)

    def get_entry_by_embedding(self, embedding: np.ndarray) -> Optional[CauseEntry]:
        """Get entry by embedding (finds best match via cosine similarity)."""
        norm = _normalize(embedding)
        vec_id, score = self._find_best_match(norm)
        if vec_id is not None and score is not None and score >= self.similarity_threshold:
            return self._entries.get(vec_id)
        return None

    def get_entry_by_vec_id(self, vec_id: str) -> Optional[CauseEntry]:
        """Get entry directly by embedding hash (vec_id)."""
        return self._entries.get(vec_id)

    def get_entry(self, name: str) -> Optional[CauseEntry]:
        """Legacy alias - use get_entry_by_name for clarity."""
        return self._get_entry_by_name(name)

    def get_all(self) -> List[CauseEntry]:
        """Get all entries (indexed by embedding/vec_id)."""
        return list(self._entries.values())

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> Dict[str, Any]:
        return {
            "entries": {vec_id: entry.to_dict() for vec_id, entry in self._entries.items()},
            "name_to_vec": dict(self._name_to_vec),
            "vector_ids": list(self._vec_ids),
            "vector_matrix": self._vector_matrix.tolist() if self._vector_matrix is not None else None,
            "similarity_threshold": self.similarity_threshold,
        }

    def restore(self, data: Dict[str, Any]) -> None:
        self._entries.clear()
        self._name_to_vec.clear()
        self.similarity_threshold = float(data.get("similarity_threshold", self.similarity_threshold))

        vector_matrix = data.get("vector_matrix")
        self._vector_matrix = np.asarray(vector_matrix, dtype=np.float32) if vector_matrix is not None else None
        self._vec_ids = list(data.get("vector_ids", []))

        entries = data.get("entries", {})
        for vec_id, payload in entries.items():
            entry = CauseEntry(
                vec_id=vec_id,
                embedding=np.asarray(payload.get("embedding", []), dtype=np.float32),
                names=list(payload.get("names", [])),
                source=payload.get("source", "unknown"),
                type=payload.get("type", "dynamic"),
                color_rgb=list(payload.get("color_rgb", [255, 255, 255])),
                thresholds=dict(payload.get("thresholds", {})),
                metadata=dict(payload.get("metadata", {})),
                stats=dict(payload.get("stats", {})),
                buffers=dict(payload.get("buffers", {"all": []})),
                created_ts=float(payload.get("created_ts", time.time())),
                updated_ts=float(payload.get("updated_ts", time.time())),
            )
            enhanced = payload.get("enhanced_embedding")
            if enhanced is not None:
                entry.enhanced_embedding = np.asarray(enhanced, dtype=np.float32)
            gp_payload = payload.get("gp_params")
            if gp_payload:
                entry.gp_params = GPParams(**gp_payload)
            self._entries[vec_id] = entry
            for name in entry.names:
                self._name_to_vec[self._name_key(name)] = vec_id

        # restore alias map if provided explicitly
        explicit_alias = data.get("name_to_vec", {})
        for name, vec_id in explicit_alias.items():
            if vec_id in self._entries:
                self._name_to_vec[name] = vec_id

    # ----------------------------------------------------------------- internal
    def _find_best_match(self, norm_vec: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        if self._vector_matrix is None or self._vector_matrix.size == 0:
            return None, None
        sims = self._vector_matrix @ norm_vec
        idx = int(np.argmax(sims))
        best_score = float(sims[idx])
        vec_id = self._vec_ids[idx]
        return vec_id, best_score

    def _append_to_index(self, norm_vec: np.ndarray, vec_id: str) -> None:
        if self._vector_matrix is None:
            self._vector_matrix = norm_vec.reshape(1, -1)
        else:
            self._vector_matrix = np.vstack([self._vector_matrix, norm_vec.reshape(1, -1)])
        self._vec_ids.append(vec_id)

    def _register_name(self, entry: CauseEntry, name: str) -> None:
        key = self._name_key(name)
        if key in self._name_to_vec and self._name_to_vec[key] == entry.vec_id:
            return
        if name not in entry.names:
            entry.names.append(name)
        self._name_to_vec[key] = entry.vec_id

    def _get_entry_by_name(self, name: str) -> Optional[CauseEntry]:
        key = self._name_key(name)
        vec_id = self._name_to_vec.get(key)
        if vec_id is None:
            return None
        return self._entries.get(vec_id)

    @staticmethod
    def _name_key(name: str) -> str:
        return (name or "").strip().lower()

    @staticmethod
    def _append_buffer(entry: CauseEntry, buffer_id: str) -> None:
        if not buffer_id:
            return
        buffers = entry.buffers.setdefault("all", [])
        if buffer_id not in buffers:
            buffers.append(buffer_id)
        entry.buffers["last"] = buffer_id