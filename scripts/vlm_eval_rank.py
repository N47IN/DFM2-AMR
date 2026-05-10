#!/usr/bin/env python3
"""
Cause-ranking metrics for LaC / DFM2 pipelines using NARadio text embeddings.

Goal:
  - You have a *single* ground-truth cause label (string) per example.
  - You have a ranked list of candidate objects from:
      1) LaC (e.g. [{"object": str, "anxiety": ...}, ...])
      2) DFM2 (e.g. [{"name": str, "score": ...}, ...])
  - You want rank-based metrics like:
      - Success@1 (top-1 accuracy)
      - Recall@K
      - Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR across dataset)

Core idea:
  - Encode the GT label + each candidate string with the NARadio text encoder.
  - Compute cosine similarity between GT embedding and each candidate embedding.
  - A candidate is a "match" if similarity >= similarity_threshold.
  - The *rank* is taken from the original ordering of candidates (their VLM scores),
    not from similarity.

This file is intentionally pure-Python + NARadio, so you can import it from
anywhere (e.g. offline evaluation notebooks, CLI tools, or ROS-free scripts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from resilience.naradio_processor import NARadioProcessor


# ---------------------------------------------------------------------------
# Text encoder wrapper (NARadio)
# ---------------------------------------------------------------------------


class NaradioTextEncoder:
    """
    Thin wrapper around `NARadioProcessor` to encode short text labels.

    This only uses the language side (encode_labels), not image features.
    """

    def __init__(
        self,
        model_version: str = "radio_v2.5-b",
        lang_model: str = "siglip",
        device: Optional[str] = None,
    ) -> None:
        # NARadioProcessor already handles model loading/device selection.
        # We disable combined segmentation and visualization to keep it light.
        self._proc = NARadioProcessor(
            radio_model_version=model_version,
            radio_lang_model=lang_model,
            enable_visualization=False,
            enable_combined_segmentation=False,
            segmentation_config_path=None,
            cause_registry=None,
        )

        if not self._proc.naradio_ready or self._proc.radio_encoder is None:
            raise RuntimeError("NaradioTextEncoder: NARadio model not ready")

        # Optional manual device override.
        if device is not None:
            # Move encoder to target device if it has a `.to()` or `.model` attribute.
            enc = self._proc.radio_encoder
            if hasattr(enc, "to"):
                self._proc.radio_encoder = enc.to(device)  # type: ignore[assignment]

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def encode(self, labels: Sequence[str]) -> np.ndarray:
        """
        Encode a list of labels into a 2D numpy array of shape (N, D).
        """
        if not labels:
            return np.zeros((0, 0), dtype=np.float32)

        # NARadioEncoder expects a list[str].
        encoder = self._proc.radio_encoder
        if encoder is None:
            raise RuntimeError("NaradioTextEncoder: radio_encoder is None")

        with torch.no_grad():
            feats = encoder.encode_labels(list(labels))  # type: ignore[attr-defined]

        # Normalize to unit length for cosine similarity.
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Ranked candidate + per-example metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedCandidate:
    name: str
    score: float  # VLM score (higher = more likely). We *preserve* this ranking.


@dataclass(frozen=True)
class SingleExampleRankMetrics:
    gt_label: str
    similarities: List[float]          # cosine(GT, candidate_i)
    match_index: Optional[int]         # 0-based index of first match, or None
    success_at_1: float                # 0 or 1
    recall_at_k: float                 # 0 or 1 for given K
    reciprocal_rank: float             # 0 if no match, else 1/(rank+1)


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity for pairs of row vectors in a and b.

    a: (1, D) GT embedding
    b: (N, D) candidate embeddings
    Returns: (N,) similarities
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((b.shape[0],), dtype=np.float32)
    # a and b already L2-normalized; just dot.
    sims = (b @ a.T).reshape(-1)
    return sims.astype(np.float32)


def compute_single_example_rank_metrics(
    *,
    gt_label: str,
    candidates: Sequence[RankedCandidate],
    encoder: NaradioTextEncoder,
    similarity_threshold: float = 0.3,
    recall_k: int = 3,
) -> SingleExampleRankMetrics:
    """
    Compute rank-based metrics for a SINGLE example.

    - Ground truth: a single cause label (gt_label).
    - Candidates: already sorted by VLM (LaC / DFM2) descending score.
    - Matching rule: first index i where cosine_sim(gt, candidate_i) >= similarity_threshold.

    Metrics:
        success_at_1: 1.0 if the *first* candidate is a match, else 0.0.
        recall_at_k:  1.0 if ANY match appears within top-K, else 0.0.
        reciprocal_rank: 1 / (rank+1) if a match exists (0-based rank); else 0.0.
    """
    if not candidates:
        return SingleExampleRankMetrics(
            gt_label=gt_label,
            similarities=[],
            match_index=None,
            success_at_1=0.0,
            recall_at_k=0.0,
            reciprocal_rank=0.0,
        )

    # Encode GT and candidate names in one batch to share encoder overhead.
    labels = [gt_label] + [c.name for c in candidates]
    embeds = encoder.encode(labels)  # (1+N, D)

    if embeds.shape[0] != len(labels):
        raise RuntimeError("Encoder returned unexpected number of embeddings")

    gt_emb = embeds[0:1, :]        # (1, D)
    cand_embs = embeds[1:, :]      # (N, D)

    sims = _cosine_sim_matrix(gt_emb, cand_embs)

    # Find first index where similarity exceeds threshold (preserving VLM rank).
    match_idx: Optional[int] = None
    for i, s in enumerate(sims):
        if s >= similarity_threshold:
            match_idx = i
            break

    # Success@1: did the TOP-1 candidate match?
    success_at_1 = 1.0 if match_idx == 0 else 0.0

    # Recall@K: did ANY candidate up to K match?
    k = max(1, int(recall_k))
    recall_at_k = 1.0 if (match_idx is not None and match_idx < k) else 0.0

    # Reciprocal Rank: 1/(rank+1) if we found a match, else 0.
    if match_idx is None:
        rr = 0.0
    else:
        rr = 1.0 / float(match_idx + 1)

    return SingleExampleRankMetrics(
        gt_label=gt_label,
        similarities=list(float(x) for x in sims),
        match_index=match_idx,
        success_at_1=success_at_1,
        recall_at_k=recall_at_k,
        reciprocal_rank=rr,
    )


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregateRankMetrics:
    n_examples: int
    mean_success_at_1: float
    mean_recall_at_k: float
    mrr: float


def aggregate_rank_metrics(
    results: Iterable[SingleExampleRankMetrics],
) -> AggregateRankMetrics:
    """
    Aggregate per-example rank metrics across a dataset.
    """
    res_list = list(results)
    n = len(res_list)
    if n == 0:
        return AggregateRankMetrics(
            n_examples=0,
            mean_success_at_1=0.0,
            mean_recall_at_k=0.0,
            mrr=0.0,
        )

    s1 = sum(r.success_at_1 for r in res_list) / n
    rk = sum(r.recall_at_k for r in res_list) / n
    mrr = sum(r.reciprocal_rank for r in res_list) / n

    return AggregateRankMetrics(
        n_examples=n,
        mean_success_at_1=float(s1),
        mean_recall_at_k=float(rk),
        mrr=float(mrr),
    )


# ---------------------------------------------------------------------------
# Convenience helpers for LaC / DFM2 outputs
# ---------------------------------------------------------------------------


def candidates_from_lac_objects(objs: Sequence[Dict]) -> List[RankedCandidate]:
    """
    Convert LaC objects list:
        [{"object": str, "anxiety": int, "score": float}, ...]
    to RankedCandidate list sorted by `score` descending.
    """
    cands: List[RankedCandidate] = []
    for o in objs:
        name = str(o.get("object", "")).strip()
        if not name:
            continue
        # Prefer normalized "score" (0..1) if present; fall back to anxiety/3.
        if "score" in o:
            try:
                s = float(o["score"])
            except Exception:
                s = 0.0
        else:
            try:
                a = float(o.get("anxiety", 0.0))
            except Exception:
                a = 0.0
            s = max(0.0, min(1.0, a / 3.0))
        cands.append(RankedCandidate(name=name, score=s))

    # Sort descending by score (VLM ranking).
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def candidates_from_dfm2_objects(objs: Sequence[Dict]) -> List[RankedCandidate]:
    """
    Convert DFM2 objects list:
        [{"name": str, "score": float}, ...]
    to RankedCandidate list sorted by `score` descending.
    """
    cands: List[RankedCandidate] = []
    for o in objs:
        name = str(o.get("name", "")).strip()
        if not name:
            continue
        try:
            s = float(o.get("score", 0.0))
        except Exception:
            s = 0.0
        cands.append(RankedCandidate(name=name, score=s))

    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


