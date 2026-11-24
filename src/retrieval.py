from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .index import VectorIndex
from .model import ClipEncoder
from .utils import normalize_rows


def top_k_search(query_vector: np.ndarray, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k < 1:
        raise ValueError("k must be at least 1.")
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    candidates = np.asarray(matrix, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError("The candidate matrix must be a non-empty 2D array.")
    if query.shape[0] != candidates.shape[1]:
        raise ValueError("Query and candidate vector dimensions do not match.")
    query = normalize_rows(query)[0]
    scores = candidates @ query
    limit = min(k, len(scores))
    indices = np.arange(len(scores))
    ordered = np.lexsort((indices, -scores))
    selected = ordered[:limit]
    return selected, scores[selected]


class Retriever:
    def __init__(self, index: VectorIndex, encoder: ClipEncoder) -> None:
        self.index = index
        self.encoder = encoder

    def search_text(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        if not text or not text.strip():
            raise ValueError("Text query must not be empty.")
        query = self.encoder.encode_texts([text])[0]
        indices, scores = top_k_search(query, self.index.image_vectors, k)
        results: list[dict[str, Any]] = []
        for idx, score in zip(indices, scores):
            row = self.index.image_metadata.iloc[int(idx)]
            results.append(
                {
                    "image_id": str(row["image_id"]),
                    "image_path": str(row["image_path"]),
                    "score": float(score),
                }
            )
        return results

    def search_image(
        self,
        image: str | Path | Image.Image,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        query = self.encoder.encode_images([image])[0]
        indices, scores = top_k_search(query, self.index.text_vectors, k)
        results: list[dict[str, Any]] = []
        for idx, score in zip(indices, scores):
            row = self.index.text_metadata.iloc[int(idx)]
            results.append(
                {
                    "text_id": str(row["text_id"]),
                    "image_id": str(row["image_id"]),
                    "caption": str(row["caption"]),
                    "image_path": str(row["image_path"]),
                    "score": float(score),
                }
            )
        return results
