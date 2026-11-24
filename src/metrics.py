from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .model import ClipEncoder


def recall_at_k(
    ranked_ids: Iterable[Iterable[str]],
    truth_ids: Iterable[set[str]],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")
    ranked = list(ranked_ids)
    truths = list(truth_ids)
    if len(ranked) != len(truths):
        raise ValueError("ranked_ids and truth_ids must contain the same number of queries.")
    if not ranked:
        raise ValueError("At least one query is required to calculate recall.")
    hits = sum(bool(set(list(predictions)[:k]) & truths[index]) for index, predictions in enumerate(ranked))
    return hits / len(ranked)


def _unique_images(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["image_id", "image_path"]].drop_duplicates("image_id").reset_index(drop=True)


def _encode_batched(
    encoder: ClipEncoder,
    items: list[str],
    *,
    modality: str,
    batch_size: int,
) -> np.ndarray:
    encode = encoder.encode_images if modality == "image" else encoder.encode_texts
    batches = []
    for start in tqdm(
        range(0, len(items), batch_size),
        desc=f"Encoding {modality}s",
        unit="batch",
    ):
        batches.append(encode(items[start : start + batch_size], batch_size=batch_size))
    return np.vstack(batches)


def _batched_top_k_indices(
    queries: np.ndarray,
    candidates: np.ndarray,
    k: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Return stable top-k candidate indices without sorting a full score matrix."""
    limit = min(k, len(candidates))
    result: list[np.ndarray] = []
    candidate_tensor = torch.from_numpy(candidates).float()
    for start in range(0, len(queries), batch_size):
        query_tensor = torch.from_numpy(queries[start : start + batch_size]).float()
        scores = query_tensor @ candidate_tensor.T
        ordered = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :limit]
        result.append(ordered.numpy())
    return np.vstack(result)


def evaluate_dual_recall(
    test_frame: pd.DataFrame,
    encoder: ClipEncoder,
    ks: tuple[int, ...] = (1, 5, 10),
    batch_size: int = 16,
) -> dict:
    required = {"image_id", "image_path", "caption"}
    if not required.issubset(test_frame.columns):
        raise ValueError(f"Test frame must contain {sorted(required)}.")
    if test_frame.empty or test_frame["image_id"].nunique() < 1:
        raise ValueError("The test set must contain at least one valid image and caption.")

    image_gallery = _unique_images(test_frame)
    text_gallery = test_frame[["image_id", "caption"]].copy().reset_index(drop=True)
    text_gallery.insert(0, "text_id", [f"text_{i:06d}" for i in range(len(text_gallery))])
    image_vectors = _encode_batched(
        encoder,
        image_gallery["image_path"].tolist(),
        modality="image",
        batch_size=batch_size,
    )
    text_vectors = _encode_batched(
        encoder,
        text_gallery["caption"].tolist(),
        modality="text",
        batch_size=batch_size,
    )

    # Evaluation queries are the same held-out items used to form each gallery.
    # Reuse their embeddings instead of running the model a second time.
    text_queries = text_vectors
    image_queries = image_vectors

    text_order = _batched_top_k_indices(text_queries, image_vectors, max(ks))
    image_ids = image_gallery["image_id"].astype(str).to_numpy()
    text_ranked_ids = image_ids[text_order].tolist()
    text_truth = [{str(image_id)} for image_id in test_frame["image_id"]]

    image_order = _batched_top_k_indices(image_queries, text_vectors, max(ks))
    text_ids = text_gallery["text_id"].astype(str).to_numpy()
    image_ranked_ids = text_ids[image_order].tolist()
    image_truth = [
        set(text_gallery.loc[text_gallery["image_id"] == image_id, "text_id"].astype(str))
        for image_id in image_gallery["image_id"]
    ]

    return {
        "config": {
            "gallery_split": "test",
            "query_split": "test",
            "image_gallery_count": int(len(image_gallery)),
            "text_gallery_count": int(len(text_gallery)),
            "text_query_count": int(len(text_queries)),
            "image_query_count": int(len(image_queries)),
            "model_name": encoder.model_name,
            "device": encoder.device_type,
            "ks": list(ks),
        },
        "text_to_image": {
            f"recall@{k}": recall_at_k(text_ranked_ids, text_truth, k) for k in ks
        },
        "image_to_text": {
            f"recall@{k}": recall_at_k(image_ranked_ids, image_truth, k) for k in ks
        },
    }
