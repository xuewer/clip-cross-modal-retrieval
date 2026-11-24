from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataset import DatasetError, split_by_image_id
from src.metrics import evaluate_dual_recall, recall_at_k
from src.retrieval import top_k_search
from src.utils import normalize_rows


class CountingEncoder:
    model_name = "test-model"
    device_type = "cpu"

    def __init__(self) -> None:
        self.image_calls = 0
        self.text_calls = 0

    def encode_images(self, images: list[str], batch_size: int = 16) -> np.ndarray:
        self.image_calls += 1
        return normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.text_calls += 1
        return normalize_rows(
            np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        )


def test_normalize_rows_l2_normalizes_each_row() -> None:
    values = np.array([[3.0, 4.0], [1.0, 0.0]])
    normalized = normalize_rows(values)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])


def test_top_k_search_is_sorted_and_caps_k() -> None:
    matrix = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]))
    indices, scores = top_k_search(np.array([1.0, 0.0]), matrix, 10)
    assert indices.tolist() == [0, 2, 1]
    assert np.all(scores[:-1] >= scores[1:])


def test_recall_at_k_matches_any_valid_positive() -> None:
    ranked = [["a", "b", "c"], ["x", "y"], ["q"]]
    truths = [{"b"}, {"z", "y"}, {"missing"}]
    assert recall_at_k(ranked, truths, 2) == pytest.approx(2 / 3)


def test_evaluation_reuses_gallery_embeddings_for_queries() -> None:
    frame = pd.DataFrame(
        {
            "image_id": ["a", "a", "b"],
            "image_path": ["a.jpg", "a.jpg", "b.jpg"],
            "caption": ["a first", "a second", "b first"],
        }
    )
    encoder = CountingEncoder()

    metrics = evaluate_dual_recall(frame, encoder)

    assert encoder.image_calls == 1
    assert encoder.text_calls == 1
    assert metrics["text_to_image"]["recall@1"] == pytest.approx(1.0)
    assert metrics["image_to_text"]["recall@1"] == pytest.approx(1.0)


def test_split_by_image_id_has_no_leakage() -> None:
    frame = pd.DataFrame(
        {
            "image_id": ["a", "a", "b", "c", "c", "d"],
            "image_path": ["a.jpg"] * 2 + ["b.jpg", "c.jpg", "c.jpg", "d.jpg"],
            "caption": ["one", "two", "three", "four", "five", "six"],
        }
    )
    train, test = split_by_image_id(frame, test_size=0.25, seed=7)
    assert not set(train["image_id"]) & set(test["image_id"])
    assert set(train["image_id"]) | set(test["image_id"]) == set(frame["image_id"])


def test_split_rejects_single_image() -> None:
    frame = pd.DataFrame(
        {"image_id": ["a"], "image_path": ["a.jpg"], "caption": ["one"]}
    )
    with pytest.raises(DatasetError):
        split_by_image_id(frame)


def test_top_k_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        top_k_search(np.array([1.0, 0.0]), np.eye(2), 0)
