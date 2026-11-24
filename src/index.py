from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_ARTIFACTS_DIR, DEFAULT_MODEL_NAME, read_json, write_json
from .model import ClipEncoder
from .utils import sha256_file


INDEX_FILES = {
    "image_vectors": "image_vectors.npy",
    "text_vectors": "text_vectors.npy",
    "image_metadata": "image_metadata.csv",
    "text_metadata": "text_metadata.csv",
    "config": "index_config.json",
}


@dataclass
class VectorIndex:
    image_vectors: np.ndarray
    text_vectors: np.ndarray
    image_metadata: pd.DataFrame
    text_metadata: pd.DataFrame
    config: dict[str, Any]


def build_index(
    frame: pd.DataFrame,
    encoder: ClipEncoder,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    source_csv: Path | None = None,
    batch_size: int = 16,
) -> VectorIndex:
    required = {"image_id", "image_path", "caption"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Frame is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot build an index from an empty dataframe.")

    image_metadata = frame[["image_id", "image_path"]].drop_duplicates("image_id").reset_index(drop=True)
    text_metadata = frame[["image_id", "image_path", "caption"]].copy().reset_index(drop=True)
    text_metadata.insert(0, "text_id", [f"text_{i:06d}" for i in range(len(text_metadata))])

    image_vectors = encoder.encode_images(image_metadata["image_path"].tolist(), batch_size=batch_size)
    text_vectors = encoder.encode_texts(text_metadata["caption"].tolist(), batch_size=batch_size)

    config: dict[str, Any] = {
        "artifact_version": 1,
        "model_name": encoder.model_name,
        "device": encoder.device_type,
        "normalized": True,
        "image_count": int(len(image_metadata)),
        "text_count": int(len(text_metadata)),
        "source_csv": str(source_csv.resolve()) if source_csv else None,
        "source_csv_sha256": sha256_file(source_csv) if source_csv and source_csv.is_file() else None,
        "embedding_dimension": int(image_vectors.shape[1]),
    }
    index = VectorIndex(
        image_vectors=image_vectors,
        text_vectors=text_vectors,
        image_metadata=image_metadata,
        text_metadata=text_metadata,
        config=config,
    )
    save_index(index, artifacts_dir)
    return index


def save_index(index: VectorIndex, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / INDEX_FILES["image_vectors"], index.image_vectors.astype(np.float32))
    np.save(artifacts_dir / INDEX_FILES["text_vectors"], index.text_vectors.astype(np.float32))
    index.image_metadata.to_csv(artifacts_dir / INDEX_FILES["image_metadata"], index=False)
    index.text_metadata.to_csv(artifacts_dir / INDEX_FILES["text_metadata"], index=False)
    write_json(artifacts_dir / INDEX_FILES["config"], index.config)


def load_index(artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> VectorIndex:
    artifacts_dir = Path(artifacts_dir)
    missing = [
        name for name in INDEX_FILES.values() if not (artifacts_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Index artifacts are incomplete in {artifacts_dir}: missing {missing}. "
            "Run `python build_index.py` first."
        )
    return VectorIndex(
        image_vectors=np.load(artifacts_dir / INDEX_FILES["image_vectors"]),
        text_vectors=np.load(artifacts_dir / INDEX_FILES["text_vectors"]),
        image_metadata=pd.read_csv(artifacts_dir / INDEX_FILES["image_metadata"]),
        text_metadata=pd.read_csv(artifacts_dir / INDEX_FILES["text_metadata"]),
        config=read_json(artifacts_dir / INDEX_FILES["config"]),
    )
