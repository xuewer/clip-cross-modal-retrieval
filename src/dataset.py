from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError


REQUIRED_COLUMNS = ["image_id", "image_path", "caption"]


class DatasetError(ValueError):
    """Raised when a captions CSV cannot be used safely."""


@dataclass
class DatasetStats:
    input_rows: int
    output_rows: int
    duplicate_rows_removed: int
    empty_rows_removed: int
    missing_image_rows_removed: int
    corrupt_image_rows_removed: int

    def as_dict(self) -> dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "empty_rows_removed": self.empty_rows_removed,
            "missing_image_rows_removed": self.missing_image_rows_removed,
            "corrupt_image_rows_removed": self.corrupt_image_rows_removed,
        }


def _resolve_image_path(raw_path: str, csv_path: Path) -> str:
    path = Path(str(raw_path).strip())
    if not path.is_absolute():
        path = csv_path.parent / path
    return str(path) if path.is_absolute() else str(path.resolve())


def _check_image(path: Path) -> tuple[bool, bool]:
    if not path.is_file():
        return False, False
    try:
        with Image.open(path) as image:
            image.verify()
        return True, True
    except (OSError, UnidentifiedImageError):
        return True, False


def load_and_clean_captions(
    csv_path: Path,
    validate_images: bool = True,
) -> tuple[pd.DataFrame, DatasetStats]:
    csv_path = Path(csv_path).resolve()
    if not csv_path.is_file():
        raise DatasetError(f"Captions CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise DatasetError(
            f"CSV must contain columns {REQUIRED_COLUMNS}; missing {missing_columns}."
        )

    input_rows = len(frame)
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["image_id"] = frame["image_id"].fillna("").astype(str).str.strip()
    frame["image_path"] = frame["image_path"].fillna("").astype(str).str.strip()
    frame["caption"] = frame["caption"].fillna("").astype(str).str.strip()

    empty_mask = (frame["image_id"] == "") | (frame["image_path"] == "") | (frame["caption"] == "")
    empty_rows_removed = int(empty_mask.sum())
    frame = frame.loc[~empty_mask].copy()

    before_dedup = len(frame)
    frame = frame.drop_duplicates(subset=REQUIRED_COLUMNS, keep="first").reset_index(drop=True)
    duplicate_rows_removed = before_dedup - len(frame)
    path_map = {
        value: _resolve_image_path(value, csv_path)
        for value in frame["image_path"].drop_duplicates()
    }
    frame["image_path"] = frame["image_path"].map(path_map)

    missing_image_rows_removed = 0
    corrupt_image_rows_removed = 0
    if validate_images:
        unique_paths = frame["image_path"].drop_duplicates().tolist()
        with ThreadPoolExecutor(max_workers=8) as executor:
            checked = executor.map(lambda value: _check_image(Path(value)), unique_paths)
        validation_cache = dict(zip(unique_paths, checked))
        valid_rows: list[bool] = []
        for raw_path in frame["image_path"]:
            if raw_path not in validation_cache:
                validation_cache[raw_path] = _check_image(Path(raw_path))
            exists, valid = validation_cache[raw_path]
            if not exists:
                missing_image_rows_removed += 1
            elif not valid:
                corrupt_image_rows_removed += 1
            valid_rows.append(exists and valid)
        frame = frame.loc[valid_rows].reset_index(drop=True)

    if frame.empty:
        raise DatasetError("No valid caption rows remain after cleaning.")
    if frame["image_id"].nunique() < 2:
        raise DatasetError("At least two distinct image_id values are required for a train/test split.")

    stats = DatasetStats(
        input_rows=input_rows,
        output_rows=len(frame),
        duplicate_rows_removed=duplicate_rows_removed,
        empty_rows_removed=empty_rows_removed,
        missing_image_rows_removed=missing_image_rows_removed,
        corrupt_image_rows_removed=corrupt_image_rows_removed,
    )
    return frame, stats


def split_by_image_id(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise DatasetError("test_size must be between 0 and 1.")
    image_ids = frame["image_id"].drop_duplicates().tolist()
    if len(image_ids) < 2:
        raise DatasetError("At least two distinct image_id values are required for a split.")

    shuffled = pd.Series(image_ids).sample(frac=1.0, random_state=seed).tolist()
    test_count = max(1, int(round(len(shuffled) * test_size)))
    test_count = min(test_count, len(shuffled) - 1)
    test_ids = set(shuffled[:test_count])

    train = frame.loc[~frame["image_id"].isin(test_ids)].reset_index(drop=True)
    test = frame.loc[frame["image_id"].isin(test_ids)].reset_index(drop=True)
    if train.empty or test.empty:
        raise DatasetError("The split produced an empty train or test set.")
    if set(train["image_id"]) & set(test["image_id"]):
        raise DatasetError("Image leakage detected: train and test image_id sets overlap.")
    return train, test


def dataframe_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "unique_images": int(frame["image_id"].nunique()),
        "captions_per_image_mean": float(frame.groupby("image_id").size().mean()),
    }
