from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_MODEL_NAME
from src.dataset import load_and_clean_captions
from src.index import build_index
from src.model import ClipEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a captions CSV into a reusable CLIP vector index.")
    parser.add_argument("--csv", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", None])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    frame, stats = load_and_clean_captions(args.csv)
    print(f"Validated {stats.output_rows} rows and {frame['image_id'].nunique()} images.")
    encoder = ClipEncoder(model_name=args.model, device=args.device)
    index = build_index(
        frame,
        encoder,
        artifacts_dir=args.artifacts_dir,
        source_csv=args.csv,
        batch_size=args.batch_size,
    )
    print(f"Index written to {args.artifacts_dir.resolve()}.")
    print(f"Images: {len(index.image_metadata)}, texts: {len(index.text_metadata)}")


if __name__ == "__main__":
    main()
