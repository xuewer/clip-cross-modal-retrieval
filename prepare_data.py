from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_DATA_DIR, write_json
from src.dataset import dataframe_summary, load_and_clean_captions, split_by_image_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Flickr-style captions CSV and split by image_id.")
    parser.add_argument("--input", type=Path, required=True, help="Input captions CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR / "splits")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip opening every image; useful for slow filesystems.",
    )
    args = parser.parse_args()

    cleaned, stats = load_and_clean_captions(
        args.input,
        validate_images=not args.skip_image_validation,
    )
    train, test = split_by_image_id(cleaned, test_size=args.test_size, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output_dir / "captions_clean.csv", index=False)
    train.to_csv(args.output_dir / "train.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)
    write_json(
        args.output_dir / "cleaning_stats.json",
        {
            "cleaning": stats.as_dict(),
            "cleaned": dataframe_summary(cleaned),
            "train": dataframe_summary(train),
            "test": dataframe_summary(test),
            "split": {"test_size": args.test_size, "seed": args.seed},
            "image_validation": {
                "performed": not args.skip_image_validation,
                "note": (
                    "All unique image files were opened and verified."
                    if not args.skip_image_validation
                    else "Skipped by command-line option; run without --skip-image-validation for strict checks."
                ),
            },
        },
    )
    print("Data preparation complete.")
    print(stats.as_dict())
    print(f"Train: {len(train)} rows, {train['image_id'].nunique()} images")
    print(f"Test:  {len(test)} rows, {test['image_id'].nunique()} images")


if __name__ == "__main__":
    main()
