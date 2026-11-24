from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


IMAGE_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(
            f"Download failed for {url}. Check network/proxy access and try again."
        ) from exc


def extract(zip_path: Path, output_dir: Path) -> None:
    print(f"Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def make_captions_csv(text_file: Path, images_dir: Path, output_csv: Path) -> int:
    import csv

    rows: list[dict[str, str]] = []
    images_dir = images_dir.absolute()
    with text_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            filename, caption = line.split("\t", 1)
            image_filename = filename.split("#", 1)[0]
            image_id = Path(image_filename).stem
            image_path = images_dir / image_filename
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": os.path.relpath(image_path, output_csv.parent),
                    "caption": caption.strip(),
                }
            )
    if not rows:
        raise RuntimeError(f"No image-caption rows found in {text_file}.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "caption"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare Flickr8k.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--skip-images", action="store_true", help="Only download text annotations.")
    parser.add_argument("--keep-zips", action="store_true")
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    text_zip = raw_dir / "Flickr8k_text.zip"
    image_zip = raw_dir / "Flickr8k_Dataset.zip"

    download(TEXT_URL, text_zip)
    extract(text_zip, raw_dir)
    if not args.skip_images:
        download(IMAGE_URL, image_zip)
        extract(image_zip, raw_dir)

    text_file = raw_dir / "Flickr8k.token.txt"
    images_dir = raw_dir / "Flicker8k_Dataset"
    if not text_file.is_file():
        raise FileNotFoundError(f"Expected annotation file not found: {text_file}")
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Expected image directory not found: {images_dir}. "
            "Run without --skip-images to download the image archive."
        )
    rows = make_captions_csv(text_file, images_dir, args.data_dir / "captions.csv")

    if not args.keep_zips:
        for archive in (text_zip, image_zip):
            if archive.is_file():
                archive.unlink()
    print(f"Prepared {rows} rows in {args.data_dir / 'captions.csv'}.")
    print("Next: python prepare_data.py --input data/captions.csv")


if __name__ == "__main__":
    main()
