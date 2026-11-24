from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_MODEL_NAME, DEFAULT_OUTPUTS_DIR, write_json
from src.dataset import load_and_clean_captions
from src.metrics import evaluate_dual_recall
from src.model import ClipEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dual-direction Recall@K on an independent image split.")
    parser.add_argument("--csv", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS_DIR / "metrics.json")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", None])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    frame, stats = load_and_clean_captions(args.csv)
    print(
        f"Validated {stats.output_rows} rows and {frame['image_id'].nunique()} images.",
        flush=True,
    )
    print(f"Loading CLIP model on {args.device or 'auto-selected device'}...", flush=True)
    encoder = ClipEncoder(model_name=args.model, device=args.device)
    print("Model loaded. Encoding evaluation galleries...", flush=True)
    metrics = evaluate_dual_recall(frame, encoder, batch_size=args.batch_size)
    metrics["data_validation"] = stats.as_dict()
    write_json(args.output, metrics)
    print(f"Metrics written to {args.output.resolve()}")
    for direction in ("text_to_image", "image_to_text"):
        print(direction, metrics[direction])


if __name__ == "__main__":
    main()
