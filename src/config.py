from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_WEIGHTS = PROJECT_ROOT / "models" / "ViT-B-32.pt"
DEFAULT_MODEL_NAME = "ViT-B/32"


def select_device(requested: str | None = None) -> str:
    """Select CUDA when available unless the caller explicitly requests a device."""
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
