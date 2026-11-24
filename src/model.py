from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_MODEL_NAME, DEFAULT_MODEL_WEIGHTS, select_device
from .utils import normalize_rows


class ClipModelError(RuntimeError):
    """Raised when OpenAI CLIP cannot be loaded or used."""


class ClipEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = select_device(device)
        self.backend = ""
        if model_name == DEFAULT_MODEL_NAME and DEFAULT_MODEL_WEIGHTS.is_file():
            try:
                self._load_open_clip_local(DEFAULT_MODEL_WEIGHTS)
                return
            except Exception as exc:
                raise ClipModelError(
                    f"Failed to load local OpenAI CLIP weights at {DEFAULT_MODEL_WEIGHTS}. "
                    "The file may be incomplete or incompatible."
                ) from exc
        try:
            import clip
            self.clip = clip
            self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
            self.model.eval()
            self.backend = "openai-clip"
        except ImportError:
            self._load_transformers_backend()
        except Exception as exc:
            raise ClipModelError(
                f"Failed to load CLIP model {model_name!r} on {self.device}. "
                "Check network access for the first model download and available disk space."
            ) from exc

    def _load_open_clip_local(self, weights_path: Path) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "The local OpenAI CLIP checkpoint requires `open_clip_torch`."
            ) from exc

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=str(weights_path),
            device=self.device,
            weights_only=False,
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()
        self.backend = "open-clip-local"

    def _load_transformers_backend(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ClipModelError(
                "Neither OpenAI CLIP nor Transformers is installed. "
                "Run `pip install -r requirements.txt`."
            ) from exc

        hf_name = (
            "openai/clip-vit-base-patch32"
            if self.model_name == DEFAULT_MODEL_NAME
            else self.model_name
        )
        try:
            self.processor = CLIPProcessor.from_pretrained(hf_name)
            self.model = CLIPModel.from_pretrained(hf_name).to(self.device)
            self.model.eval()
            self.backend = "transformers-openai-clip"
            self.model_name = hf_name
        except Exception as exc:
            raise ClipModelError(
                f"Failed to load OpenAI CLIP weights {hf_name!r}. "
                "The package may be installed, but the model files must also be downloaded "
                "once with network access or supplied from a local cache."
            ) from exc

    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            raise ValueError("At least one text query is required.")
        if any(not str(text).strip() for text in texts):
            raise ValueError("Text queries must not be empty.")

        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                if self.backend == "openai-clip":
                    tokens = self.clip.tokenize(batch).to(self.device)
                    features = self.model.encode_text(tokens).float().cpu().numpy()
                elif self.backend == "open-clip-local":
                    tokens = self.tokenizer(batch).to(self.device)
                    features = self.model.encode_text(tokens).float().cpu().numpy()
                else:
                    inputs = self.processor(
                        text=batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    features = self.model.get_text_features(**inputs).float().cpu().numpy()
                outputs.append(normalize_rows(features))
        return np.vstack(outputs)

    def encode_images(
        self,
        images: Sequence[str | Path | Image.Image],
        batch_size: int = 16,
    ) -> np.ndarray:
        if not images:
            raise ValueError("At least one image is required.")

        image_items = list(images)
        for item in image_items:
            if isinstance(item, Image.Image):
                continue
            path = Path(item)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path}")

        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(image_items), batch_size):
                batch: list[Image.Image] = []
                for item in image_items[start : start + batch_size]:
                    if isinstance(item, Image.Image):
                        batch.append(item.convert("RGB"))
                        continue
                    try:
                        with Image.open(Path(item)) as image:
                            batch.append(image.convert("RGB").copy())
                    except OSError as exc:
                        raise ValueError(f"Unable to open image: {item}") from exc
                if self.backend in {"openai-clip", "open-clip-local"}:
                    tensors = torch.stack([self.preprocess(image) for image in batch]).to(self.device)
                    features = self.model.encode_image(tensors).float().cpu().numpy()
                else:
                    inputs = self.processor(images=batch, return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    features = self.model.get_image_features(**inputs).float().cpu().numpy()
                outputs.append(normalize_rows(features))
        return np.vstack(outputs)

    @property
    def device_type(self) -> str:
        return self.device
