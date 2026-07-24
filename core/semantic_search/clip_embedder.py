"""CLIP image/text embedder via open_clip."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
from PIL import Image

from core.config import Settings, get_settings

# ViT-B-32 openai → 512-d
CLIP_DIM = 512


class ClipEmbedder:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.device = self.settings.clip_device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.settings.clip_model,
            pretrained=self.settings.clip_pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.settings.clip_model)
        self.model.eval()
        self.dim = CLIP_DIM

    @torch.inference_mode()
    def embed_image(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        image = self.preprocess(Image.open(path).convert("RGB"))
        image = image.unsqueeze(0).to(self.device)
        features = self.model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32).reshape(-1)

    @torch.inference_mode()
    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32).reshape(-1)


@lru_cache(maxsize=1)
def get_clip_embedder() -> ClipEmbedder:
    return ClipEmbedder()
