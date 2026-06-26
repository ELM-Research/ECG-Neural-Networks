from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from einops import rearrange
from transformers import AutoConfig, AutoModel


@dataclass
class SiglipConfig:
    model: str = "google/siglip2-base-patch16-naflex"
    segment_len: int = 2500
    patch_size: int = 100
    num_leads: int = 12
    d_model: int = None


@dataclass
class SiglipOutput:
    loss: Optional[torch.Tensor]
    out: Optional[torch.Tensor]


class Ecg1DEmbeddings(nn.Module):
    # 1D ECG patch stem replacing SigLIP's 2D image patch embedding
    def __init__(self, patch_dim: int, num_patches: int, hidden_size: int):
        super().__init__()
        self.patch_embedding = nn.Linear(patch_dim, hidden_size)
        self.position_embedding = nn.Embedding(num_patches, hidden_size)

    def forward(self, pixel_values, spatial_shapes=None):  # spatial_shapes unused (1D)
        pos = torch.arange(pixel_values.shape[1], device=pixel_values.device)
        return self.patch_embedding(pixel_values) + self.position_embedding(pos).unsqueeze(0)


class SigLIP(nn.Module):
    def __init__(self, cfg: SiglipConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.segment_len % cfg.patch_size == 0, "segment_len must be divisible by patch_size"
        # SigLIP architecture from config, randomly initialized (trained from scratch)
        self.vision_encoder = AutoModel.from_config(AutoConfig.from_pretrained(cfg.model))
        # swap the 2D image patch stem for a 1D ECG stem: each time-patch keeps all 12 leads
        hidden = self.vision_encoder.config.vision_config.hidden_size
        self.vision_encoder.vision_model.embeddings = Ecg1DEmbeddings(
            cfg.num_leads * cfg.patch_size, cfg.segment_len // cfg.patch_size, hidden
        )
        self.cfg.d_model = hidden

    def forward(self, signal, condition, **kwargs):
        # (B, 12, L) -> (B, num_patches, 12*patch_size): split time, keep all leads per patch
        patches = rearrange(signal, "b c (n p) -> b n (c p)", p=self.cfg.patch_size)
        spatial_shapes = torch.zeros(patches.shape[0], 2, dtype=torch.long, device=patches.device)
        out = self.vision_encoder(
            input_ids=condition["input_ids"],
            attention_mask=condition.get("attention_mask"),
            pixel_values=patches,
            spatial_shapes=spatial_shapes,
            return_loss=True,
        )
        return SiglipOutput(loss=out.loss, out=out.logits_per_text)
