from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


@dataclass
class SigLIP2Config:
    model: str = "google/siglip2-base-patch16-naflex"
    segment_len: int = 2500
    patch_size: int = 25
    num_leads: int = 12
    d_model: int = None
    pretrained: bool = True


@dataclass
class SigLIP2Output:
    loss: Optional[torch.Tensor]
    out: Optional[torch.Tensor]


class ECGEmbedding(nn.Module):
    def __init__(self, patch_dim, num_patches, hidden_size):
        super().__init__()
        self.patch_embedding = nn.Linear(patch_dim, hidden_size)
        self.position_embedding = nn.Embedding(num_patches, hidden_size)
        nn.init.trunc_normal_(self.patch_embedding.weight, std=patch_dim**-0.5)
        nn.init.zeros_(self.patch_embedding.bias)
        nn.init.normal_(self.position_embedding.weight, std=hidden_size**-0.5)

    def forward(self, patches, spatial_shapes=None):
        positions = torch.arange(patches.shape[1], device=patches.device)
        return self.patch_embedding(patches) + self.position_embedding(positions)[None]


class ECGEncoder(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.model.embeddings = ECGEmbedding(
            config.num_leads * config.patch_size,
            config.segment_len // config.patch_size,
            model.config.hidden_size,
        )
        self.shape = (config.num_leads, config.segment_len)
        self.patch_size = config.patch_size

    def forward(self, signal):
        if tuple(signal.shape[1:]) != self.shape:
            raise ValueError(f"Expected ECG shape (batch, {self.shape[0]}, {self.shape[1]}), got {tuple(signal.shape)}")
        patches = signal.unfold(-1, self.patch_size, self.patch_size).transpose(1, 2).flatten(2)
        spatial_shapes = torch.zeros((patches.shape[0], 2), dtype=torch.long, device=patches.device)
        return self.model(patches, None, spatial_shapes)


class SigLIP2(nn.Module):
    def __init__(self, cfg: SigLIP2Config):
        super().__init__()
        if cfg.segment_len % cfg.patch_size:
            raise ValueError("segment_len must be divisible by patch_size")
        self.cfg = cfg
        model = (AutoModel.from_pretrained(cfg.model) if cfg.pretrained
                 else AutoModel.from_config(AutoConfig.from_pretrained(cfg.model)))
        self.encoder = ECGEncoder(model.vision_model, cfg)
        self.text_model = model.text_model
        self.logit_scale = model.logit_scale
        self.logit_bias = model.logit_bias
        self.cfg.d_model = model.config.vision_config.hidden_size

    def forward(self, signal, condition):
        vision_output = self.encoder(signal)
        text_output = self.text_model(
            input_ids=condition["input_ids"],
            attention_mask=condition.get("attention_mask"),
        )
        image_embeds = vision_output.pooler_output
        text_embeds = text_output.pooler_output
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logits = text_embeds @ image_embeds.t().to(text_embeds.device)
        logits = logits * self.logit_scale.to(text_embeds.device).exp() + self.logit_bias.to(text_embeds.device)
        labels = torch.eye(logits.shape[0], device=logits.device).mul(2).sub(1)
        loss = -torch.nn.functional.logsigmoid(labels * logits).sum(-1).mean()
        return SigLIP2Output(loss=loss, out=logits)

    def load_state_dict(self, state_dict, *args, **kwargs):
        if any(key.startswith("vision_encoder.") for key in state_dict):
            state_dict = {
                ("encoder.model." + key.removeprefix("vision_encoder.vision_model.")
                 if key.startswith("vision_encoder.vision_model.") else key.removeprefix("vision_encoder.")): value
                for key, value in state_dict.items()
            }
        return super().load_state_dict(state_dict, *args, **kwargs)
