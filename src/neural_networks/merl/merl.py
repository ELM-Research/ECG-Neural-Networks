import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModel

from neural_networks.merl.blocks import AttentionPool2d, get_resnet

@dataclass
class MerlConfig:
    proj_hidden: int = 256
    proj_out: int = 256
    in_channels: int = 2048
    num_layers: int = 12
    dropout: float = 0.1
    seq_len: int = 2500
    lm: str = "ncbi/MedCPT-Query-Encoder"
    frozen_text_layers: int = 6
    resnet_type: str = "resnet101"
    distributed: bool = False
    spacial_dim: int = None
    d_model: int = 2048

    def __post_init__(self):
        if self.seq_len == 2500:
            self.spacial_dim = 157
        elif self.seq_len == 1250:
            self.spacial_dim = 79
        else:
            self.spacial_dim = 32


@dataclass
class MerlOutput:
    loss: Optional[torch.Tensor]
    out: Optional[torch.Tensor]


class Merl(nn.Module):
    def __init__(self, cfg: MerlConfig):
        super().__init__()
        self.cfg = cfg
        self.resnet = get_resnet(cfg.resnet_type)
        self.lm = AutoModel.from_pretrained(cfg.lm)
        for layer in self.lm.encoder.layer[:cfg.frozen_text_layers]:
            for p in layer.parameters():
                p.requires_grad = False

        self.downconv = nn.Conv1d(cfg.in_channels, cfg.proj_out, kernel_size=1)
        self.att_pool_head = AttentionPool2d(
            spacial_dim=cfg.spacial_dim, embed_dim=cfg.proj_out,
            num_heads=4, output_dim=cfg.proj_out,
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(p=cfg.dropout)
        self.dropout2 = nn.Dropout(p=cfg.dropout)
        self.linear1 = nn.Linear(cfg.proj_out, cfg.proj_out, bias=False)
        self.linear2 = nn.Linear(cfg.proj_out, cfg.proj_out, bias=False)
        self.proj_t = nn.Sequential(
            nn.Linear(768, cfg.proj_hidden),
            nn.GELU(),
            nn.Linear(cfg.proj_hidden, cfg.proj_out),
        )

    def forward(self, signal: torch.Tensor, condition: dict):
        out = self.resnet(signal)
        ecg_feat = self.downconv(out)

        proj_ecg, _ = self.att_pool_head(ecg_feat)
        proj_ecg = proj_ecg.flatten(1)

        ecg_pooled = self.avgpool(ecg_feat).flatten(1)
        ecg1 = self.dropout1(self.linear1(ecg_pooled))
        ecg2 = self.dropout2(self.linear2(ecg_pooled))

        text_emb = self.lm(**condition).pooler_output
        proj_text = self.proj_t(text_emb)

        if self.cfg.distributed:
            proj_ecg, proj_text, ecg1, ecg2 = self._gather(proj_ecg, proj_text, ecg1, ecg2)

        cma_loss = self.contrastive_loss(proj_ecg, proj_text)
        uma_loss = self.contrastive_loss(ecg1, ecg2)
        return MerlOutput(loss=cma_loss + uma_loss, out=out)
    
    def get_features(self, signal):
        return self.resnet(signal)

    def _gather(self, *tensors):
        return tuple(torch.cat(torch.distributed.nn.all_gather(t), dim=0) for t in tensors)

    @staticmethod
    def contrastive_loss(x, y, temperature=0.07):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        sim = torch.einsum("i d, j d -> i j", x, y) / temperature
        labels = torch.arange(x.shape[0], device=x.device)
        return F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
