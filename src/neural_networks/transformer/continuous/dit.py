from dataclasses import dataclass
from typing import Optional, Literal
import math
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    input_dim: int = 12
    d_model: int = 512
    n_heads: int = 8
    dim_ff: int = 2048
    num_layers: int = 12
    dropout: float = 0.1
    max_seq_len: int = 2500
    loss_type: Literal["ddpm", "rectified_flow"] = "rectified_flow"
    num_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    condition: Optional[Literal["label", "text", "lead"]] = None
    num_label_classes: int = 2
    text_max_len: int = 64
    condition_dropout: float = 0.1
    text_feature_extractor: str = None



@dataclass
class DiTOutput:
    loss: Optional[torch.Tensor]
    prediction: torch.Tensor


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(num_classes, d_model)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.emb(labels)


class LeadEmbedder(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, lead: torch.Tensor) -> torch.Tensor:
        x = self.proj(lead.unsqueeze(-1))
        pos = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos_emb(pos)
        return x.mean(dim=1)


class TextEmbedder(nn.Module):
    def __init__(self, text_feature_extractor: str, d_model: int):
        super().__init__()
        self.text_feature_extractor = text_feature_extractor
        self.text_feature_extractor.requires_grad_(False)
        self.text_feature_extractor.eval()
        self.proj = nn.Linear(self.text_feature_extractor.config.hidden_size, d_model)

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_feature_extractor.eval()
        return self
    
    def forward(self, condition) -> torch.Tensor:
        with torch.no_grad():
            out = self.text_feature_extractor(**condition)
        # return self.proj(out.last_hidden_state[:, -1, :])
        return self.proj(out.last_hidden_state.mean(dim=1)) # two ways but i like means better..

class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        
        y = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        y, _ = self.self_attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * y
        
        y = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.ff(y)
        
        return x


class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.SiLU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )
        self.layers = nn.ModuleList([DiTBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_layers)])
        
        self.final_norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
        )
        self.output_proj = nn.Linear(cfg.d_model, cfg.input_dim)

        if cfg.condition == "label":
            self.condition_embedder = LabelEmbedder(cfg.num_label_classes, cfg.d_model)
        elif cfg.condition == "text":
            text_feature_extractor = AutoModel.from_pretrained(cfg.text_feature_extractor)
            self.condition_embedder = TextEmbedder(text_feature_extractor, cfg.d_model)
        elif cfg.condition == "lead":
            self.condition_embedder = LeadEmbedder(cfg.d_model, cfg.max_seq_len)

        if cfg.condition is not None:
            self.null_cond_emb = nn.Parameter(torch.zeros(cfg.d_model))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        if self.cfg.loss_type == "ddpm":
            betas = torch.linspace(self.cfg.beta_start, self.cfg.beta_end, self.cfg.num_steps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _encode_condition(self, condition: Optional[torch.Tensor], bsz: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.cfg.condition is None or condition is None:
            return None
        condition_emb = self.condition_embedder(condition)
        if self.training:
            drop_mask = torch.rand(bsz, device=device) < self.cfg.condition_dropout
            condition_emb = torch.where(drop_mask.unsqueeze(-1), self.null_cond_emb.expand(bsz, -1), condition_emb)
        return condition_emb

    def _forward_net(self, xt: torch.Tensor, t: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, 
                     condition_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = xt.shape
        cond = self.time_mlp(timestep_embedding(t, self.cfg.d_model))
        if condition_emb is not None:
            cond = cond + condition_emb
        pos = torch.arange(seq_len, device=xt.device).unsqueeze(0).expand(bsz, -1)
        h = self.input_proj(xt) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h, cond, key_padding_mask=padding_mask)
        
        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return self.output_proj(h)
    
    def _forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, padding_mask: Optional[torch.Tensor], 
                          condition_emb: Optional[torch.Tensor], cfg_scale: float) -> torch.Tensor:
        if condition_emb is not None and cfg_scale != 1.0:
            pred_cond = self._forward_net(x, t, padding_mask, condition_emb)
            pred_uncond = self._forward_net(x, t, padding_mask, self.null_cond_emb.expand(x.shape[0], -1))
            return pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        return self._forward_net(x, t, padding_mask, condition_emb)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            return (F.mse_loss(pred, target, reduction="none") * mask).sum() / mask.sum()
        return F.mse_loss(pred, target)

    def forward(self, signal: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                 condition: Optional[torch.Tensor] = None) -> DiTOutput:
        x1 = signal.transpose(1, 2)
        bsz = x1.shape[0]
        condition_emb = self._encode_condition(condition, bsz, x1.device)
        if self.cfg.loss_type == "rectified_flow":
            x0 = torch.randn_like(x1)
            t = torch.rand(bsz, device=x1.device)
            t_expanded = t.view(bsz, 1, 1)
            xt = (1 - t_expanded) * x0 + t_expanded * x1
            pred = self._forward_net(xt, t, padding_mask, condition_emb)
            target = x1 - x0
        elif self.cfg.loss_type == "ddpm":
            t_int = torch.randint(0, self.cfg.num_steps, (bsz,), device=x1.device)
            t = t_int.float() / self.cfg.num_steps
            noise = torch.randn_like(x1)
            sqrt_alpha = self.alphas_cumprod[t_int].sqrt().view(bsz, 1, 1)
            sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t_int]).sqrt().view(bsz, 1, 1)
            xt = sqrt_alpha * x1 + sqrt_one_minus_alpha * noise
            pred = self._forward_net(xt, t, padding_mask, condition_emb)
            target = noise

        loss = self._compute_loss(pred, target, padding_mask)
        return DiTOutput(loss=loss, prediction=pred.transpose(1, 2))
    
    @torch.inference_mode()
    def reconstruct(self, signal: torch.Tensor, t_value: float = 0.5, padding_mask: Optional[torch.Tensor] = None,
                    condition: Optional[torch.Tensor] = None, cfg_scale: float = 1.0) -> dict[str, torch.Tensor]:
        x1 = signal.transpose(1, 2)  # (B, C, L) -> (B, L, C)
        bsz = x1.shape[0]
        condition_emb = self._encode_condition(condition, bsz, x1.device)

        if self.cfg.loss_type == "rectified_flow":
            x0 = torch.randn_like(x1)
            t = torch.full((bsz,), t_value, device=x1.device)
            t_exp = t.view(bsz, 1, 1)
            xt = (1 - t_exp) * x0 + t_exp * x1
            pred = self._forward_with_cfg(xt, t, padding_mask, condition_emb, cfg_scale)
            x1_pred = xt + (1 - t_exp) * pred
        elif self.cfg.loss_type == "ddpm":
            t_int = int(t_value * self.cfg.num_steps)
            t_int = max(0, min(t_int, self.cfg.num_steps - 1))
            t_idx = torch.full((bsz,), t_int, device=x1.device, dtype=torch.long)
            t = t_idx.float() / self.cfg.num_steps
            noise = torch.randn_like(x1)
            sqrt_alpha = self.alphas_cumprod[t_idx].sqrt().view(bsz, 1, 1)
            sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t_idx]).sqrt().view(bsz, 1, 1)
            xt = sqrt_alpha * x1 + sqrt_one_minus_alpha * noise
            pred = self._forward_with_cfg(xt, t, padding_mask, condition_emb, cfg_scale)
            x1_pred = (xt - sqrt_one_minus_alpha * pred) / sqrt_alpha

        return {
            "x1_pred": x1_pred.transpose(1, 2),
            "xt": xt.transpose(1, 2),
            "t": t_value,
        }

    @torch.inference_mode()
    def sample(self, shape: tuple, device: torch.device, num_steps: Optional[int] = None, padding_mask: Optional[torch.Tensor] = None,
               condition: Optional[torch.Tensor] = None, cfg_scale: float = 1.0) -> torch.Tensor:
        num_steps = num_steps or self.cfg.num_steps
        bsz, c, seq_len = shape
        x = torch.randn(bsz, seq_len, c, device=device)
        condition_emb = self._encode_condition(condition, bsz, device)

        if self.cfg.loss_type == "rectified_flow":
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((bsz,), (i + 0.5) / num_steps, device=device)
                v = self._forward_with_cfg(x, t, padding_mask, condition_emb, cfg_scale)
                x = x + v * dt
        elif self.cfg.loss_type == "ddpm":
            if num_steps != self.cfg.num_steps:
                raise ValueError("DDPM requires num_steps equal to training steps")
            for i in reversed(range(num_steps)):
                t = torch.full((bsz,), i / self.cfg.num_steps, device=device)
                pred_noise = self._forward_with_cfg(x, t, padding_mask, condition_emb, cfg_scale)

                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                alpha_cumprod_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0, device=device)

                x0_pred = (x - (1 - alpha_cumprod).sqrt() * pred_noise) / alpha_cumprod.sqrt()
                x0_pred = x0_pred.clamp(0, 1)

                mean = (alpha_cumprod_prev.sqrt() * self.betas[i] * x0_pred + alpha.sqrt() * (1 - alpha_cumprod_prev) * x) / (1 - alpha_cumprod)

                if i > 0:
                    variance = self.betas[i] * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
                    x = mean + variance.sqrt() * torch.randn_like(x)
                else:
                    x = mean

        return x.transpose(1, 2)

    def get_features(self, signal: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = signal.transpose(1, 2)
        bsz, seq_len, _ = x.shape
        t = torch.zeros(bsz, device=x.device)
        cond = self.time_mlp(timestep_embedding(t, self.cfg.d_model))
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.input_proj(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h, cond, key_padding_mask=padding_mask)
        h = self.final_norm(h)
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            return (h * mask).sum(1) / mask.sum(1)
        return h.mean(dim=1)