import math
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


@dataclass
class MarketBeliefConfig:
    input_dim: int = 3
    model_dim: int = 128
    nhead: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    dropout: float = 0.1
    max_seq_len: int = 128
    pred_horizon: int = 8
    output_dim: int = 3
    latent_dim: int = 128
    use_feature_mask: bool = True
    use_time_delta: bool = True
    use_presence_flag: bool = True


class MarketBeliefTransformer(nn.Module):
    def __init__(self, cfg: MarketBeliefConfig):
        super().__init__()
        self.cfg = cfg

        extra_dim = 0
        if cfg.use_feature_mask:
            extra_dim += cfg.input_dim
        if cfg.use_time_delta:
            extra_dim += 1
        if cfg.use_presence_flag:
            extra_dim += 1

        total_in = cfg.input_dim + extra_dim

        self.input_proj = nn.Sequential(
            nn.Linear(total_in, cfg.model_dim),
            nn.LayerNorm(cfg.model_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=cfg.model_dim,
            max_len=cfg.max_seq_len + 1,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.model_dim))

        self.belief_head = nn.Sequential(
            nn.Linear(cfg.model_dim, cfg.model_dim),
            nn.GELU(),
            nn.Linear(cfg.model_dim, cfg.latent_dim),
        )

        self.future_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.model_dim),
            nn.GELU(),
            nn.Linear(cfg.model_dim, cfg.pred_horizon * cfg.output_dim),
        )

        self.recon_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.model_dim),
            nn.GELU(),
            nn.Linear(cfg.model_dim, cfg.output_dim),
        )

        self.presence_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.model_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.model_dim // 2, 1),
        )

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _build_input(
        self,
        x: torch.Tensor,
        feature_mask: Optional[torch.Tensor],
        time_delta: Optional[torch.Tensor],
        presence_flag: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts = [x]
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        if self.cfg.use_feature_mask:
            if feature_mask is None:
                feature_mask = torch.ones(B, T, D, device=device, dtype=dtype)
            parts.append(feature_mask)

        if self.cfg.use_time_delta:
            if time_delta is None:
                time_delta = torch.zeros(B, T, 1, device=device, dtype=dtype)
            parts.append(time_delta)

        if self.cfg.use_presence_flag:
            if presence_flag is None:
                presence_flag = torch.ones(B, T, 1, device=device, dtype=dtype)
            parts.append(presence_flag)

        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None,
        presence_flag: Optional[torch.Tensor] = None,
        timestep_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape

        h = self._build_input(x, feature_mask, time_delta, presence_flag)
        h = self.input_proj(h)

        cls = self.cls_token.expand(B, 1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)

        if timestep_mask is None:
            timestep_mask = torch.ones(B, T, device=x.device, dtype=torch.bool)
        else:
            timestep_mask = timestep_mask.bool()

        cls_valid = torch.ones(B, 1, device=x.device, dtype=torch.bool)
        full_valid = torch.cat([cls_valid, timestep_mask], dim=1)
        src_key_padding_mask = ~full_valid

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        cls_out = h[:, 0]
        belief = self.belief_head(cls_out)

        pred_future = self.future_head(belief).view(
            B, self.cfg.pred_horizon, self.cfg.output_dim
        )
        recon_current = self.recon_head(belief)
        presence_logit = self.presence_head(belief)

        return {
            "belief": belief,
            "pred_future": pred_future,
            "recon_current": recon_current,
            "presence_logit": presence_logit,
        }
