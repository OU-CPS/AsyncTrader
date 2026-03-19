import math
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int):
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.float()
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(1e-6)
    return num / den

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
        T = x.size(1)
        return x + self.pe[:, :T]

@dataclass
class BeliefTransformerConfig:
    obs_dim: int
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    dropout: float = 0.1
    max_seq_len: int = 128
    include_time_delta: bool = True
    include_obs_mask: bool = True
    include_agent_present: bool = True
    include_sensor_health: bool = True
    latent_dim: int = 128
    market_dim: int = 3
    reconstruct_dim: Optional[int] = None

class BeliefTransformer(nn.Module):
    def __init__(self, cfg: BeliefTransformerConfig):
        super().__init__()
        self.cfg = cfg
        reconstruct_dim = cfg.reconstruct_dim or cfg.obs_dim
        self.reconstruct_dim = reconstruct_dim
        extra_dim = 0
        if cfg.include_obs_mask:
            extra_dim += cfg.obs_dim
        if cfg.include_time_delta:
            extra_dim += 1
        if cfg.include_agent_present:
            extra_dim += 1
        if cfg.include_sensor_health:
            extra_dim += 1
        input_dim = cfg.obs_dim + extra_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, cfg.model_dim),
            nn.LayerNorm(cfg.model_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=cfg.model_dim,
            max_len=cfg.max_seq_len,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
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
        self.market_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.model_dim),
            nn.GELU(),
            nn.Linear(cfg.model_dim, cfg.market_dim),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.model_dim),
            nn.GELU(),
            nn.Linear(cfg.model_dim, reconstruct_dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.model_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.model_dim // 2, cfg.market_dim),
            nn.Softplus(),
        )
        self.presence_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.model_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.model_dim // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _build_input(
        self,
        obs_seq: torch.Tensor,
        obs_feature_mask: Optional[torch.Tensor] = None,
        time_delta_seq: Optional[torch.Tensor] = None,
        agent_present_seq: Optional[torch.Tensor] = None,
        sensor_health_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [obs_seq]
        B, T, D = obs_seq.shape
        device = obs_seq.device
        dtype = obs_seq.dtype
        if self.cfg.include_obs_mask:
            if obs_feature_mask is None:
                obs_feature_mask = torch.ones(B, T, D, device=device, dtype=dtype)
            parts.append(obs_feature_mask)
        if self.cfg.include_time_delta:
            if time_delta_seq is None:
                time_delta_seq = torch.zeros(B, T, 1, device=device, dtype=dtype)
            parts.append(time_delta_seq)
        if self.cfg.include_agent_present:
            if agent_present_seq is None:
                agent_present_seq = torch.ones(B, T, 1, device=device, dtype=dtype)
            parts.append(agent_present_seq)
        if self.cfg.include_sensor_health:
            if sensor_health_seq is None:
                sensor_health_seq = torch.ones(B, T, 1, device=device, dtype=dtype)
            parts.append(sensor_health_seq)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        obs_seq: torch.Tensor,
        obs_feature_mask: Optional[torch.Tensor] = None,
        timestep_mask: Optional[torch.Tensor] = None,
        time_delta_seq: Optional[torch.Tensor] = None,
        agent_present_seq: Optional[torch.Tensor] = None,
        sensor_health_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = obs_seq.shape
        x = self._build_input(
            obs_seq=obs_seq,
            obs_feature_mask=obs_feature_mask,
            time_delta_seq=time_delta_seq,
            agent_present_seq=agent_present_seq,
            sensor_health_seq=sensor_health_seq,
        )
        x = self.input_proj(x)
        x = self.pos_enc(x)
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        if timestep_mask is None:
            timestep_mask = torch.ones(B, T, device=obs_seq.device, dtype=torch.bool)
        else:
            timestep_mask = timestep_mask.bool()
        cls_valid = torch.ones(B, 1, device=obs_seq.device, dtype=torch.bool)
        full_valid_mask = torch.cat([cls_valid, timestep_mask], dim=1)
        src_key_padding_mask = ~full_valid_mask
        h = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )
        cls_out = h[:, 0]
        token_out = h[:, 1:]
        belief = self.belief_head(cls_out)
        market_pred = self.market_head(belief)
        reconstruction = self.reconstruction_head(belief)
        market_uncertainty = self.uncertainty_head(belief)
        presence_logit = self.presence_head(belief)
        return {
            "belief": belief,
            "token_embeddings": token_out,
            "attn_mask_used": timestep_mask.float(),
            "market_pred": market_pred,
            "reconstruction": reconstruction,
            "market_uncertainty": market_uncertainty,
            "presence_logit": presence_logit,
        }

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, latent_dim=128, market_dim=3, hidden_dim=128, timesteps=50):
        super().__init__()
        self.timesteps = timesteps
        self.market_dim = market_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.x_proj = nn.Sequential(
            nn.Linear(market_dim, hidden_dim),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, market_dim),
        )

    def forward(self, noisy_market: torch.Tensor, t: torch.Tensor, latent_belief: torch.Tensor):
        t = t.float().unsqueeze(-1) / float(self.timesteps)
        te = self.time_embed(t)
        ce = self.context_proj(latent_belief)
        xe = self.x_proj(noisy_market)
        h = torch.cat([xe, te, ce], dim=-1)
        return self.net(h)

    @torch.no_grad()
    def imagine_scenarios(self, latent_belief: torch.Tensor, num_scenarios: int = 8):
        B = latent_belief.size(0)
        device = latent_belief.device
        c = latent_belief.repeat_interleave(num_scenarios, dim=0)
        x = torch.randn(B * num_scenarios, self.market_dim, device=device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((B * num_scenarios,), t, device=device, dtype=torch.long)
            eps_hat = self.forward(x, t_tensor, c)
            alpha = 1.0 - (1.0 / self.timesteps)
            x = (x - (1.0 - alpha) * eps_hat) / math.sqrt(alpha)
            if t > 0:
                x = x + 0.01 * torch.randn_like(x)
        return x.view(B, num_scenarios, self.market_dim)

class RiskAwareTradingAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        market_dim: int = 3,
        use_diffusion: bool = False,
    ):
        super().__init__()
        cfg = BeliefTransformerConfig(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            market_dim=market_dim,
            reconstruct_dim=obs_dim,
        )
        self.transformer = BeliefTransformer(cfg)
        self.use_diffusion = use_diffusion
        if use_diffusion:
            self.imagination_engine = ConditionalDiffusionModel(
                latent_dim=latent_dim,
                market_dim=market_dim,
                hidden_dim=128,
                timesteps=30,
            )
            uncertainty_dim = market_dim
        else:
            self.imagination_engine = None
            uncertainty_dim = market_dim
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim + uncertainty_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        obs_feature_mask: Optional[torch.Tensor] = None,
        timestep_mask: Optional[torch.Tensor] = None,
        time_delta_seq: Optional[torch.Tensor] = None,
        agent_present_seq: Optional[torch.Tensor] = None,
        sensor_health_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bt_out = self.transformer(
            obs_seq=obs_seq,
            obs_feature_mask=obs_feature_mask,
            timestep_mask=timestep_mask,
            time_delta_seq=time_delta_seq,
            agent_present_seq=agent_present_seq,
            sensor_health_seq=sensor_health_seq,
        )
        belief = bt_out["belief"]
        if self.use_diffusion:
            imagined = self.imagination_engine.imagine_scenarios(
                latent_belief=belief,
                num_scenarios=8,
            )
            uncertainty = imagined.std(dim=1)
        else:
            uncertainty = bt_out["market_uncertainty"]
        policy_in = torch.cat([belief, uncertainty], dim=-1)
        action = self.policy_head(policy_in)
        return {
            "action": action,
            "belief": belief,
            "market_pred": bt_out["market_pred"],
            "market_uncertainty": uncertainty,
            "reconstruction": bt_out["reconstruction"],
            "presence_logit": bt_out["presence_logit"],
            "token_embeddings": bt_out["token_embeddings"],
        }
