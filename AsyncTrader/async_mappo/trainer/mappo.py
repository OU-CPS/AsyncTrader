import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Categorical
import math

from Environment.macro_action import (
    NUM_MACRO_KINDS,
    NUM_DURATIONS,
    NUM_INTENSITIES,
    DURATION_BINS,
    get_valid_duration_mask,
    get_valid_intensity_mask,
)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for Async MAPPO")
SEED = 42
set_global_seed(SEED)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MacroActor(nn.Module):
    """
    Recurrent macro actor:
    1) encode current decision state
    2) predict macro kind
    3) condition duration/intensity on chosen kind
    """
    def __init__(
        self,
        obs_dim,
        num_kinds=NUM_MACRO_KINDS,
        num_durs=NUM_DURATIONS,
        num_ints=NUM_INTENSITIES,
        hidden_dim=128,
        time_emb_dim=32,
    ):
        super().__init__()
        self.num_kinds = num_kinds
        self.num_durs = num_durs
        self.num_ints = num_ints
        self.hidden_dim = hidden_dim

        self.encoder = AgentCentricEncoder(
            obs_dim,
            time_emb_dim=time_emb_dim,
            hidden_dim=hidden_dim,
        )

        self.kind_head = nn.Linear(hidden_dim, num_kinds)

        self.dur_head = nn.Sequential(
            nn.Linear(hidden_dim + num_kinds, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_durs),
        )

        self.int_head = nn.Sequential(
            nn.Linear(hidden_dim + num_kinds, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ints),
        )

    @staticmethod
    def _mask_logits(logits, mask):
        if mask is None:
            return logits
        mask = mask.bool()
        assert mask.any(dim=-1).all()
        return logits.masked_fill(~mask, -1e9)

    def encode(self, obs, step_t, h_prev):
        return self.encoder(obs, step_t, h_prev)

    def get_kind_logits(self, h, valid_kind_mask=None):
        logits_kind = self.kind_head(h)
        return self._mask_logits(logits_kind, valid_kind_mask)

    def get_conditional_logits(
        self,
        h,
        chosen_kind,
        valid_dur_mask=None,
        valid_int_mask=None,
    ):
        kind_oh = F.one_hot(chosen_kind, num_classes=self.num_kinds).float()
        z = torch.cat([h, kind_oh], dim=-1)

        logits_dur = self.dur_head(z)
        logits_int = self.int_head(z)

        logits_dur = self._mask_logits(logits_dur, valid_dur_mask)
        logits_int = self._mask_logits(logits_int, valid_int_mask)

        return logits_dur, logits_int

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] integer or float timestep
        t = t.float().unsqueeze(-1)  # [B, 1]
        device = t.device
        half_dim = self.emb_dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=device).float() *
            (-math.log(10000.0) / max(half_dim - 1, 1))
        )  # [half_dim]
        angles = t * freq.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.emb_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # [B, emb_dim]


class AgentCentricEncoder(nn.Module):
    def __init__(self, obs_dim: int, time_emb_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.pre_mlp = nn.Sequential(
            nn.Linear(obs_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, obs: torch.Tensor, step_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(step_t)
        x = torch.cat([obs, t_emb], dim=-1)
        x = self.pre_mlp(x)
        h_new = self.gru(x, h_prev)
        return h_new


class ACACCritic(nn.Module):
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        hidden_dim: int = 128,
        time_emb_dim: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.encoder = AgentCentricEncoder(obs_dim, time_emb_dim=time_emb_dim, hidden_dim=hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def init_hidden_bank(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.n_agents, self.hidden_dim, device=device)

    def forward_step(
        self,
        local_obs: torch.Tensor,        # [B, obs_dim]
        step_t: torch.Tensor,           # [B]
        hidden_bank: torch.Tensor,      # [B, N, H]
        agent_id: torch.Tensor,         # [B]
    ):
        B = local_obs.shape[0]
        idx = torch.arange(B, device=local_obs.device)

        h_prev_i = hidden_bank[idx, agent_id]              # [B, H]
        h_new_i = self.encoder(local_obs, step_t, h_prev_i)  # [B, H]

        hidden_bank_new = hidden_bank.clone()
        hidden_bank_new[idx, agent_id] = h_new_i

        attn_out, _ = self.attn(hidden_bank_new, hidden_bank_new, hidden_bank_new)  # [B, N, H]
        pooled = attn_out.mean(dim=1)  # [B, H]
        value = self.value_head(pooled).squeeze(-1)  # [B]

        return value, hidden_bank_new
    

class AsynchMAPPO:
    def __init__(
        self,
        n_agents,
        actor_obs_dim,   
        num_kinds=NUM_MACRO_KINDS,
        num_durs=NUM_DURATIONS,
        num_ints=NUM_INTENSITIES,
        lr=3e-4,
        clip_eps=0.2,
        k_epochs=5,
        batch_size=512,
        ent_coef=0.01,
        max_grad_norm=0.5
    ):
        self.n_agents = n_agents
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.duration_bins = torch.tensor(DURATION_BINS, dtype=torch.float32, device=device)

        self.actor_prosumer = MacroActor(
            actor_obs_dim,
            num_kinds=num_kinds,
            num_durs=num_durs,
            num_ints=num_ints,
            hidden_dim=128,
        ).to(device)

        self.actor_consumer = MacroActor(
            actor_obs_dim,
            num_kinds=num_kinds,
            num_durs=num_durs,
            num_ints=num_ints,
            hidden_dim=128,
        ).to(device)
        
        self.critic = ACACCritic(
            n_agents=n_agents,
            obs_dim=actor_obs_dim,
            hidden_dim=128,
            time_emb_dim=32,
            num_heads=4,
        ).to(device)

        self.opt_a_prosumer = torch.optim.Adam(self.actor_prosumer.parameters(), lr=lr)
        self.opt_a_consumer = torch.optim.Adam(self.actor_consumer.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.initial_lr = lr
        for opt in [
            self.opt_a_prosumer, self.opt_a_consumer, self.opt_c
        ]:
            for pg in opt.param_groups:
                pg["initial_lr"] = lr


    def lr_decay(self, episode, total_episodes):
        frac = 1.0 - (episode - 1.0) / total_episodes
        frac = max(frac, 0.0)
        new_lr = self.initial_lr * frac
        for opt in [self.opt_a_prosumer, self.opt_a_consumer, self.opt_c]:
            for pg in opt.param_groups:
                pg["lr"] = new_lr


    @torch.no_grad()
    def select_action(
        self,
        actor_obs,
        valid_kind_mask,
        valid_dur_mask,
        valid_int_mask,
        agent_type,
        agent_id,
        step_t,
        critic_hidden_bank,
        actor_hidden_prev,
        deterministic=False,
        return_masks=False,
    ):
        a_obs = torch.FloatTensor(actor_obs).unsqueeze(0).to(device)
        kind_mask = torch.BoolTensor(valid_kind_mask).unsqueeze(0).to(device)
        dur_mask = torch.BoolTensor(valid_dur_mask).unsqueeze(0).to(device)
        int_mask = torch.BoolTensor(valid_int_mask).unsqueeze(0).to(device)

        step_t_t = torch.tensor([step_t], dtype=torch.long, device=device)
        agent_id_t = torch.tensor([agent_id], dtype=torch.long, device=device)
        hidden_bank_t = torch.FloatTensor(critic_hidden_bank).unsqueeze(0).to(device)
        actor_h_prev_t = torch.FloatTensor(actor_hidden_prev).unsqueeze(0).to(device)

        actor = self.actor_prosumer if agent_type == 1 else self.actor_consumer

        h_new = actor.encode(a_obs, step_t_t, actor_h_prev_t)

        logits_kind = actor.get_kind_logits(h_new, kind_mask)
        dist_kind = Categorical(logits=logits_kind)

        if deterministic:
            kind = torch.argmax(logits_kind, dim=-1)
        else:
            kind = dist_kind.sample()
        kind_cond_dur_mask = torch.BoolTensor(
            get_valid_duration_mask(int(kind.item()))
        ).unsqueeze(0).to(device)
        kind_cond_int_mask = torch.BoolTensor(
            get_valid_intensity_mask(int(kind.item()))
        ).unsqueeze(0).to(device)
        if valid_dur_mask is not None:
            kind_cond_dur_mask = kind_cond_dur_mask & dur_mask
        if valid_int_mask is not None:
            kind_cond_int_mask = kind_cond_int_mask & int_mask

        logits_dur, logits_int = actor.get_conditional_logits(
            h_new,
            chosen_kind=kind,
            valid_dur_mask=kind_cond_dur_mask,
            valid_int_mask=kind_cond_int_mask,
        )

        dist_dur = Categorical(logits=logits_dur)
        dist_int = Categorical(logits=logits_int)

        if deterministic:
            dur = torch.argmax(logits_dur, dim=-1)
            intensity = torch.argmax(logits_int, dim=-1)
        else:
            dur = dist_dur.sample()
            intensity = dist_int.sample()

        value, hidden_bank_next = self.critic.forward_step(
            local_obs=a_obs,
            step_t=step_t_t,
            hidden_bank=hidden_bank_t,
            agent_id=agent_id_t,
        )

        logprob = (
            dist_kind.log_prob(kind)
            + dist_dur.log_prob(dur)
            + dist_int.log_prob(intensity)
        )

        output = (
            kind.item(),
            dur.item(),
            intensity.item(),
            logprob.item(),
            value.item(),
            hidden_bank_next.squeeze(0).cpu().numpy(),
            h_new.squeeze(0).cpu().numpy(),
        )
        if return_masks:
            output = output + (
                kind_cond_dur_mask.squeeze(0).cpu().numpy(),
                kind_cond_int_mask.squeeze(0).cpu().numpy(),
            )
        return output

    def update(self, buffer_data):
        a_obs = buffer_data["actor_obs"].float().to(device)
        step_t = buffer_data["step_t"].long().to(device)
        agent_ids = buffer_data["agent_ids"].long().to(device)
        critic_hidden_before = buffer_data["critic_hidden_before"].float().to(device)

        ac_kinds = buffer_data["action_kinds"].long().to(device)
        ac_durs = buffer_data["action_durs"].long().to(device)
        ac_ints = buffer_data["action_intensities"].long().to(device) 
        
        returns = buffer_data["returns"].float().to(device)
        advs = buffer_data["advantages"].float().to(device)
        if advs.numel() > 1:
            advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)
            
        old_logprobs = buffer_data["logprobs"].float().to(device)
        old_values = buffer_data["values"].float().to(device)
        
        valid_kind_masks = buffer_data["valid_masks"].bool().to(device)
        valid_dur_masks = buffer_data["valid_duration_masks"].bool().to(device)
        valid_int_masks = buffer_data["valid_intensity_masks"].bool().to(device)
        
        actor_hidden_before = buffer_data["actor_hidden_before"].float().to(device)
        agent_types = buffer_data["agent_types"].long().to(device)





        dataset = torch.utils.data.TensorDataset(
            a_obs, step_t, agent_ids, critic_hidden_before, actor_hidden_before,
            ac_kinds, ac_durs, ac_ints, returns, advs, old_logprobs, old_values,
            valid_kind_masks, valid_dur_masks, valid_int_masks, agent_types
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.k_epochs):
            for batch in loader:
                (b_a_obs, b_step, b_ids, b_critic_h, b_actor_h,
                 b_kinds, b_durs, b_ints, b_ret, b_adv, b_old_lp, b_old_v,
                 b_k_mask, b_d_mask, b_i_mask, b_type) = batch
                
                # 1. CRITIC UPDATE
                values, _ = self.critic.forward_step(b_a_obs, b_step, b_critic_h, b_ids)
                value_pred_clipped = b_old_v + (values - b_old_v).clamp(-self.clip_eps, self.clip_eps)
                loss_c = 0.5 * torch.max((values - b_ret).pow(2), (value_pred_clipped - b_ret).pow(2)).mean()
                
                self.opt_c.zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_c.step()
                for agent_type_idx, actor_net, opt_net in [(1, self.actor_prosumer, self.opt_a_prosumer), 
                                                           (0, self.actor_consumer, self.opt_a_consumer)]:
                    mask = (b_type == agent_type_idx)
                    if mask.any():
                        self._update_actor(
                            actor_net, opt_net, b_a_obs[mask], b_step[mask], b_actor_h[mask],
                            b_kinds[mask], b_durs[mask], b_ints[mask], b_adv[mask], 
                            b_old_lp[mask], b_k_mask[mask], b_d_mask[mask], b_i_mask[mask]
                        )

    def _update_actor(self, actor, opt_a, a_obs, step_t, actor_h_before, kinds, durs, ints, adv, old_lp, k_mask, d_mask, i_mask):
        h = actor.encode(a_obs, step_t, actor_h_before)
        logits_kind = actor.get_kind_logits(h, k_mask)
        logits_dur, logits_int = actor.get_conditional_logits(h, kinds, d_mask, i_mask)

        dist_kind, dist_dur, dist_int = Categorical(logits=logits_kind), Categorical(logits=logits_dur), Categorical(logits=logits_int)

        new_lp = dist_kind.log_prob(kinds) + dist_dur.log_prob(durs) + dist_int.log_prob(ints)
        entropy = dist_kind.entropy().mean() + 0.5 * dist_dur.entropy().mean() + 0.5 * dist_int.entropy().mean()

        ratio = torch.exp(new_lp - old_lp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        loss_a = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

        opt_a.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
        opt_a.step()

        
    @torch.no_grad()
    def get_values(self, actor_obs, agent_id, step_t, critic_hidden_bank):
        a_obs = torch.FloatTensor(actor_obs).unsqueeze(0).to(device)
        step_t_t = torch.tensor([step_t], dtype=torch.long, device=device)
        agent_id_t = torch.tensor([agent_id], dtype=torch.long, device=device)
        hidden_bank_t = torch.FloatTensor(critic_hidden_bank).unsqueeze(0).to(device)
        value, _ = self.critic.forward_step(
            local_obs=a_obs,
            step_t=step_t_t,
            hidden_bank=hidden_bank_t,
            agent_id=agent_id_t,
        )
        return value.item()
    
        
    def save(self, path):
        torch.save({
            "actor_prosumer": self.actor_prosumer.state_dict(),
            "actor_consumer": self.actor_consumer.state_dict(),
            "critic": self.critic.state_dict(), 
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=device)
        self.actor_prosumer.load_state_dict(data["actor_prosumer"])
        self.actor_consumer.load_state_dict(data["actor_consumer"])
        self.critic.load_state_dict(data["critic"])
