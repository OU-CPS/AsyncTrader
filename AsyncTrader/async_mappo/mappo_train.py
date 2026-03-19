import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.reward_plots_logs import (
    save_moving_average_plots,
    save_training_performance_csv,
)

from Environment.solar_sys_environment import SolarSys
from Environment.macro_rollout_buffer import MacroRolloutBuffer
from Environment.smart_contract_aync_wrapper import AsyncMacroWrapper
from Pearl.async_mappo.trainer.mappo import AsynchMAPPO
from Pearl.belief_module import MarketBeliefTransformer, MarketBeliefConfig
from Environment.macro_action import (
    NUM_MACRO_KINDS,
    NUM_DURATIONS,
    NUM_INTENSITIES,
    DURATION_BINS,
    MacroAction,
)


def _init_belief_module(device: torch.device):
    cfg = MarketBeliefConfig(
        input_dim=3,
        latent_dim=128,
        max_seq_len=8,
        pred_horizon=8,
        output_dim=3,
    )
    model = MarketBeliefTransformer(cfg).to(device)

    candidate_paths = [
        os.path.join(PROJECT_ROOT, "Pearl", "belief_training_runs", "best_belief_model.pth"),
        os.path.join(PROJECT_ROOT, "Pearl", "belief_training_runs", "final_belief_model.pth"),
        os.path.join(PROJECT_ROOT, "Pearl", "best_belief_module.pth"),
    ]

    loaded_path = None
    loaded_mode = None
    for ckpt_path in candidate_paths:
        if not os.path.exists(ckpt_path):
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state_dict, strict=True)
            loaded_path = ckpt_path
            loaded_mode = "strict"
            break
        except RuntimeError as exc:
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                loaded_path = ckpt_path
                loaded_mode = f"non-strict (missing={len(missing)}, unexpected={len(unexpected)})"
                break
            except Exception as fallback_exc:
                print(f"[WARN] Failed to load belief checkpoint '{ckpt_path}': {exc} | fallback: {fallback_exc}")
        except Exception as exc:
            print(f"[WARN] Failed to load belief checkpoint '{ckpt_path}': {exc}")

    if loaded_path is None:
        print("[WARN] No compatible belief checkpoint found.")
    else:
        print(f"Belief module loaded from: {loaded_path} ({loaded_mode})")

    model.eval()
    return model, cfg, loaded_path is not None


def _build_market_state(env: SolarSys, step_fallback: int):
    step = min(env.current_step, env.num_steps - 1)
    if step < 0:
        step = min(step_fallback, env.num_steps - 1)

    total_demand = sum(float(env.demands[hid][step]) for hid in env.house_ids)
    total_solar = sum(float(env.solars[hid][step]) for hid in env.house_ids)

    ts = env.data.index[step]
    time_norm = (ts.hour * 60 + ts.minute) / (24 * 60)

    demand_den = max(1e-8, env.global_max_demand * max(1, env.num_agents))
    solar_den = max(1e-8, env.global_max_solar * max(1, env.num_agents))
    norm_demand = total_demand / demand_den
    norm_solar = total_solar / solar_den

    return [norm_demand, norm_solar, time_norm]


def main():
    STATE_TO_RUN = "pennsylvania"
    DATA_FILE_PATH = "/Users/ananygupta/Desktop/PeARL_sync/Data/2solar_1nonsolar_2018-02-01_2018-04-07.csv"

    num_episodes = 15000
    batch_size = 256
    checkpoint_interval = 5000
    plot_interval = 100
    window_size = 1000
    REWARD_SCALE = 0.1
    USE_BELIEF = True
    SEED = 42

    DEBUG = True
    DEBUG_PRINT_EVERY = 50

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    base_env = SolarSys(
        data_path=DATA_FILE_PATH,
        state=STATE_TO_RUN,
        time_freq="15min",
    )

    print("Base Observation space:", base_env.observation_space)
    print("Base Action space     :", base_env.action_space)

    env = AsyncMacroWrapper(base_env, gamma=0.99)

    num_agents = base_env.num_agents
    agent_groups = base_env.agent_groups
    max_steps = base_env.num_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if USE_BELIEF:
        belief_model, belief_cfg, belief_loaded = _init_belief_module(device)
        if not belief_loaded:
            print("[WARN] Belief checkpoint unavailable. Disabling belief for this run.")
            USE_BELIEF = False

    if USE_BELIEF:
        seq_len = belief_cfg.max_seq_len
        belief_dim = belief_cfg.latent_dim
    else:
        belief_model = None
        seq_len = 0
        belief_dim = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"async_mappo_macro_{STATE_TO_RUN}_{num_agents}agents_{num_episodes}eps_{timestamp}"
    root_dir = os.path.join("training_data", run_name)
    logs_dir = os.path.join(root_dir, "logs")
    plots_dir = os.path.join(root_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving training outputs to: {root_dir}")

    actor_obs_dim = env.macro_obs_dim + belief_dim

    mappo = AsynchMAPPO(
        n_agents=num_agents,
        actor_obs_dim=actor_obs_dim,
        num_kinds=NUM_MACRO_KINDS,
        num_durs=NUM_DURATIONS,
        num_ints=NUM_INTENSITIES,
        lr=3e-4,
        clip_eps=0.05,
        k_epochs=3,
        batch_size=batch_size,
        ent_coef=0.01,
        max_grad_norm=0.5,
    )

    buffer = MacroRolloutBuffer(num_agents=num_agents, gamma=0.99, gae_lambda=0.95)

    episode_rewards = []
    episode_total_rewards = []
    agent_rewards_log = [[] for _ in range(num_agents)]
    best_mean_reward = -1e9
    best_model_path = os.path.join(logs_dir, "best_model.pth")
    daily_rewards = []

    training_start_time = time.time()
    episode_log_data = []
    performance_metrics_log = []
    agent_charge_log = [[] for _ in range(num_agents)]
    agent_discharge_log = [[] for _ in range(num_agents)]

    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()

        kind_counts = np.zeros(NUM_MACRO_KINDS, dtype=np.int32)
        dur_counts = np.zeros(len(DURATION_BINS), dtype=np.int32)
        decision_events = 0
        ready_count_sum = 0

        if hasattr(mappo, "lr_decay"):
            mappo.lr_decay(episode, num_episodes)
        if hasattr(base_env, "set_training_progress"):
            base_env.set_training_progress(float(episode) / float(max(1, num_episodes)))

        macro_obs_list, step_info = env.reset()

        critic_hidden_bank = np.zeros((num_agents, 128), dtype=np.float32)
        actor_hidden_bank = np.zeros((num_agents, 128), dtype=np.float32)

        market_history = []

        decision_mask = step_info["decision_mask"]
        valid_masks = step_info["valid_actions_mask"]
        valid_dur_masks = step_info["valid_duration_mask"]
        valid_int_masks = step_info["valid_intensity_mask"]

        if DEBUG and episode == 1:
            print(f"[DEBUG] len(macro_obs_list) = {len(macro_obs_list)}")
            print(f"[DEBUG] macro_obs[0] shape = {np.asarray(macro_obs_list[0]).shape}")
            print(f"[DEBUG] decision_mask shape = {np.asarray(decision_mask).shape}")
            print(f"[DEBUG] valid_masks shape = {np.asarray(valid_masks).shape}")

        pending_transitions = {}

        if episode > 1:
            last_episode_metrics = base_env.get_episode_metrics()
            last_episode_metrics["Episode"] = episode - 1
            performance_metrics_log.append(last_episode_metrics)

        total_reward = np.zeros(num_agents, dtype=np.float32)
        done = False
        step_count = 0
        day_logs = []
        macro_rewards = np.zeros(num_agents, dtype=np.float32)

        total_p2p_volume = 0.0
        total_p2p_from_battery = 0.0
        total_p2p_from_solar = 0.0
        total_forced_import = 0.0
        total_grid_import = 0.0
        primitive_sum = np.zeros(6, dtype=np.float64)
        primitive_nonzero = np.zeros(6, dtype=np.int64)
        primitive_steps = 0
        episode_charges = [[] for _ in range(num_agents)]
        episode_discharges = [[] for _ in range(num_agents)]

        belief_vector = np.zeros(belief_dim, dtype=np.float32)

        while not done:
            if USE_BELIEF:
                current_market_state = _build_market_state(base_env, step_count)
                market_history.append(current_market_state)
                if len(market_history) > seq_len:
                    market_history.pop(0)

                padded_history = market_history.copy()
                while len(padded_history) < seq_len:
                    padded_history.insert(0, [0.0, 0.0, 0.0])

                history_tensor = torch.tensor([padded_history], dtype=torch.float32, device=device)
                with torch.no_grad():
                    belief_out = belief_model(history_tensor)
                    belief_vector = belief_out["belief"].squeeze(0).cpu().numpy().astype(np.float32)
                if not np.isfinite(belief_vector).all():
                    raise ValueError("[BELIEF] Non-finite belief vector detected (NaN/Inf).")

            macro_actions_dict = {}
            ready_agents = np.where(decision_mask == 1.0)[0]
            if ready_agents.size > 1:
                np.random.shuffle(ready_agents)
            ready_count_sum += int(ready_agents.size)

            for i in ready_agents:

                augmented_obs = np.concatenate([macro_obs_list[i], belief_vector], axis=0)

                if DEBUG and augmented_obs.shape[0] != actor_obs_dim:
                    raise ValueError(
                        f"[DEBUG] Agent {i} actor obs dim mismatch: "
                        f"got {augmented_obs.shape[0]}, expected {actor_obs_dim}"
                    )

                if i in pending_transitions:
                    pt = pending_transitions[i]
                    real_duration = int(step_info["macro_duration"][i])
                    scaled_macro_reward = float(macro_rewards[i]) * REWARD_SCALE

                    buffer.add(
                        agent_id=i,
                        actor_obs=pt["actor_obs"],
                        critic_hidden_before=pt["critic_hidden_before"],
                        actor_hidden_before=pt["actor_hidden_before"],
                        step_t=pt["step_t"],
                        action_kind=pt["action_kind"],
                        action_dur=pt["action_dur"],
                        action_intensity=pt["action_intensity"],
                        reward=scaled_macro_reward,
                        value=pt["value"],
                        logprob=pt["logprob"],
                        duration=real_duration if real_duration > 0 else pt["actual_duration"],
                        done=False,
                        valid_mask=pt["valid_mask"],
                        valid_intensity_mask=pt["valid_int_mask"],
                        valid_duration_mask=pt["valid_dur_mask"],
                        agent_type=pt["agent_type"],
                    )

                critic_hidden_before = critic_hidden_bank.copy()
                actor_hidden_before = actor_hidden_bank[i].copy()

                (
                    kind_idx,
                    dur_idx,
                    intensity,
                    logprob,
                    value,
                    c_hid,
                    a_hid,
                    used_dur_mask,
                    used_int_mask,
                ) = mappo.select_action(
                    actor_obs=augmented_obs,
                    valid_kind_mask=valid_masks[i],
                    valid_dur_mask=valid_dur_masks[i],
                    valid_int_mask=valid_int_masks[i],
                    agent_type=agent_groups[i],
                    agent_id=i,
                    step_t=step_count,
                    critic_hidden_bank=critic_hidden_bank,
                    actor_hidden_prev=actor_hidden_bank[i],
                    return_masks=True,
                )

                critic_hidden_bank = c_hid
                actor_hidden_bank[i] = a_hid

                kind_counts[kind_idx] += 1
                dur_counts[dur_idx] += 1
                decision_events += 1

                actual_duration_steps = DURATION_BINS[dur_idx]
                macro_actions_dict[i] = MacroAction(
                    kind=kind_idx,
                    duration_steps=actual_duration_steps,
                    steps_remaining=actual_duration_steps,
                    intensity=intensity,
                )

                pending_transitions[i] = {
                    "actor_obs": augmented_obs,
                    "critic_hidden_before": critic_hidden_before,
                    "actor_hidden_before": actor_hidden_before,
                    "step_t": step_count,
                    "action_kind": kind_idx,
                    "action_dur": dur_idx,
                    "action_intensity": intensity,
                    "actual_duration": actual_duration_steps,
                    "value": value,
                    "logprob": logprob,
                    "valid_mask": np.asarray(valid_masks[i]).copy(),
                    "valid_dur_mask": np.asarray(used_dur_mask, dtype=bool).copy(),
                    "valid_int_mask": np.asarray(used_int_mask, dtype=bool).copy(),
                    "agent_type": agent_groups[i],
                }

            macro_obs_list, macro_rewards, terminated, truncated, step_info = env.step(macro_actions_dict)
            total_reward += macro_rewards
            done = bool(terminated or truncated)

            env_info = step_info.get("env_info", {})
            if env_info:
                day_logs.append(env_info)
                total_p2p_volume += float(np.sum(env_info.get("total_peer_bought", 0.0)))
                total_p2p_from_battery += float(np.sum(env_info.get("p2p_from_battery", 0.0)))
                total_p2p_from_solar += float(np.sum(env_info.get("p2p_from_solar", 0.0)))
                total_forced_import += float(np.sum(env_info.get("forced_import", 0.0)))
                total_grid_import += float(np.sum(env_info.get("grid_import_with_p2p", 0.0)))

                primitives = step_info.get("primitives", None)
                if primitives is not None:
                    primitives_arr = np.asarray(primitives, dtype=np.float32)
                    if primitives_arr.shape == (num_agents, 6):
                        primitive_sum += primitives_arr.sum(axis=0)
                        primitive_nonzero += (primitives_arr > 1e-6).sum(axis=0)
                        primitive_steps += num_agents

                charge_arr = env_info.get("charge_amount", None)
                discharge_arr = env_info.get("discharge_amount", None)
                if charge_arr is not None:
                    for i in range(num_agents):
                        episode_charges[i].append(float(charge_arr[i]))
                if discharge_arr is not None:
                    for i in range(num_agents):
                        episode_discharges[i].append(float(discharge_arr[i]))

            decision_mask = step_info["decision_mask"]
            valid_masks = step_info["valid_actions_mask"]
            valid_dur_masks = step_info["valid_duration_mask"]
            valid_int_masks = step_info["valid_intensity_mask"]

            step_count += 1
            if step_count >= max_steps:
                break

        last_values = np.zeros(num_agents, dtype=np.float32)
        for i in range(num_agents):
            final_augmented_obs = np.concatenate([macro_obs_list[i], belief_vector], axis=0)
            last_values[i] = mappo.get_values(
                actor_obs=final_augmented_obs,
                agent_id=i,
                step_t=step_count,
                critic_hidden_bank=critic_hidden_bank,
            )

        for i in range(num_agents):
            if i not in pending_transitions:
                continue

            pt = pending_transitions[i]
            real_duration = int(step_info["macro_duration"][i])
            if real_duration > 0:
                pt["actual_duration"] = real_duration

            scaled_macro_reward = float(macro_rewards[i]) * REWARD_SCALE
            buffer.add(
                agent_id=i,
                actor_obs=pt["actor_obs"],
                critic_hidden_before=pt["critic_hidden_before"],
                actor_hidden_before=pt["actor_hidden_before"],
                step_t=pt["step_t"],
                action_kind=pt["action_kind"],
                action_dur=pt["action_dur"],
                action_intensity=pt["action_intensity"],
                reward=scaled_macro_reward,
                value=pt["value"],
                logprob=pt["logprob"],
                duration=pt["actual_duration"],
                done=done,
                valid_mask=pt["valid_mask"],
                valid_duration_mask=pt["valid_dur_mask"],
                valid_intensity_mask=pt["valid_int_mask"],
                agent_type=pt["agent_type"],
            )

        last_dones = np.array([done] * num_agents, dtype=bool)
        buffer.compute_returns_and_advantages(last_values, last_dones)
        flat_data = buffer.get_flattened_data()

        did_update = 0
        if flat_data["returns"].shape[0] >= batch_size:
            mappo.update(flat_data)
            did_update = 1

        buffer.reset()

        sum_ep_reward = float(np.sum(total_reward))
        mean_ep_reward = float(np.mean(total_reward))
        episode_total_rewards.append(sum_ep_reward)
        episode_rewards.append(mean_ep_reward)
        daily_rewards.append(mean_ep_reward)

        for i in range(num_agents):
            agent_rewards_log[i].append(float(total_reward[i]))

        if len(daily_rewards) % window_size == 0:
            last_means = daily_rewards[-window_size:]
            block_mean = sum(last_means) / window_size
            block_idx = len(daily_rewards) // window_size
            print(f"→ Block {block_idx} | Mean Reward: {block_mean:.3f}")

        baseline_cost = sum(
            float(np.sum(entry["grid_import_no_p2p"])) * base_env.get_grid_price(int(entry["step"]))
            for entry in day_logs
            if "grid_import_no_p2p" in entry and "step" in entry
        )
        actual_cost = sum(float(np.sum(entry["costs"])) for entry in day_logs if "costs" in entry)
        cost_reduction = (baseline_cost - actual_cost) / baseline_cost if abs(baseline_cost) > 1e-6 else 0.0

        baseline_import = sum(float(np.sum(entry["grid_import_no_p2p"])) for entry in day_logs if "grid_import_no_p2p" in entry)
        actual_import = sum(float(np.sum(entry["grid_import_with_p2p"])) for entry in day_logs if "grid_import_with_p2p" in entry)
        grid_reduction = (baseline_import - actual_import) / baseline_import if abs(baseline_import) > 1e-6 else 0.0

        baseline_cost_cf = sum(float(np.sum(entry.get("cost_no_p2p_counterfactual", 0.0))) for entry in day_logs)
        baseline_import_cf = sum(float(np.sum(entry.get("grid_import_no_p2p_counterfactual", 0.0))) for entry in day_logs)
        cost_reduction_cf = (baseline_cost_cf - actual_cost) / baseline_cost_cf if abs(baseline_cost_cf) > 1e-6 else 0.0
        grid_reduction_cf = (baseline_import_cf - actual_import) / baseline_import_cf if abs(baseline_import_cf) > 1e-6 else 0.0
        forced_import_ratio = total_forced_import / max(1e-6, total_grid_import)
        idle_rate = kind_counts[0] / max(1, decision_events)
        battery_p2p_share = total_p2p_from_battery / max(1e-6, total_p2p_from_battery + total_p2p_from_solar)
        reward_w_supply = 0.0
        reward_w_unmatched = 0.0
        if day_logs:
            reward_w_supply = float(np.mean([float(entry.get("reward_w_supply", 0.0)) for entry in day_logs]))
            reward_w_unmatched = float(np.mean([float(entry.get("reward_w_unmatched", 0.0)) for entry in day_logs]))
        primitive_mean = primitive_sum / max(1, primitive_steps)
        primitive_active_rate = primitive_nonzero / max(1, primitive_steps)

        if mean_ep_reward > best_mean_reward:
            best_mean_reward = mean_ep_reward
            mappo.save(best_model_path)

        if episode % checkpoint_interval == 0:
            ckpt_path = os.path.join(logs_dir, f"checkpoint_{episode}.pth")
            mappo.save(ckpt_path)

        episode_duration = time.time() - episode_start_time

        print(
            f"Ep {episode}/{num_episodes} "
            f"| Time: {episode_duration:.1f}s "
            f"| Mean Reward: {mean_ep_reward:.2f} "
            f"| Cost Red: {cost_reduction:.2%} "
            f"| Grid Red: {grid_reduction:.2%} "
            f"| Cost Red CF: {cost_reduction_cf:.2%} "
            f"| Grid Red CF: {grid_reduction_cf:.2%} "
            f"| P2P Vol: {total_p2p_volume:.2f} "
            f"| BattP2P: {battery_p2p_share:.2%} "
            f"| ForcedImp: {forced_import_ratio:.2%} "
            f"| IdleRate: {idle_rate:.2%} "
            f"| wSupply: {reward_w_supply:.3f} "
            f"| wUnmatch: {reward_w_unmatched:.3f} "
            f"| Update: {did_update}"
        )

        if DEBUG and episode % DEBUG_PRINT_EVERY == 0:
            avg_ready = ready_count_sum / max(1, step_count)
            print(f"[DEBUG][Ep {episode}] kind_counts={kind_counts.tolist()}")
            print(f"[DEBUG][Ep {episode}] dur_counts={dur_counts.tolist()}")
            print(f"[DEBUG][Ep {episode}] decision_events={decision_events}, avg_ready_per_step={avg_ready:.2f}")

        episode_log_data.append(
            {
                "Episode": episode,
                "Steps": step_count,
                "Mean_Reward": mean_ep_reward,
                "Total_Reward": sum_ep_reward,
                "Cost_Reduction_Pct": cost_reduction * 100,
                "Grid_Reduction_Pct": grid_reduction * 100,
                "Cost_Reduction_CF_Pct": cost_reduction_cf * 100,
                "Grid_Reduction_CF_Pct": grid_reduction_cf * 100,
                "P2P_Total_Volume": total_p2p_volume,
                "P2P_From_Battery_Share_Pct": battery_p2p_share * 100,
                "Forced_Import_Ratio_Pct": forced_import_ratio * 100,
                "Idle_Rate_Pct": idle_rate * 100,
                "Did_Update": did_update,
                "Reward_w_Supply": reward_w_supply,
                "Reward_w_Unmatched": reward_w_unmatched,
                "Mean_aBuyGrid": primitive_mean[0],
                "Mean_aSellGrid": primitive_mean[1],
                "Mean_aBuyPeers": primitive_mean[2],
                "Mean_aSellPeers": primitive_mean[3],
                "Mean_aChargeBatt": primitive_mean[4],
                "Mean_aDischargeBatt": primitive_mean[5],
                "Active_aBuyGrid_Pct": primitive_active_rate[0] * 100,
                "Active_aSellGrid_Pct": primitive_active_rate[1] * 100,
                "Active_aBuyPeers_Pct": primitive_active_rate[2] * 100,
                "Active_aSellPeers_Pct": primitive_active_rate[3] * 100,
                "Active_aChargeBatt_Pct": primitive_active_rate[4] * 100,
                "Active_aDischargeBatt_Pct": primitive_active_rate[5] * 100,
                "Baseline_Cost": baseline_cost,
                "Baseline_Cost_CF": baseline_cost_cf,
                "Actual_Cost": actual_cost,
                "Episode_Duration": episode_duration,
                "Total_Charge": sum(float(np.sum(entry.get("charge_amount", 0.0))) for entry in day_logs),
                "Total_Discharge": sum(float(np.sum(entry.get("discharge_amount", 0.0))) for entry in day_logs),
            }
        )

        for i in range(num_agents):
            agent_charge_log[i].append(float(np.mean(episode_charges[i])) if episode_charges[i] else 0.0)
            agent_discharge_log[i].append(float(np.mean(episode_discharges[i])) if episode_discharges[i] else 0.0)

        if DEBUG and episode % DEBUG_PRINT_EVERY == 0:
            print(
                f"[DEBUG][Ep {episode}] "
                f"steps={step_count}, "
                f"contracts_in_buffer={flat_data['returns'].shape[0]}, "
                f"mean_reward={mean_ep_reward:.3f}, "
                f"cost_red={cost_reduction:.3%}, "
                f"grid_red={grid_reduction:.3%}, "
                f"belief_norm={float(np.linalg.norm(belief_vector)):.3f}, "
                f"battery_p2p_share={battery_p2p_share:.3%}, "
                f"forced_imp={forced_import_ratio:.3%}, "
                f"idle_rate={idle_rate:.3%}, "
                f"aPeers={primitive_mean[2]:.3f}/{primitive_mean[3]:.3f}, "
                f"aBatt={primitive_mean[4]:.3f}/{primitive_mean[5]:.3f}, "
                f"updated={did_update}"
            )

        if episode % plot_interval == 0 and episode > 1:
            print(f"Updating plots at episode {episode}...")

            temp_df_rewards = pd.DataFrame(episode_log_data)
            temp_df_perf = pd.DataFrame(performance_metrics_log)

            cols_to_drop_temp = [
                c
                for c in [
                    "degradation_cost_over_time",
                    "cost_savings_over_time",
                    "grid_reduction_over_time",
                ]
                if c in temp_df_perf.columns
            ]

            if not temp_df_perf.empty:
                temp_df_final = pd.merge(
                    temp_df_rewards,
                    temp_df_perf.drop(columns=cols_to_drop_temp),
                    on="Episode",
                    how="left",
                )
            else:
                temp_df_final = temp_df_rewards

            current_ma_window = min(window_size, max(1, episode // 5))

            save_moving_average_plots(
                df_final_log=temp_df_final,
                plots_dir=plots_dir,
                num_episodes=episode,
                ma_window=current_ma_window,
            )

            temp_df_final.to_csv(os.path.join(logs_dir, "training_log_intermediate.csv"), index=False)

    final_episode_metrics = base_env.get_episode_metrics()
    final_episode_metrics["Episode"] = num_episodes
    performance_metrics_log.append(final_episode_metrics)

    training_time = time.time() - training_start_time

    np.save(os.path.join(logs_dir, "agent_rewards.npy"), np.array(agent_rewards_log))
    np.save(os.path.join(logs_dir, "mean_rewards.npy"), np.array(episode_rewards))

    df_rewards_log = pd.DataFrame(episode_log_data)
    df_perf_log = pd.DataFrame(performance_metrics_log)

    cols_to_drop = [
        c
        for c in [
            "degradation_cost_over_time",
            "cost_savings_over_time",
            "grid_reduction_over_time",
        ]
        if c in df_perf_log.columns
    ]
    df_final_log = pd.merge(df_rewards_log, df_perf_log.drop(columns=cols_to_drop), on="Episode")

    save_moving_average_plots(
        df_final_log=df_final_log,
        plots_dir=plots_dir,
        num_episodes=num_episodes,
        ma_window=min(window_size, num_episodes // 2),
    )

    log_csv_path = save_training_performance_csv(
        df_final_log=df_final_log,
        logs_dir=logs_dir,
        total_training_time_s=training_time,
    )

    print(f"DONE. Total time: {training_time:.2f}s. Log: {log_csv_path}")


if __name__ == "__main__":
    main()
