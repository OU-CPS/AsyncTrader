import gymnasium as gym
import pandas as pd
import numpy as np
import random
from Environment.battery import BatteryManager
from Environment.grid_pricing import GridPricing
from Environment.rewards import AdaptiveRewardShaper

random.seed(42)
np.random.seed(42)
 
class SolarSys(gym.Env):

    def __init__(
        self,
        data_path="/Users/ananygupta/Desktop/Solar_trading/Data/6solar_4nonsolar_2018-02-01_2018-04-07.csv",
        state="oklahoma",
        time_freq="15T",
        reward_weights=None,
        episode_days=5,
        battery_assignment="all",  # Options: "solar_only" or "all"
        allow_peer_charging=True,
    ):
        super().__init__()
        # Store config
        self.data_path = data_path
        self.time_freq = time_freq
        time_freq = str(time_freq).strip()
        if time_freq.endswith("T"):         
            time_freq = time_freq[:-1] + "min" 
        self.state = state.lower()

        try:
            all_data = pd.read_csv(data_path)
            all_data["local_15min"] = pd.to_datetime(all_data["local_15min"], utc=True)
            all_data.set_index("local_15min", inplace=True)
            energy_cols = [c for c in all_data.columns if c.startswith("grid_") or c.startswith("total_solar_")]
            all_data = all_data[energy_cols].resample(time_freq).sum()
            all_data = all_data.fillna(0.0)

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file {data_path} not found.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Data file {data_path} is empty.")
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

        # Compute global maxima for normalization
        grid_cols = [c for c in all_data.columns if c.startswith("grid_")]
        solar_cols = [c for c in all_data.columns if c.startswith("total_solar_")]
        all_grid = all_data[grid_cols].values
        all_solar = all_data[solar_cols].values

        self.battery_assignment = battery_assignment
        self.allow_peer_charging = allow_peer_charging

        # max total demand = max(grid + solar) over all time & agents
        self.global_max_demand = float((all_grid + all_solar).max()) + 1.0 + 1e-8

        # max solar generation alone
        self.global_max_solar = float(all_solar.max())+ 1.0  + 1e-8

        # Store the resampled dataset
        self.all_data = all_data

        self.time_freq = time_freq
        freq_offset = pd.tseries.frequencies.to_offset(time_freq)
        minutes_per_step = freq_offset.nanos / 1e9 / 60.0
        self.steps_per_day = int(24 * 60 // minutes_per_step)
        self.episode_days = episode_days
        self.steps_per_episode = self.steps_per_day * self.episode_days
        self.pricing = GridPricing(state=self.state, steps_per_day=self.steps_per_day)
        self.max_grid_price = self.pricing.max_grid_price
        self.feed_in_tariff = self.pricing.feed_in_tariff
        self.minutes_per_step = freq_offset.nanos / 1e9 / 60.0
        self.time_delta_hours = self.minutes_per_step / 60.0
        total_rows = len(self.all_data)
        self.total_days = total_rows // self.steps_per_day
        if self.total_days < self.episode_days:
            raise ValueError(
                f"After resampling, dataset only has {self.total_days} days, "
                f"which is less than the requested episode length of {self.episode_days} days."
            )

        self.house_ids = [
            col.split("_")[1] for col in self.all_data.columns
            if col.startswith("grid_")
        ]
        self.num_agents = len(self.house_ids)
        self.original_no_p2p_import = {}
        for hid in self.house_ids:
            col_grid = f"grid_{hid}"
            self.original_no_p2p_import[hid] = self.all_data[col_grid].clip(lower=0.0).values
        
        # Determine population groups
        # group 1 = has any solar; group 0 = never solar
        solar_cols = [f"total_solar_{hid}" for hid in self.house_ids]
        solar_sums = self.all_data[solar_cols].sum(axis=0).to_dict()
        self.agent_groups = [
            1 if solar_sums[f"total_solar_{hid}"] > 0 else 0
            for hid in self.house_ids
        ]

        # Count the number of houses in each group
        self.group_counts = {
            0: self.agent_groups.count(0),
            1: self.agent_groups.count(1)
        }
        print(f"Number of houses in each group: {self.group_counts}")

        # Battery logic
        self.battery_options = {
            "teslapowerwall": {"max_capacity": 13.5, "charge_efficiency": 0.95, "discharge_efficiency": 0.90, "max_charge_rate": 5.0, "max_discharge_rate": 5.0, "degradation_cost_per_kwh": 0.005},
            "enphase":         {"max_capacity": 5.0,  "charge_efficiency": 0.95, "discharge_efficiency": 0.90, "max_charge_rate": 2.0, "max_discharge_rate": 2.0, "degradation_cost_per_kwh": 0.005},
            "franklin":        {"max_capacity": 20.0, "charge_efficiency": 0.95, "discharge_efficiency": 0.92, "max_charge_rate": 8.0, "max_discharge_rate": 8.0, "degradation_cost_per_kwh": 0.003},
        }
        

        # Identify which houses actually have solar
        self.solar_houses = [
            hid for hid in self.house_ids
            if (self.all_data[f"total_solar_{hid}"] > 0).any()
        ]

        # Decide which houses receive batteries
        if self.battery_assignment == "solar_only":
            self.battery_houses = list(self.solar_houses)
        elif self.battery_assignment == "all":
            self.battery_houses = list(self.house_ids)
        else:
            raise ValueError("battery_assignment must be 'solar_only' or 'all'")

        # Battery manager
        self.battery_manager = BatteryManager(
            house_ids=self.house_ids,
            battery_houses=self.battery_houses,
            battery_options=self.battery_options,
            seed=42,
        )

        # Observation & Action Spaces
        # [own_demand, own_solar, grid_price, peer_price,
        #  total_demand_others, total_solar_others, SOC, time_of_day]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_agents, 12),
            dtype=np.float32
        )
        
        # Action Space: 6 discrete continuous actions spanning [0.0, 1.0]
        # [BuyGrid, SellGrid, BuyPeers, SellPeers, ChargeBatt, DischargeBatt]
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_agents, 6),
            dtype=np.float32
        )
        
        self.episode_metrics = {}
        self._initialize_episode_metrics()
        
        # Initialize episode variables
        self.data = None
        self.env_log = []
        self.day_index = -1  
        self.current_step = 0
        self.num_steps = self.steps_per_episode
        self.demands = {}
        self.solars = {}
        self.previous_actions = {
            hid: np.zeros(6) for hid in self.house_ids 
        }
        self.reward_shaper = AdaptiveRewardShaper(base_weights=reward_weights)
        self.training_progress = 0.0
        self._last_reward_debug = None


    def _initialize_episode_metrics(self):
        """Initialize or reset all metrics tracked over a single episode."""
        self.cumulative_grid_reduction = 0.0
        self.cumulative_grid_reduction_peak = 0.0
        self.cumulative_degradation_cost = 0.0
        self.agent_cost_savings = np.zeros(self.num_agents)
        self.degradation_cost_timeseries = []
        self.cost_savings_timeseries = []
        self.grid_reduction_timeseries = []

    def set_training_progress(self, progress: float):
        self.training_progress = float(np.clip(progress, 0.0, 1.0))
        self.reward_shaper.set_progress(self.training_progress)

    def get_grid_price(self, step_idx):
        return self.pricing.grid_price(step_idx)

    def get_peer_price(self, step_idx, total_surplus, total_shortfall):
        grid_price = self.get_grid_price(step_idx)
        feed_in_tariff = self.feed_in_tariff 

        base_price = grid_price * 0.90
        net_demand = total_shortfall - total_surplus
        total_potential_trade = total_shortfall + total_surplus + 1e-6
        elasticity_factor = 0.3
        price_multiplier = np.exp(elasticity_factor * (net_demand / total_potential_trade))
        peer_price = base_price * price_multiplier
        final_price = float(np.clip(peer_price, feed_in_tariff, grid_price))

        return final_price


    def reset(self):
        # Finalize and store metrics from completed episode before resetting
        if self.current_step > 0:
            positive_savings = self.agent_cost_savings[self.agent_cost_savings > 0]
            if len(positive_savings) > 1:
                fairness_on_savings = self._compute_jains_index(positive_savings)
            else:
                fairness_on_savings = 0.0

            # Store all final metrics
            self.episode_metrics = {
                "grid_reduction_entire_day": self.cumulative_grid_reduction,
                "grid_reduction_peak_hours": self.cumulative_grid_reduction_peak,
                "total_cost_savings": np.sum(self.agent_cost_savings),
                "fairness_on_cost_savings": fairness_on_savings,
                "battery_degradation_cost_total": self.cumulative_degradation_cost,
                "degradation_cost_over_time": self.degradation_cost_timeseries,
                "cost_savings_over_time": self.cost_savings_timeseries,
                "grid_reduction_over_time": self.grid_reduction_timeseries,
            }
        

        max_start_day = self.total_days - self.episode_days
        self.day_index = np.random.randint(0, max_start_day + 1)

        start_row = self.day_index * self.steps_per_day
        

        end_row = start_row + self.steps_per_episode 
        
        episode_data = self.all_data.iloc[start_row:end_row].copy()
        self.data = episode_data  
        self.no_p2p_import_episode = {} 
        for hid in self.house_ids:
            self.no_p2p_import_episode[hid] = self.original_no_p2p_import[hid][start_row:end_row]

        self.demands = {}
        self.solars = {}

        for hid in self.house_ids:
            col_grid = f"grid_{hid}"
            col_solar = f"total_solar_{hid}"

            grid_series = episode_data[col_grid].fillna(0.0)
            solar_series = episode_data[col_solar].fillna(0.0).clip(lower=0.0)

            demand_array = grid_series.values + solar_series.values
            demand_array = np.clip(demand_array, 0.0, None)

            self.demands[hid] = demand_array
            self.solars[hid] = solar_series.values

        self.current_step = 0
        self.env_log = []
        self._last_reward_debug = None
        

        for hid in self.house_ids:
            self.previous_actions[hid] = np.zeros(6)
        
        self._initialize_episode_metrics()
        self.battery_manager.reset_soc_uniform(low_frac=0.20, high_frac=0.60)

        obs = self._get_obs()
        obs_list = [obs[i] for i in range(self.num_agents)]
        return obs_list, {}


    def _apply_deadband(self, a, threshold=0.75, power=2.0):

        a = np.asarray(a, dtype=np.float32)
        x = np.clip((a - threshold) / (1.0 - threshold), 0.0, 1.0)
        return x ** power
    
    def step(self, actions):
        # Validate & clamp actions
        actions = np.array(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, 6):
            raise ValueError(
                f"Actions shape mismatch: got {actions.shape}, expected {(self.num_agents, 6)}"
            )

        # Clamp between 0.0 and 1.0
        actions = np.clip(actions, 0.0, 1.0)
  
        # -------------------------------------------------
        effective_actions = np.zeros_like(actions, dtype=np.float32)

        # You can tune these thresholds later
        # effective_actions[:, 0] = actions[:, 0]  # BuyGrid: leave raw for now
        # effective_actions[:, 1] = self._apply_deadband(actions[:, 1], threshold=0.84, power=2.0)  # SellGrid
        # effective_actions[:, 2] = self._apply_deadband(actions[:, 2], threshold=0.70, power=2.0)  # BuyPeers
        # effective_actions[:, 3] = self._apply_deadband(actions[:, 3], threshold=0.70, power=2.0)  # SellPeers
        # effective_actions[:, 4] = self._apply_deadband(actions[:, 4], threshold=0.65, power=2.0)  # ChargeBatt
        # effective_actions[:, 5] = self._apply_deadband(actions[:, 5], threshold=0.75, power=2.0)  # DischargeBatt
        effective_actions = actions.copy() 
        a_buyGrid       = effective_actions[:, 0]
        a_sellGrid      = effective_actions[:, 1]
        a_buyPeers      = effective_actions[:, 2]
        a_sellPeers     = effective_actions[:, 3]
        a_chargeBatt    = effective_actions[:, 4]
        a_dischargeBatt = effective_actions[:, 5]

        # Gather current baseline demand & solar
        demands = []
        solars = []
        for i, hid in enumerate(self.house_ids):
            demands.append(self.demands[hid][self.current_step])
            solars.append(self.solars[hid][self.current_step])

        demands = np.array(demands, dtype=np.float32)
        solars = np.array(solars, dtype=np.float32)

        # -------------------------------------------------
        # 1. SELF-CONSUMPTION FIRST
        # -------------------------------------------------
        raw_surplus = np.maximum(solars - demands, 0.0)
        raw_shortfall = np.maximum(demands - solars, 0.0)

        # Prices based on pre-trade market condition
        total_surplus_raw = float(np.sum(raw_surplus))
        total_shortfall_raw = float(np.sum(raw_shortfall))
        peer_price = self.get_peer_price(self.current_step, total_surplus_raw, total_shortfall_raw)
        grid_price = self.get_grid_price(self.current_step)

        p2p_buy_request = a_buyPeers * raw_shortfall

        # Solar-backed offer
        solar_sell_offer = a_sellPeers * raw_surplus
        battery_sell_offer = np.zeros(self.num_agents, dtype=np.float32)
        for i, hid in enumerate(self.house_ids):
            if not self.battery_manager.has_battery(hid):
                continue
            if raw_shortfall[i] > 1e-9:
                continue

            batt_state = self.battery_manager.batteries[hid]
            max_energy_output = batt_state.spec.max_discharge_rate * self.time_delta_hours
            available_energy = batt_state.soc * batt_state.spec.discharge_efficiency
            desired_offer = float(a_sellPeers[i]) * float(a_dischargeBatt[i]) * max_energy_output
            battery_sell_offer[i] = min(desired_offer, available_energy)

        p2p_sell_offer = solar_sell_offer + battery_sell_offer

        total_sell = np.sum(p2p_sell_offer)
        total_buy = np.sum(p2p_buy_request)
        matched = min(total_sell, total_buy)

        if matched > 1e-9:
            sell_fraction = p2p_sell_offer / (total_sell + 1e-12)
            buy_fraction = p2p_buy_request / (total_buy + 1e-12)
            actual_sold = matched * sell_fraction
            actual_bought = matched * buy_fraction
        else:
            actual_sold = np.zeros(self.num_agents, dtype=np.float32)
            actual_bought = np.zeros(self.num_agents, dtype=np.float32)

        # Split peer sales by source (solar vs battery) and settle battery SOC
        offer_mix = p2p_sell_offer + 1e-12
        solar_share = solar_sell_offer / offer_mix
        batt_share = battery_sell_offer / offer_mix

        from_solar_p2p = actual_sold * solar_share
        from_batt_p2p = actual_sold * batt_share

        # Numerical safety
        from_solar_p2p = np.minimum(from_solar_p2p, solar_sell_offer)
        from_batt_p2p = np.minimum(from_batt_p2p, battery_sell_offer)

        for i, hid in enumerate(self.house_ids):
            if not self.battery_manager.has_battery(hid):
                continue
            batt_state = self.battery_manager.batteries[hid]
            batt_state.soc -= from_batt_p2p[i] / batt_state.spec.discharge_efficiency
            batt_state.soc = max(0.0, batt_state.soc)

        # Remaining after deficit-serving P2P
        remaining_surplus = raw_surplus - from_solar_p2p
        remaining_shortfall = raw_shortfall - actual_bought

        # -------------------------------------------------
        # 3. BATTERY CHARGING + PEER STORAGE TRADING
        # -------------------------------------------------
        batt_accept_limit = self.battery_manager.charge_acceptance_limit(self.time_delta_hours)
        local_charge_request = a_chargeBatt * np.minimum(remaining_surplus, batt_accept_limit)
        local_charge_request = np.minimum(local_charge_request, batt_accept_limit)

        residual_charge_capacity = np.maximum(batt_accept_limit - local_charge_request, 0.0)
        surplus_after_local_reservation = np.maximum(remaining_surplus - local_charge_request, 0.0)
        peer_charge_bought = np.zeros(self.num_agents, dtype=np.float32)
        peer_charge_sold = np.zeros(self.num_agents, dtype=np.float32)

        if self.allow_peer_charging:
            storage_buy_request = a_buyPeers * residual_charge_capacity
            storage_sell_offer = a_sellPeers * surplus_after_local_reservation

            total_storage_buy = float(np.sum(storage_buy_request))
            total_storage_sell = float(np.sum(storage_sell_offer))
            matched_storage = min(total_storage_buy, total_storage_sell)

            if matched_storage > 1e-9:
                buy_fraction_storage = storage_buy_request / (total_storage_buy + 1e-12)
                sell_fraction_storage = storage_sell_offer / (total_storage_sell + 1e-12)

                peer_charge_bought = matched_storage * buy_fraction_storage
                peer_charge_sold = matched_storage * sell_fraction_storage

        settled_charge_supply = local_charge_request + peer_charge_bought

        charge_amount_kwh = self.battery_manager.apply_charge(
            a_charge=np.ones(self.num_agents, dtype=np.float32),
            surplus_kwh=settled_charge_supply,
            time_delta=self.time_delta_hours,
        )
        if not np.allclose(charge_amount_kwh, settled_charge_supply, atol=1e-6):
            raise RuntimeError("Charge settlement mismatch: charged energy differs from settled supply.")

        local_charge_used = local_charge_request
        peer_charge_used = peer_charge_bought

        remaining_surplus = np.maximum(
            remaining_surplus - local_charge_used - peer_charge_sold,
            0.0
        )

        discharge_amount_kwh = self.battery_manager.apply_discharge(
            a_discharge=a_dischargeBatt,
            shortfall_kwh=remaining_shortfall,
            time_delta=self.time_delta_hours,
        )

        remaining_shortfall = np.maximum(
            remaining_shortfall - discharge_amount_kwh,
            0.0
        )

        grid_import = np.zeros(self.num_agents, dtype=np.float32)
        grid_export = np.zeros(self.num_agents, dtype=np.float32)
        grid_import = a_buyGrid * remaining_shortfall
        grid_export = a_sellGrid * remaining_surplus
        forced_import = np.maximum(remaining_shortfall - grid_import, 0.0)
        grid_import += forced_import
        forced_import_penalty = forced_import * grid_price

        # -------------------------------------------------
        # ECONOMICS & REWARDS
        # -------------------------------------------------
        total_peer_bought = actual_bought + peer_charge_used
        total_peer_sold = actual_sold + peer_charge_sold

        costs = (
            (grid_import * grid_price)
            - (grid_export * self.feed_in_tariff)
            + (total_peer_bought * peer_price)
            - (total_peer_sold * peer_price)
        )

        no_p2p_import_this_step = np.array([
            self.no_p2p_import_episode[hid][self.current_step]
            for hid in self.house_ids
        ], dtype=np.float32)

        no_p2p_import_counterfactual = raw_shortfall.copy()
        no_p2p_export_counterfactual = raw_surplus.copy()

        actual_cost_for_reward = costs

        cost_no_p2p = no_p2p_import_this_step * grid_price
        cost_no_p2p_counterfactual = (
            no_p2p_import_counterfactual * grid_price
            - no_p2p_export_counterfactual * self.feed_in_tariff
        )
        step_cost_savings_per_agent = cost_no_p2p - actual_cost_for_reward
        next_cumulative_cost_savings = self.agent_cost_savings + step_cost_savings_per_agent

        final_rewards = self._compute_rewards(
            cost_base=cost_no_p2p,
            cost_actual=actual_cost_for_reward,
            grid_base=no_p2p_import_this_step,
            grid_actual=grid_import,
            forced_import=forced_import,
            shortfall_before_force=remaining_shortfall,
            p2p_buy=total_peer_bought,
            p2p_sell=total_peer_sold,
            p2p_buy_request=p2p_buy_request,
            raw_surplus=raw_surplus,
            all_cost_savings_prev=self.agent_cost_savings,
            all_cost_savings_next=next_cumulative_cost_savings,
        )

        step_grid_reduction = np.sum(no_p2p_import_this_step - grid_import)
        self.cumulative_grid_reduction += step_grid_reduction
        self.grid_reduction_timeseries.append(step_grid_reduction)

        if self.pricing.is_peak(self.current_step):
            self.cumulative_grid_reduction_peak += step_grid_reduction

        self.agent_cost_savings = next_cumulative_cost_savings
        self.cost_savings_timeseries.append(np.sum(step_cost_savings_per_agent))

        step_degradation_cost = self.battery_manager.degradation_cost_step(
            charge_amount=charge_amount_kwh,
            discharge_amount=discharge_amount_kwh,
        )

        self.cumulative_degradation_cost += step_degradation_cost
        self.degradation_cost_timeseries.append(step_degradation_cost)

        info = {
            "p2p_buy": actual_bought,
            "p2p_sell": actual_sold,
            "p2p_from_solar": from_solar_p2p,
            "p2p_from_battery": from_batt_p2p,
            "battery_sell_offer": battery_sell_offer,
            "peer_charge_bought": peer_charge_used,
            "peer_charge_sold": peer_charge_sold,
            "total_peer_bought": total_peer_bought,
            "total_peer_sold": total_peer_sold,
            "grid_import_with_p2p": grid_import,
            "grid_import_no_p2p": no_p2p_import_this_step,
            "grid_import_no_p2p_counterfactual": no_p2p_import_counterfactual,
            "grid_export_no_p2p_counterfactual": no_p2p_export_counterfactual,
            "grid_export": grid_export,
            "forced_import": forced_import, 
            "costs": costs,
            "cost_no_p2p_counterfactual": cost_no_p2p_counterfactual,
            "reward_schedule_progress": float(self.training_progress),
            "charge_amount": charge_amount_kwh,
            "discharge_amount": discharge_amount_kwh,
            "step": self.current_step,
            "step_grid_reduction": step_grid_reduction,
            "step_cost_savings": np.sum(step_cost_savings_per_agent),
            "step_degradation_cost": step_degradation_cost,
        }
        if isinstance(self._last_reward_debug, dict):
            w = self._last_reward_debug.get("weights", {})
            cm = self._last_reward_debug.get("component_means", {})
            for k, v in w.items():
                info[f"reward_w_{k}"] = float(v)
            for k, v in cm.items():
                info[f"reward_comp_{k}"] = float(v)

        self.current_step += 1
        done = (self.current_step >= self.num_steps)

        obs_next = self._get_obs()
        obs_next_list = [obs_next[i] for i in range(self.num_agents)]
        rewards_list = [final_rewards[i] for i in range(self.num_agents)]

        terminated = done
        truncated = False
        return obs_next_list, rewards_list, terminated, truncated, info

        for i, hid in enumerate(self.house_ids):
            norm_own_demand = demands[i] / self.global_max_demand
            norm_own_solar = solars[i] / self.global_max_solar
            sum_others_demand = float(demands.sum() - demands[i])
            sum_others_solar = float(solars.sum() - solars[i])
            norm_others_demand = sum_others_demand / (self.global_max_demand * self.num_agents)
            norm_others_solar = sum_others_solar / (self.global_max_solar * self.num_agents)
            has_solar_flag = float(self.agent_groups[i])
            has_batt_flag = 1.0 if self.battery_manager.has_battery(hid) else 0.0
            if has_batt_flag > 0.0:
                batt_state = self.battery_manager.batteries[hid]
                batt_cap = batt_state.spec.max_capacity
                batt_rate = batt_state.spec.max_charge_rate
            else:
                batt_cap = 0.0
                batt_rate = 0.0
            norm_batt_cap = batt_cap / max_possible_cap
            norm_batt_rate = batt_rate / max_possible_rate
            obs.append([
            norm_own_demand,
            norm_own_solar,
            soc_frac_arr[i],
            norm_grid_price,
            norm_peer_price,
            norm_others_demand,
            norm_others_solar,
            time_norm,
            has_solar_flag,
            has_batt_flag,
            norm_batt_cap,
            norm_batt_rate
            ])
            
        return np.array(obs, dtype=np.float32)


    def _compute_jains_index(self, usage_array):
        """Simple Jain's Fairness Index."""
        x = np.array(usage_array, dtype=np.float32)
        numerator = (np.sum(x))**2
        denominator = len(x) * np.sum(x**2) + 1e-8
        return numerator / denominator


    def _compute_rewards(
        self,
        cost_base,
        cost_actual,
        grid_base,
        grid_actual,
        forced_import,
        shortfall_before_force,
        p2p_buy,
        p2p_sell,
        p2p_buy_request,
        raw_surplus,
        all_cost_savings_prev,
        all_cost_savings_next,
        eps=1e-6,
    ):
        rewards, debug = self.reward_shaper.compute(
            cost_base=cost_base,
            cost_actual=cost_actual,
            grid_base=grid_base,
            grid_actual=grid_actual,
            forced_import=forced_import,
            shortfall_before_force=shortfall_before_force,
            p2p_buy=p2p_buy,
            p2p_sell=p2p_sell,
            p2p_buy_request=p2p_buy_request,
            raw_surplus=raw_surplus,
        )
        self._last_reward_debug = debug
        return rewards

    def get_episode_metrics(self):
        """
        Return performance metrics for the last completed episode.
        Call after episode finishes (after env.reset()).
        """
        return self.episode_metrics


    def save_log(self, filename="env_log.csv"):
        """Save environment step log to CSV."""
        columns = [
            "Step", "Total_Grid_Import", "Total_Grid_Export",
            "Total_P2P_Buy", "Total_P2P_Sell", "Total_Cost",
        ]
        df = pd.DataFrame(self.env_log, columns=columns)
        df.to_csv(filename, index=False)
        print(f"Environment log saved to {filename}")



 
