import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import Optional

from Environment.macro_action import (
    MacroAction,
    macro_to_primitive,
    check_termination,
    get_valid_actions_mask,
    macro_is_still_meaningful,
    IDLE,
    NUM_MACRO_KINDS,
    DURATION_BINS,
    NUM_INTENSITIES,           
    NUM_DURATIONS,            
    DEFAULT_INTENSITY,
)

MAX_MACRO_DURATION = max(DURATION_BINS)


@dataclass
class AgentContractState:
    active: bool
    macro: Optional[MacroAction]
    last_raw_obs: np.ndarray
    last_macro_obs: np.ndarray
    cumulative_reward: float
    contract_age: int
    prev_macro_kind: int
    prev_macro_duration: int
    mismatch_streak: int


class AsyncMacroWrapper(gym.Wrapper):

    def __init__(self, env, gamma=0.99, enforce_valid_mask=True):
        super().__init__(env)

        self.num_agents = env.unwrapped.num_agents
        self.gamma = float(gamma)
        self.enforce_valid_mask = enforce_valid_mask

        self.contract_states = {}
        self.last_decision_mask = None

        self.raw_obs_dim = int(env.observation_space.shape[-1])
        self.macro_obs_dim = self.raw_obs_dim + 1 + NUM_MACRO_KINDS

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.macro_obs_dim),
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------
    def reset(self, **kwargs):
        obs_list, info = self.env.reset(**kwargs)
        obs_array = np.asarray(obs_list, dtype=np.float32)

        self.contract_states = {}
        self.last_decision_mask = np.ones(self.num_agents, dtype=np.float32)

        valid_actions_mask = np.zeros((self.num_agents, NUM_MACRO_KINDS), dtype=bool)
        valid_intensity_mask = np.zeros((self.num_agents, NUM_INTENSITIES), dtype=bool)
        valid_duration_mask = np.zeros((self.num_agents, NUM_DURATIONS), dtype=bool)   

        for i in range(self.num_agents):
            macro_obs = self._build_macro_obs(
                raw_obs=obs_array[i],
                prev_macro_kind=IDLE,
                prev_macro_duration=0,
            )

            self.contract_states[i] = AgentContractState(
                active=False,
                macro=None,
                last_raw_obs=obs_array[i].copy(),
                last_macro_obs=macro_obs,
                cumulative_reward=0.0,
                contract_age=0,
                prev_macro_kind=IDLE,
                prev_macro_duration=0,
                mismatch_streak=0,
            )

            valid_actions_mask[i] = get_valid_actions_mask(obs_array[i], self.env, i)
            valid_intensity_mask[i] = np.ones(NUM_INTENSITIES, dtype=bool)
            valid_duration_mask[i] = np.ones(NUM_DURATIONS, dtype=bool)

        macro_obs_array = np.asarray(
            [self.contract_states[i].last_macro_obs for i in range(self.num_agents)],
            dtype=np.float32,
        )

        info = dict(info)
        info["decision_mask"] = self.last_decision_mask.copy()
        info["valid_actions_mask"] = valid_actions_mask
        info["valid_intensity_mask"] = valid_intensity_mask
        info["valid_duration_mask"] = valid_duration_mask  
        info["contract_age"] = np.zeros(self.num_agents, dtype=np.int32)
        info["macro_duration"] = np.zeros(self.num_agents, dtype=np.int32)
        info["macro_discount"] = np.ones(self.num_agents, dtype=np.float32)
        info["prev_macro_kind"] = np.full(self.num_agents, IDLE, dtype=np.int32)
        info["mismatch_streak"] = np.zeros(self.num_agents, dtype=np.int32)

        return macro_obs_array, info

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------
    def step(self, macro_actions_dict):
        """
        macro_actions_dict: {agent_id: MacroAction}
        ONLY for agents with decision_mask == 1 from previous wrapper step.
        """
        ready_agents = set(np.where(self.last_decision_mask == 1.0)[0].tolist())
        provided_agents = set(macro_actions_dict.keys())

        missing = ready_agents - provided_agents
        extra = provided_agents - ready_agents

        if missing:
            raise ValueError(f"Missing macro actions for ready agents: {sorted(missing)}")
        if extra:
            raise ValueError(f"Received actions for non-ready agents: {sorted(extra)}")

        # -----------------------------------------------------
        # Start new contracts for ready agents
        # -----------------------------------------------------
        for i in range(self.num_agents):
            if self.last_decision_mask[i] != 1.0:
                continue

            state = self.contract_states[i]
            incoming = macro_actions_dict[i]

            if not isinstance(incoming, MacroAction):
                raise TypeError(f"Agent {i} action must be MacroAction, got {type(incoming)}")

            if not (0 <= int(incoming.kind) < NUM_MACRO_KINDS):
                raise ValueError(f"Agent {i} invalid macro kind: {incoming.kind}")

            # FIXED: Enforce both lower and upper bounds for duration scaling
            if not (0 < int(incoming.duration_steps) <= MAX_MACRO_DURATION):
                raise ValueError(f"Agent {i} invalid duration: {incoming.duration_steps}. Must be between 1 and {MAX_MACRO_DURATION}.")

            if self.enforce_valid_mask:
                valid_mask = get_valid_actions_mask(state.last_raw_obs, self.env, i)
                if not valid_mask[int(incoming.kind)]:
                    raise ValueError(
                        f"Agent {i} selected invalid macro {incoming.kind}. "
                        f"Valid mask: {valid_mask.astype(int)}"
                    )

            state.macro = MacroAction(
                kind=int(incoming.kind),
                duration_steps=int(incoming.duration_steps),
                steps_remaining=int(incoming.duration_steps),
                intensity=int(getattr(incoming, "intensity", DEFAULT_INTENSITY)),
            )
            state.active = True
            state.contract_age = 0
            state.cumulative_reward = 0.0
            state.mismatch_streak = 0

        # -----------------------------------------------------
        # Convert active contracts to primitive actions
        # -----------------------------------------------------
        primitives = np.zeros((self.num_agents, 6), dtype=np.float32)

        for i in range(self.num_agents):
            state = self.contract_states[i]

            if state.active and state.macro is not None:
                primitives[i] = macro_to_primitive(
                    state.macro,
                    obs=state.last_raw_obs,
                    env=self.env,
                    agent_idx=i,
                )
            else:
                primitives[i] = np.zeros(6, dtype=np.float32)

        next_obs_list, raw_rewards, terminated, truncated, env_info = self.env.step(primitives)
        next_obs_array = np.asarray(next_obs_list, dtype=np.float32)
        raw_rewards = np.asarray(raw_rewards, dtype=np.float32)

        decision_mask = np.zeros(self.num_agents, dtype=np.float32)
        macro_rewards = np.zeros(self.num_agents, dtype=np.float32)
        valid_actions_mask = np.zeros((self.num_agents, NUM_MACRO_KINDS), dtype=bool)
        valid_intensity_mask = np.zeros((self.num_agents, NUM_INTENSITIES), dtype=bool) 
        valid_duration_mask = np.zeros((self.num_agents, NUM_DURATIONS), dtype=bool)  
        finished_duration = np.zeros(self.num_agents, dtype=np.int32)
        macro_discount = np.ones(self.num_agents, dtype=np.float32)
        prev_macro_kind_arr = np.full(self.num_agents, IDLE, dtype=np.int32)

        for i in range(self.num_agents):
            state = self.contract_states[i]

            state.last_raw_obs = next_obs_array[i].copy()

            if (not state.active) or (state.macro is None):
                decision_mask[i] = 1.0
                macro_rewards[i] = 0.0
                valid_actions_mask[i] = get_valid_actions_mask(state.last_raw_obs, self.env, i)
                valid_intensity_mask[i] = np.ones(NUM_INTENSITIES, dtype=bool)
                valid_duration_mask[i] = np.ones(NUM_DURATIONS, dtype=bool)
                prev_macro_kind_arr[i] = state.prev_macro_kind
                continue
            state.cumulative_reward += raw_rewards[i] * (self.gamma ** state.contract_age)

            state.contract_age += 1
            state.macro.steps_remaining -= 1

            termination_result = check_termination(
                macro=state.macro,
                obs=state.last_raw_obs,
                env=self.env,
                agent_idx=i,
                mismatch_streak=state.mismatch_streak,
            )
            if isinstance(termination_result, tuple):
                should_end, state.mismatch_streak = termination_result
            else:
                should_end = termination_result
                still_meaningful = macro_is_still_meaningful(
                    state.macro.kind,
                    state.last_raw_obs,
                    self.env,
                    i,
                )
                if still_meaningful:
                    state.mismatch_streak = 0
                else:
                    state.mismatch_streak += 1

            if should_end or terminated or truncated:
                done_kind = int(state.macro.kind)
                done_dur = int(state.contract_age)

                decision_mask[i] = 1.0
                macro_rewards[i] = state.cumulative_reward
                finished_duration[i] = done_dur
                macro_discount[i] = float(self.gamma ** done_dur)
                prev_macro_kind_arr[i] = done_kind

                # Build NEW decision observation only at wake-up time
                state.prev_macro_kind = done_kind
                state.prev_macro_duration = done_dur
                state.last_macro_obs = self._build_macro_obs(
                    raw_obs=state.last_raw_obs,
                    prev_macro_kind=done_kind,
                    prev_macro_duration=done_dur,
                )

                valid_actions_mask[i] = get_valid_actions_mask(state.last_raw_obs, self.env, i)
                valid_intensity_mask[i] = np.ones(NUM_INTENSITIES, dtype=bool)
                valid_duration_mask[i] = np.ones(NUM_DURATIONS, dtype=bool)

                state.active = False
                state.macro = None
                state.cumulative_reward = 0.0
                state.contract_age = 0
                state.mismatch_streak = 0

            else:
                decision_mask[i] = 0.0
                macro_rewards[i] = 0.0
                valid_actions_mask[i] = np.zeros(NUM_MACRO_KINDS, dtype=bool)
                valid_intensity_mask[i] = np.zeros(NUM_INTENSITIES, dtype=bool) 
                valid_duration_mask[i] = np.zeros(NUM_DURATIONS, dtype=bool)    
                prev_macro_kind_arr[i] = state.prev_macro_kind

        self.last_decision_mask = decision_mask.copy()

        macro_obs_array = np.asarray(
            [self.contract_states[i].last_macro_obs for i in range(self.num_agents)],
            dtype=np.float32,
        )

        contract_ages = np.asarray(
            [self.contract_states[i].contract_age for i in range(self.num_agents)],
            dtype=np.int32,
        )

        mismatch_streaks = np.asarray(
            [self.contract_states[i].mismatch_streak for i in range(self.num_agents)],
            dtype=np.int32,
        )

        info = {
            "decision_mask": decision_mask,
            "valid_actions_mask": valid_actions_mask,
            "valid_intensity_mask": valid_intensity_mask,   
            "valid_duration_mask": valid_duration_mask,     
            "contract_age": contract_ages,
            "macro_duration": finished_duration,
            "macro_discount": macro_discount,
            "prev_macro_kind": prev_macro_kind_arr,
            "mismatch_streak": mismatch_streaks,
            "primitives": primitives.copy(),
            "env_info": env_info,
            "env_rewards": raw_rewards,
        }

        return macro_obs_array, macro_rewards, terminated, truncated, info

    # ---------------------------------------------------------
    # MACRO-OBS BUILDING
    # ---------------------------------------------------------
    def _build_macro_obs(self, raw_obs, prev_macro_kind, prev_macro_duration):
        raw_obs = np.asarray(raw_obs, dtype=np.float32)

        dur_norm = np.array(
            [float(prev_macro_duration) / float(MAX_MACRO_DURATION + 1e-8)],
            dtype=np.float32,
        )

        kind_one_hot = np.zeros(NUM_MACRO_KINDS, dtype=np.float32)
        if 0 <= int(prev_macro_kind) < NUM_MACRO_KINDS:
            kind_one_hot[int(prev_macro_kind)] = 1.0

        return np.concatenate([raw_obs, dur_norm, kind_one_hot], axis=0)
