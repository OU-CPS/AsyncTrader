import numpy as np
import torch

class MacroRolloutBuffer:
    def __init__(self, num_agents, gamma=0.99, gae_lambda=0.95):
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        """Clears the buffer for the next rollout phase."""
        self.data = {
            i: {
                "actor_obs": [],   
                "critic_hidden_before": [],
                "actor_hidden_before": [],
                "step_t": [],
                "agent_ids": [],
                "action_kinds": [],
                "action_durs": [],
                "action_intensities": [],
                "rewards": [],     
                "values": [],      
                "logprobs": [],    
                "durations": [],   
                "dones": [],       
                "valid_masks": [], 
                "valid_intensity_masks": [], 
                "valid_duration_masks": [],  
                "agent_types": []  
            } for i in range(self.num_agents)
        }

    def add(
            self,
            agent_id,
            actor_obs,
            critic_hidden_before,
            actor_hidden_before,
            step_t,
            action_kind,
            action_dur,
            reward,
            value,
            logprob,
            duration,
            action_intensity,
            done,
            valid_mask,
            valid_intensity_mask, 
            valid_duration_mask, 
            agent_type,
        ):
        """Stores a single completed macro-transition for a specific agent."""
        agent_buffer = self.data[agent_id]
        agent_buffer["actor_obs"].append(actor_obs)
        agent_buffer["critic_hidden_before"].append(critic_hidden_before)
        agent_buffer["actor_hidden_before"].append(actor_hidden_before)
        agent_buffer["step_t"].append(step_t)
        agent_buffer["agent_ids"].append(agent_id)
        agent_buffer["action_kinds"].append(action_kind)
        agent_buffer["action_durs"].append(action_dur)
        agent_buffer["rewards"].append(reward)
        agent_buffer["values"].append(value)
        agent_buffer["logprobs"].append(logprob)
        agent_buffer["durations"].append(duration)
        agent_buffer["action_intensities"].append(action_intensity)
        agent_buffer["dones"].append(done)
        agent_buffer["valid_masks"].append(valid_mask)
        agent_buffer["valid_intensity_masks"].append(valid_intensity_mask)
        agent_buffer["valid_duration_masks"].append(valid_duration_mask)  
        agent_buffer["agent_types"].append(agent_type)

    def compute_returns_and_advantages(self, last_values, last_dones):
        """
        Computes the SMDP-modified GAE and returns for all agents.
        last_values: array of shape (num_agents,) containing the critic's value of the final observation.
        last_dones: array of shape (num_agents,) indicating if the final state is terminal.
        """
        for i in range(self.num_agents):
            agent_data = self.data[i]
            n_transitions = len(agent_data["rewards"])
            
            if n_transitions == 0:
                agent_data["advantages"] = []
                agent_data["returns"] = []
                continue

            advantages = np.zeros(n_transitions, dtype=np.float32)
            last_gae_lam = 0.0

        
            for step in reversed(range(n_transitions)):
                tau = agent_data["durations"][step]
                done = float(agent_data["dones"][step])

                if step == n_transitions - 1:
                    next_value = last_values[i]
                    next_non_terminal = 1.0 - float(last_dones[i])
                else:
                    next_value = agent_data["values"][step + 1]
                    next_non_terminal = 1.0 - done

                discount = (self.gamma ** tau) * next_non_terminal
                delta = agent_data["rewards"][step] + discount * next_value - agent_data["values"][step]
                last_gae_lam = delta + discount * self.gae_lambda * last_gae_lam
                advantages[step] = last_gae_lam

            values_array = np.array(agent_data["values"], dtype=np.float32)
            agent_data["advantages"] = advantages.tolist()
            agent_data["returns"] = (advantages + values_array).tolist()

    def get_flattened_data(self):
        """
        Flattens the per-agent lists into single PyTorch tensors for the PPO update.
        Yields dictionary of tensors ready for network training.
        """
        all_actor_obs = [] 
        all_action_kinds, all_action_durs = [], []
        all_action_intensities = []
        all_returns, all_advantages = [], []
        all_logprobs, all_values = [], []
        all_valid_masks, all_agent_types = [], []
        all_valid_intensity_masks, all_valid_duration_masks = [], [] 
        all_critic_hidden_before, all_actor_hidden_before = [], []
        all_step_t, all_agent_ids = [], []

        for i in range(self.num_agents):
            agent_data = self.data[i]
            if len(agent_data["rewards"]) == 0:
                continue
                
            all_actor_obs.extend(agent_data["actor_obs"])
            all_action_kinds.extend(agent_data["action_kinds"])
            all_action_durs.extend(agent_data["action_durs"])
            all_returns.extend(agent_data["returns"])
            all_advantages.extend(agent_data["advantages"])
            all_logprobs.extend(agent_data["logprobs"])
            all_values.extend(agent_data["values"])
            all_valid_masks.extend(agent_data["valid_masks"])
            all_valid_intensity_masks.extend(agent_data["valid_intensity_masks"])
            all_valid_duration_masks.extend(agent_data["valid_duration_masks"])  
            all_action_intensities.extend(agent_data["action_intensities"])
            all_agent_types.extend(agent_data["agent_types"])
            all_step_t.extend(agent_data["step_t"])
            all_agent_ids.extend(agent_data["agent_ids"])
            all_critic_hidden_before.extend(agent_data["critic_hidden_before"])
            all_actor_hidden_before.extend(agent_data["actor_hidden_before"])


        data_dict = {
            "actor_obs": torch.tensor(np.array(all_actor_obs), dtype=torch.float32),
            "critic_hidden_before": torch.tensor(np.array(all_critic_hidden_before), dtype=torch.float32),
            "actor_hidden_before": torch.tensor(np.array(all_actor_hidden_before), dtype=torch.float32),
            "step_t": torch.tensor(all_step_t, dtype=torch.long),
            "agent_ids": torch.tensor(all_agent_ids, dtype=torch.long),
            "action_kinds": torch.tensor(all_action_kinds, dtype=torch.long),
            "action_durs": torch.tensor(all_action_durs, dtype=torch.long),
            "action_intensities": torch.tensor(all_action_intensities, dtype=torch.float32),
            "returns": torch.tensor(all_returns, dtype=torch.float32),
            "advantages": torch.tensor(all_advantages, dtype=torch.float32),
            "logprobs": torch.tensor(all_logprobs, dtype=torch.float32),
            "values": torch.tensor(all_values, dtype=torch.float32),
            "valid_masks": torch.tensor(np.array(all_valid_masks), dtype=torch.bool),
            "valid_intensity_masks": torch.tensor(np.array(all_valid_intensity_masks), dtype=torch.bool), 
            "valid_duration_masks": torch.tensor(np.array(all_valid_duration_masks), dtype=torch.bool),   
            "agent_types": torch.tensor(all_agent_types, dtype=torch.long),
        }

        return data_dict