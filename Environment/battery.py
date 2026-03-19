from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import random


@dataclass(frozen=True)
class BatterySpec:
    max_capacity: float
    charge_efficiency: float
    discharge_efficiency: float
    max_charge_rate: float
    max_discharge_rate: float
    degradation_cost_per_kwh: float


@dataclass
class BatteryState:
    spec: BatterySpec
    soc: float = 0.0  # kWh stored


class BatteryManager:
    """
    Battery manager with minimal modifications:
      - all solar houses get Franklin batteries
      - charge is limited by real surplus
      - discharge is limited by real shortfall
      - SOC randomization preserved
      - degradation methods preserved
    """

    def __init__(
        self,
        house_ids: List[str],
        battery_houses: List[str],
        battery_options: Dict[str, Dict[str, float]],
        seed: Optional[int] = 42,
    ):
        self.house_ids = list(house_ids)
        self.num_agents = len(self.house_ids)
        self.house_to_idx = {hid: i for i, hid in enumerate(self.house_ids)}

        self._py_rng = random.Random(seed)

        self.specs: Dict[str, BatterySpec] = {
            name: BatterySpec(**cfg) for name, cfg in battery_options.items()
        }

        # Assign Franklin to every battery house
        self.batteries: Dict[str, BatteryState] = {}
        for hid in battery_houses:
            self.batteries[hid] = BatteryState(spec=self.specs["franklin"], soc=0.0)

    def has_battery(self, hid: str) -> bool:
        return hid in self.batteries
    
    def charge_acceptance_limit(self, time_delta: float) -> np.ndarray:
        """
        Max incoming energy (kWh this timestep) each battery can absorb right now.
        """
        out = np.zeros(self.num_agents, dtype=np.float32)
        for hid, b in self.batteries.items():
            i = self.house_to_idx[hid]

            max_energy_input = b.spec.max_charge_rate * time_delta
            space_left = b.spec.max_capacity - b.soc
            max_absorbable_energy = space_left / b.spec.charge_efficiency

            out[i] = min(max_energy_input, max_absorbable_energy)

        return out

    def reset_soc_uniform(self, low_frac: float = 0.00, high_frac: float = 0.00) -> None:
        for hid, b in self.batteries.items():
            low = low_frac * b.spec.max_capacity
            high = high_frac * b.spec.max_capacity
            b.soc = self._py_rng.uniform(low, high)

    def soc_fraction_array(self) -> np.ndarray:
        soc_frac = np.full(self.num_agents, -1.0, dtype=np.float32)
        for hid, b in self.batteries.items():
            i = self.house_to_idx[hid]
            soc_frac[i] = float(b.soc / (b.spec.max_capacity + 1e-12))
        return soc_frac

    def apply_discharge(
        self,
        a_discharge: np.ndarray,
        shortfall_kwh: np.ndarray,
        time_delta: float,
    ) -> np.ndarray:
        """
        Discharge only up to:
          1. requested amount
          2. battery hardware limit
          3. available SOC
          4. actual load shortfall
        Returns discharged energy in kWh for this timestep.
        """
        discharge_amount_kwh = np.zeros(self.num_agents, dtype=np.float32)

        for hid, b in self.batteries.items():
            i = self.house_to_idx[hid]

            max_energy_output = b.spec.max_discharge_rate * time_delta
            available_energy = b.soc * b.spec.discharge_efficiency
            desired_energy = float(a_discharge[i]) * max_energy_output

            actual_kwh = min(
                desired_energy,
                available_energy,
                float(shortfall_kwh[i]),
            )

            b.soc -= actual_kwh / b.spec.discharge_efficiency
            b.soc = max(0.0, b.soc)

            discharge_amount_kwh[i] = actual_kwh

        return discharge_amount_kwh

    def apply_charge(
        self,
        a_charge: np.ndarray,
        surplus_kwh: np.ndarray,
        time_delta: float,
    ) -> np.ndarray:
        """
        Charge only up to:
          1. requested amount
          2. battery hardware limit
          3. remaining battery capacity
          4. actual available solar surplus
        Returns charged energy in kWh for this timestep.
        """
        charge_amount_kwh = np.zeros(self.num_agents, dtype=np.float32)

        for hid, b in self.batteries.items():
            i = self.house_to_idx[hid]

            max_energy_input = b.spec.max_charge_rate * time_delta
            space_left = b.spec.max_capacity - b.soc
            max_absorbable_energy = space_left / b.spec.charge_efficiency
            desired_energy = float(a_charge[i]) * max_energy_input

            actual_kwh = min(
                desired_energy,
                max_absorbable_energy,
                float(surplus_kwh[i]),
            )

            b.soc += actual_kwh * b.spec.charge_efficiency
            b.soc = min(b.spec.max_capacity, b.soc)

            charge_amount_kwh[i] = actual_kwh

        return charge_amount_kwh

    def degradation_cost_step(
        self,
        charge_amount: np.ndarray,
        discharge_amount: np.ndarray,
    ) -> float:
        total = 0.0
        for hid, b in self.batteries.items():
            i = self.house_to_idx[hid]
            total += float(charge_amount[i] + discharge_amount[i]) * b.spec.degradation_cost_per_kwh
        return float(total)

    def degradation_penalty_array(
        self,
        charge_amount: np.ndarray,
        discharge_amount: np.ndarray,
    ) -> np.ndarray:
        pen = np.zeros(self.num_agents, dtype=np.float32)
        for hid, b in self.batteries.items():
            i = self.house_to_idx[hid]
            pen[i] = float(charge_amount[i] + discharge_amount[i]) * b.spec.degradation_cost_per_kwh
        return pen

    def soc_penalty_array(self, target: float = 0.5) -> np.ndarray:
        pen = np.zeros(self.num_agents, dtype=np.float32)
        for hid, b in self.batteries.items():
            i = self.house_to_idx[hid]
            soc_frac = float(b.soc / (b.spec.max_capacity + 1e-12))
            pen[i] = (soc_frac - target) ** 2
        return pen