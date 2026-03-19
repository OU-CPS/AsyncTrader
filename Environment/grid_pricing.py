# grid_pricing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class PricingConfig:
    max_grid_price: float
    feed_in_tariff: float
    price_fn: Callable[[int, int], float]  


def _hour_of_step(step_idx: int, steps_per_day: int) -> int:
    minutes_per_step = 24 * 60 / max(steps_per_day, 1)
    hour = int((step_idx * minutes_per_step) // 60) % 24
    return hour


def _oklahoma_price(step_idx: int, steps_per_day: int) -> float:
    # OG&E TOU: Peak 2pm–7pm
    hour = _hour_of_step(step_idx, steps_per_day)
    return 0.2112 if 14 <= hour < 19 else 0.0434


def _colorado_price(step_idx: int, steps_per_day: int) -> float:
    # Xcel CO TOU: On 3pm–7pm, Mid 1pm–3pm
    hour = _hour_of_step(step_idx, steps_per_day)
    if 15 <= hour < 19:
        return 0.32
    elif 13 <= hour < 15:
        return 0.22
    else:
        return 0.12


def _pennsylvania_price(step_idx: int, steps_per_day: int) -> float:
    # Duquesne Light EV TOU: Peak 1pm–9pm, Super off 11pm–6am
    hour = _hour_of_step(step_idx, steps_per_day)
    if 13 <= hour < 21:
        return 0.125048
    elif hour >= 23 or hour < 6:
        return 0.057014
    else:
        return 0.079085


_PRICING_TABLE: Dict[str, PricingConfig] = {
    "oklahoma": PricingConfig(
        max_grid_price=0.2112,
        feed_in_tariff=0.04,
        price_fn=_oklahoma_price,
    ),
    "colorado": PricingConfig(
        max_grid_price=0.32,
        feed_in_tariff=0.055,
        price_fn=_colorado_price,
    ),
    "pennsylvania": PricingConfig(
        max_grid_price=0.12505,
        feed_in_tariff=0.06,
        price_fn=_pennsylvania_price,
    ),
}


class GridPricing:
    """
    Handles: state selection, grid price lookup, max_grid_price, feed_in_tariff,
    and 'is_peak' detection.
    """

    def __init__(self, state: str, steps_per_day: int):
        self.state = state.lower().strip()
        if self.state not in _PRICING_TABLE:
            raise ValueError(f"State '{state}' unsupported. Options: {list(_PRICING_TABLE.keys())}")

        self.steps_per_day = int(steps_per_day)
        self.cfg = _PRICING_TABLE[self.state]

        self.max_grid_price = float(self.cfg.max_grid_price)
        self.feed_in_tariff = float(self.cfg.feed_in_tariff)

    def grid_price(self, step_idx: int) -> float:
        return float(self.cfg.price_fn(int(step_idx), self.steps_per_day))

    def is_peak(self, step_idx: int, tol: float = 0.99) -> bool:
        # Matches your old logic: grid_price >= max_grid_price*0.99
        return self.grid_price(step_idx) >= self.max_grid_price * tol