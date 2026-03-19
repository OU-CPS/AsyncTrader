import numpy as np


class AdaptiveRewardShaper:

    DEFAULT_WEIGHTS = {
        "cost": 1.00,
        "grid": 2.00,
        "p2p": 0.08,
        "forced": 0.18,
        "supply": 0.35,
        "unmatched": 0.30,
    }

    def __init__(
        self,
        base_weights=None,
        use_schedule=True,
        use_auto_weight=True,
        ema_alpha=0.03,
    ):
        self.base_weights = dict(self.DEFAULT_WEIGHTS)
        if base_weights:
            self.base_weights.update(base_weights)

        self.use_schedule = bool(use_schedule)
        self.use_auto_weight = bool(use_auto_weight)
        self.ema_alpha = float(ema_alpha)
        self.progress = 0.0
        self._eps = 1e-6

        self.ema_abs_components = {
            "cost": 1.0,
            "grid": 1.0,
            "p2p": 1.0,
            "forced": 1.0,
            "supply": 1.0,
            "unmatched": 1.0,
        }

    def set_progress(self, progress: float):
        self.progress = float(np.clip(progress, 0.0, 1.0))

    def _scheduled_weights(self):
        if not self.use_schedule:
            return dict(self.base_weights)

        p = self.progress
        early_mult = {
            "cost": 0.70,
            "grid": 0.80,
            "p2p": 1.80,
            "forced": 1.00,
            "supply": 2.20,
            "unmatched": 0.80,
        }
        late_mult = {
            "cost": 1.30,
            "grid": 1.40,
            "p2p": 0.80,
            "forced": 1.20,
            "supply": 0.90,
            "unmatched": 1.50,
        }

        w = {}
        for k in self.base_weights:
            m = (1.0 - p) * early_mult[k] + p * late_mult[k]
            w[k] = self.base_weights[k] * m
        return w

    def _auto_rebalance(self, weights, comp_means):
        if not self.use_auto_weight:
            return weights
        for k, m in comp_means.items():
            prev = self.ema_abs_components[k]
            self.ema_abs_components[k] = (1.0 - self.ema_alpha) * prev + self.ema_alpha * float(abs(m))
        scaled = {}
        for k, w in weights.items():
            inv = 1.0 / (self.ema_abs_components[k] + self._eps)
            scaled[k] = w * inv

        # Keep total weight mass stable
        target_sum = sum(weights.values()) + self._eps
        current_sum = sum(scaled.values()) + self._eps
        factor = target_sum / current_sum
        for k in scaled:
            scaled[k] *= factor
        return scaled

    def compute(
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
    ):
        den_floor = 1e-3

        # 1) local cost savings
        cost_saving = cost_base - cost_actual
        norm_cost = cost_saving / np.maximum(np.abs(cost_base), den_floor)
        norm_cost = np.clip(norm_cost, -1.0, 1.0)

        # 2) local grid reduction
        grid_red = grid_base - grid_actual
        norm_grid = grid_red / np.maximum(grid_base, den_floor)
        norm_grid = np.clip(norm_grid, -1.0, 1.0)

        # 3) p2p activity
        p2p_act = p2p_buy + p2p_sell
        norm_p2p = p2p_act / np.maximum(grid_base, den_floor)
        norm_p2p = np.clip(norm_p2p, 0.0, 1.0)

        # 4) forced import dependence
        norm_forced = forced_import / np.maximum(shortfall_before_force, den_floor)
        norm_forced = np.clip(norm_forced, 0.0, 1.0)

        # 5) seller-side supply quality
        norm_supply = p2p_sell / np.maximum(raw_surplus, den_floor)
        norm_supply = np.clip(norm_supply, 0.0, 1.0)

        # 6) unmatched peer buy request penalty
        unmatched_buy = np.maximum(p2p_buy_request - p2p_buy, 0.0)
        norm_unmatched = unmatched_buy / np.maximum(shortfall_before_force, den_floor)
        norm_unmatched = np.clip(norm_unmatched, 0.0, 1.0)

        comp_means = {
            "cost": float(np.mean(norm_cost)),
            "grid": float(np.mean(norm_grid)),
            "p2p": float(np.mean(norm_p2p)),
            "forced": float(np.mean(norm_forced)),
            "supply": float(np.mean(norm_supply)),
            "unmatched": float(np.mean(norm_unmatched)),
        }

        w = self._scheduled_weights()
        w = self._auto_rebalance(w, comp_means)

        rewards = (
            w["cost"] * norm_cost
            + w["grid"] * norm_grid
            + w["p2p"] * norm_p2p
            + w["supply"] * norm_supply
            - w["forced"] * norm_forced
            - w["unmatched"] * norm_unmatched
        )

        debug = {
            "weights": w,
            "component_means": comp_means,
            "schedule_progress": self.progress,
        }
        return rewards.astype(np.float32), debug

