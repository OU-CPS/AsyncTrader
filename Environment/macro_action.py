import numpy as np
from dataclasses import dataclass

IDLE = 0
DEFICIT_PEER_ONLY = 1
DEFICIT_BATTERY_ONLY = 2
DEFICIT_HYBRID = 3
SURPLUS_STORE_LOCAL = 4
SURPLUS_SELL_PEER = 5
SURPLUS_EXPORT_GRID = 6
SURPLUS_SELL_THEN_EXPORT = 7
CHARGE_FROM_PEERS = 8
NUM_MACRO_KINDS = 9

MACRO_NAMES = {
    IDLE: "IDLE",
    DEFICIT_PEER_ONLY: "DEFICIT_PEER_ONLY",
    DEFICIT_BATTERY_ONLY: "DEFICIT_BATTERY_ONLY",
    DEFICIT_HYBRID: "DEFICIT_HYBRID",
    SURPLUS_STORE_LOCAL: "SURPLUS_STORE_LOCAL",
    SURPLUS_SELL_PEER: "SURPLUS_SELL_PEER",
    SURPLUS_EXPORT_GRID: "SURPLUS_EXPORT_GRID",
    SURPLUS_SELL_THEN_EXPORT: "SURPLUS_SELL_THEN_EXPORT",
    CHARGE_FROM_PEERS: "CHARGE_FROM_PEERS",
}

DURATION_BINS = [1, 2, 4, 6]
NUM_DURATIONS = len(DURATION_BINS)

INTENSITY_LOW = 0
INTENSITY_MED = 1
INTENSITY_HIGH = 2
NUM_INTENSITIES = 3
DEFAULT_INTENSITY = INTENSITY_MED

INTENSITY_NAMES = {
    INTENSITY_LOW: "LOW",
    INTENSITY_MED: "MED",
    INTENSITY_HIGH: "HIGH",
}

BUY_PEER_LEVELS = np.array([0.78, 0.90, 0.99], dtype=np.float32)
SELL_PEER_LEVELS = np.array([0.78, 0.90, 0.99], dtype=np.float32)
CHARGE_LEVELS = np.array([0.74, 0.88, 0.99], dtype=np.float32)
DISCHARGE_LEVELS = np.array([0.82, 0.92, 0.99], dtype=np.float32)
EXPORT_LEVELS = np.array([0.88, 0.94, 0.99], dtype=np.float32)
BUY_GRID_LEVELS = np.array([0.35, 0.70, 1.00], dtype=np.float32)

REGIME_EPS = 0.02
TERMINATION_PATIENCE = 2
SOC_MIN_DISCHARGE = 0.12
SOC_DISCHARGE_PREFERRED = 0.20
SOC_MAX_CHARGE = 0.92

@dataclass
class MacroAction:
    kind: int
    duration_steps: int
    steps_remaining: int
    intensity: int = DEFAULT_INTENSITY

def _clip_intensity(intensity: int) -> int:
    return int(np.clip(intensity, 0, NUM_INTENSITIES - 1))

def _level(levels: np.ndarray, intensity: int) -> float:
    return float(levels[_clip_intensity(intensity)])

def macro_name(kind: int) -> str:
    return MACRO_NAMES.get(int(kind), f"UNKNOWN_{kind}")

def intensity_name(intensity: int) -> str:
    return INTENSITY_NAMES.get(int(intensity), f"INT_{intensity}")

def get_physical_state(obs, env, agent_idx):
    unwrapped = env.unwrapped
    hid = unwrapped.house_ids[agent_idx]
    step = min(unwrapped.current_step, unwrapped.num_steps - 1)
    raw_demand = float(unwrapped.demands[hid][step])
    raw_solar = float(unwrapped.solars[hid][step])
    surplus = max(0.0, raw_solar - raw_demand)
    shortfall = max(0.0, raw_demand - raw_solar)
    soc = float(obs[2])
    has_batt = bool(obs[9] > 0.5)
    return surplus, shortfall, soc, has_batt

def get_valid_actions_mask(obs, env, agent_idx, eps=REGIME_EPS):
    surplus, shortfall, soc, has_batt = get_physical_state(obs, env, agent_idx)
    mask = np.zeros(NUM_MACRO_KINDS, dtype=bool)
    mask[IDLE] = True
    if shortfall > eps:
        mask[DEFICIT_PEER_ONLY] = True
        if has_batt and soc > SOC_MIN_DISCHARGE:
            mask[DEFICIT_BATTERY_ONLY] = True
            mask[DEFICIT_HYBRID] = True
    if surplus > eps:
        mask[SURPLUS_SELL_PEER] = True
        mask[SURPLUS_EXPORT_GRID] = True
        mask[SURPLUS_SELL_THEN_EXPORT] = True
        if has_batt and soc < SOC_MAX_CHARGE:
            mask[SURPLUS_STORE_LOCAL] = True
    if has_batt and soc > SOC_MIN_DISCHARGE and shortfall <= eps:
        mask[SURPLUS_SELL_PEER] = True
    if has_batt and soc < SOC_MAX_CHARGE and shortfall <= eps and surplus <= eps:
        mask[CHARGE_FROM_PEERS] = True
    return mask

def get_valid_intensity_mask(macro_kind: int):
    mask = np.ones(NUM_INTENSITIES, dtype=bool)
    if int(macro_kind) == IDLE:
        mask[:] = False
        mask[DEFAULT_INTENSITY] = True
    return mask

def get_valid_duration_mask(macro_kind: int):
    mask = np.ones(NUM_DURATIONS, dtype=bool)
    if int(macro_kind) == IDLE:
        mask[:] = False
        mask[0] = True
    return mask

def macro_to_primitive(macro_or_kind, obs, env, agent_idx, intensity=None):
    if isinstance(macro_or_kind, MacroAction):
        kind = int(macro_or_kind.kind)
        intensity = int(macro_or_kind.intensity)
    else:
        kind = int(macro_or_kind)
        intensity = DEFAULT_INTENSITY if intensity is None else int(intensity)
    intensity = _clip_intensity(intensity)
    a = np.zeros(6, dtype=np.float32)
    surplus, shortfall, soc, has_batt = get_physical_state(obs, env, agent_idx)
    if kind == IDLE:
        return a
    elif kind == DEFICIT_PEER_ONLY:
        if shortfall > REGIME_EPS:
            a[2] = _level(BUY_PEER_LEVELS, intensity)
            a[0] = 0.55
    elif kind == DEFICIT_BATTERY_ONLY:
        if shortfall > REGIME_EPS and has_batt and soc > SOC_MIN_DISCHARGE:
            a[5] = _level(DISCHARGE_LEVELS, intensity)
            a[0] = 0.45
    elif kind == DEFICIT_HYBRID:
        if shortfall > REGIME_EPS:
            a[2] = _level(BUY_PEER_LEVELS, intensity)
            if has_batt and soc > max(SOC_MIN_DISCHARGE, SOC_DISCHARGE_PREFERRED):
                a[5] = _level(DISCHARGE_LEVELS, intensity)
            a[0] = 0.35
    elif kind == SURPLUS_STORE_LOCAL:
        if surplus > REGIME_EPS and has_batt and soc < SOC_MAX_CHARGE:
            a[4] = _level(CHARGE_LEVELS, intensity)
    elif kind == SURPLUS_SELL_PEER:
        if surplus > REGIME_EPS:
            a[3] = _level(SELL_PEER_LEVELS, intensity)
            if has_batt and soc > SOC_DISCHARGE_PREFERRED and shortfall <= REGIME_EPS:
                a[5] = _level(DISCHARGE_LEVELS, intensity)
        elif has_batt and soc > SOC_MIN_DISCHARGE and shortfall <= REGIME_EPS:
            a[3] = _level(SELL_PEER_LEVELS, intensity)
            a[5] = _level(DISCHARGE_LEVELS, intensity)
    elif kind == SURPLUS_EXPORT_GRID:
        if surplus > REGIME_EPS:
            a[1] = _level(EXPORT_LEVELS, intensity)
    elif kind == SURPLUS_SELL_THEN_EXPORT:
        if surplus > REGIME_EPS:
            a[3] = _level(SELL_PEER_LEVELS, intensity)
            a[1] = _level(EXPORT_LEVELS, intensity)
    elif kind == CHARGE_FROM_PEERS:
        if has_batt and soc < SOC_MAX_CHARGE and shortfall <= REGIME_EPS and surplus <= REGIME_EPS:
            a[2] = _level(BUY_PEER_LEVELS, intensity)
            a[4] = _level(CHARGE_LEVELS, intensity)
    return a

def macro_is_still_meaningful(macro_kind, obs, env, agent_idx, eps=REGIME_EPS):
    surplus, shortfall, soc, has_batt = get_physical_state(obs, env, agent_idx)
    kind = int(macro_kind)
    if kind == IDLE:
        return True
    elif kind == DEFICIT_PEER_ONLY:
        return shortfall > eps
    elif kind == DEFICIT_BATTERY_ONLY:
        return (shortfall > eps) and has_batt and (soc > SOC_MIN_DISCHARGE)
    elif kind == DEFICIT_HYBRID:
        return shortfall > eps
    elif kind == SURPLUS_STORE_LOCAL:
        return (surplus > eps) and has_batt and (soc < SOC_MAX_CHARGE)
    elif kind == SURPLUS_SELL_PEER:
        return (surplus > eps) or (has_batt and (soc > SOC_MIN_DISCHARGE) and (shortfall <= eps))
    elif kind == SURPLUS_EXPORT_GRID:
        return surplus > eps
    elif kind == SURPLUS_SELL_THEN_EXPORT:
        return surplus > eps
    elif kind == CHARGE_FROM_PEERS:
        return has_batt and (soc < SOC_MAX_CHARGE) and (shortfall <= eps) and (surplus <= eps)
    return False

def check_termination(
    macro: MacroAction,
    obs,
    env,
    agent_idx,
    mismatch_streak=0,
    eps=REGIME_EPS,
    patience=TERMINATION_PATIENCE,
):
    if macro.steps_remaining <= 0:
        return True
    surplus, shortfall, soc, has_batt = get_physical_state(obs, env, agent_idx)
    kind = int(macro.kind)
    if kind == DEFICIT_BATTERY_ONLY:
        if (not has_batt) or (soc <= SOC_MIN_DISCHARGE):
            return True
    elif kind == SURPLUS_STORE_LOCAL:
        if (not has_batt) or (soc >= SOC_MAX_CHARGE):
            return True
    elif kind == CHARGE_FROM_PEERS:
        if (not has_batt) or (soc >= SOC_MAX_CHARGE):
            return True
    still_meaningful = macro_is_still_meaningful(
        kind, obs, env, agent_idx, eps=eps
    )
    if not still_meaningful and mismatch_streak >= patience:
        return True
    return False
