import os
import sys
import csv
import random
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Environment.solar_sys_environment import SolarSys
from belief_module import MarketBeliefTransformer, MarketBeliefConfig
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_global_market_series(env):
    """
    Build clean global market series from SolarSys env.

    Returns:
      features: [N, 3]
         [:,0] total demand normalized
         [:,1] total solar normalized
         [:,2] time of day normalized
    """
    total_steps = len(env.data)

    all_demands = np.zeros(total_steps, dtype=np.float32)
    all_solars = np.zeros(total_steps, dtype=np.float32)
    all_times = np.zeros(total_steps, dtype=np.float32)

    for hid in env.house_ids:
        all_demands += env.demands[hid].astype(np.float32)
        all_solars += env.solars[hid].astype(np.float32)

    for step in range(total_steps):
        ts = env.data.index[step]
        all_times[step] = (ts.hour * 60 + ts.minute) / (24.0 * 60.0)

    max_demand = float(all_demands.max()) + 1e-8
    max_solar = float(all_solars.max()) + 1e-8

    norm_demands = all_demands / max_demand
    norm_solars = all_solars / max_solar

    features = np.stack([norm_demands, norm_solars, all_times], axis=1).astype(np.float32)

    stats = {
        "max_demand": max_demand,
        "max_solar": max_solar,
    }
    return features, stats


class GlobalMarketBeliefDataset(Dataset):
    """
    Sequence dataset that simulates async corruption:
      - sensor dropout
      - stale observations
      - temporary absence / leave-join
      - missing timesteps
    """
    def __init__(
        self,
        features: np.ndarray,
        seq_len: int = 8,
        pred_horizon: int = 8,
        sensor_dropout_prob: float = 0.10,
        timestep_dropout_prob: float = 0.05,
        stale_prob: float = 0.15,
        max_stale_steps: int = 3,
        absence_prob: float = 0.05,
        seed: int = 42,
    ):
        super().__init__()
        self.features = features
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.sensor_dropout_prob = sensor_dropout_prob
        self.timestep_dropout_prob = timestep_dropout_prob
        self.stale_prob = stale_prob
        self.max_stale_steps = max_stale_steps
        self.absence_prob = absence_prob
        self.rng = np.random.default_rng(seed)

        self.valid_start = len(features) - seq_len - pred_horizon + 1
        if self.valid_start <= 0:
            raise ValueError("Not enough data for the requested seq_len and pred_horizon.")

    def __len__(self):
        return self.valid_start

    def _corrupt_window(self, x_clean):
        """
        x_clean: [T, D]
        Returns:
          x_obs            [T, D]
          feature_mask     [T, D]
          time_delta       [T, 1]
          presence_flag    [T, 1]
          timestep_mask    [T]
        """
        T, D = x_clean.shape
        x_obs = x_clean.copy()
        feature_mask = np.ones((T, D), dtype=np.float32)
        time_delta = np.zeros((T, 1), dtype=np.float32)
        presence_flag = np.ones((T, 1), dtype=np.float32)
        timestep_mask = np.ones((T,), dtype=np.float32)

        # 1) Simulate leave / join / absence
        absent_now = False
        absent_remaining = 0

        for t in range(T):
            if not absent_now and self.rng.random() < self.absence_prob:
                absent_now = True
                absent_remaining = int(self.rng.integers(1, min(4, T - t) + 1))

            if absent_now:
                presence_flag[t, 0] = 0.0
                x_obs[t, :] = 0.0
                feature_mask[t, :] = 0.0
                time_delta[t, 0] = min(1.0, absent_remaining / max(1, self.max_stale_steps))
                absent_remaining -= 1
                if absent_remaining <= 0:
                    absent_now = False

        # 2) Simulate missing full timesteps
        for t in range(T):
            if presence_flag[t, 0] > 0.5 and self.rng.random() < self.timestep_dropout_prob:
                timestep_mask[t] = 0.0
                x_obs[t, :] = 0.0
                feature_mask[t, :] = 0.0

        # 3) Simulate feature-level sensor dropout
        for t in range(T):
            if timestep_mask[t] < 0.5:
                continue
            for d in range(D):
                if self.rng.random() < self.sensor_dropout_prob:
                    x_obs[t, d] = 0.0
                    feature_mask[t, d] = 0.0

        # 4) Simulate stale observations
        # Replace current value with an older one, and record time_delta
        for t in range(T):
            if timestep_mask[t] < 0.5 or presence_flag[t, 0] < 0.5:
                continue
            if self.rng.random() < self.stale_prob:
                lag = int(self.rng.integers(1, min(self.max_stale_steps, t) + 1)) if t > 0 else 0
                if lag > 0:
                    x_obs[t, :] = x_obs[t - lag, :]
                    feature_mask[t, :] = feature_mask[t - lag, :]
                    time_delta[t, 0] = lag / float(max(1, self.max_stale_steps))

        return x_obs, feature_mask, time_delta, presence_flag, timestep_mask

    def __getitem__(self, idx):
        x_clean = self.features[idx : idx + self.seq_len]  # [T, 3]
        y_future = self.features[idx + self.seq_len : idx + self.seq_len + self.pred_horizon]  # [H, 3]
        y_current = x_clean[-1].copy()  # reconstruct most recent clean step

        x_obs, feature_mask, time_delta, presence_flag, timestep_mask = self._corrupt_window(x_clean)

        target_presence = np.array([1.0 if presence_flag[-1, 0] > 0.5 else 0.0], dtype=np.float32)

        sample = {
            "x_obs": torch.tensor(x_obs, dtype=torch.float32),
            "feature_mask": torch.tensor(feature_mask, dtype=torch.float32),
            "time_delta": torch.tensor(time_delta, dtype=torch.float32),
            "presence_flag": torch.tensor(presence_flag, dtype=torch.float32),
            "timestep_mask": torch.tensor(timestep_mask, dtype=torch.float32),
            "y_future": torch.tensor(y_future, dtype=torch.float32),
            "y_current": torch.tensor(y_current, dtype=torch.float32),
            "target_presence": torch.tensor(target_presence, dtype=torch.float32),
        }
        return sample


# ------------------------------------------------------------
# Time-aware train/val split
# ------------------------------------------------------------
def make_datasets(
    features,
    seq_len,
    pred_horizon,
    train_frac=0.8,
    **dataset_kwargs,
):
    split_idx = int(len(features) * train_frac)

    # Keep overlap so validation windows have enough history
    train_features = features[:split_idx]
    val_features = features[max(0, split_idx - seq_len - pred_horizon) :]

    train_ds = GlobalMarketBeliefDataset(
        train_features,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        **dataset_kwargs,
    )
    val_ds = GlobalMarketBeliefDataset(
        val_features,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        **dataset_kwargs,
    )
    return train_ds, val_ds


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------
def compute_losses(outputs, batch, device):
    y_future = batch["y_future"].to(device)
    y_current = batch["y_current"].to(device)
    target_presence = batch["target_presence"].to(device)

    pred_future = outputs["pred_future"]
    recon_current = outputs["recon_current"]
    presence_logit = outputs["presence_logit"]

    loss_future = nn.functional.mse_loss(pred_future, y_future)
    loss_recon = nn.functional.mse_loss(recon_current, y_current)
    loss_presence = nn.functional.binary_cross_entropy_with_logits(
        presence_logit, target_presence
    )

    total_loss = 1.0 * loss_future + 0.5 * loss_recon + 0.2 * loss_presence

    return total_loss, {
        "loss_total": float(total_loss.detach().cpu()),
        "loss_future": float(loss_future.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "loss_presence": float(loss_presence.detach().cpu()),
    }


# ------------------------------------------------------------
# Live plotter
# ------------------------------------------------------------
class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.train_total = []
        self.val_total = []
        self.train_future = []
        self.val_future = []

    def update(self, history):
        self.train_total.append(history["train_loss_total"][-1])
        self.val_total.append(history["val_loss_total"][-1])
        self.train_future.append(history["train_loss_future"][-1])
        self.val_future.append(history["val_loss_future"][-1])

        self.ax.clear()
        self.ax.plot(self.train_total, label="Train Total")
        self.ax.plot(self.val_total, label="Val Total")
        self.ax.plot(self.train_future, label="Train Future")
        self.ax.plot(self.val_future, label="Val Future")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Belief Module Training")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.fig.tight_layout()
        plt.pause(0.001)

    def save(self, path):
        self.fig.savefig(path, dpi=200, bbox_inches="tight")


# ------------------------------------------------------------
# Train / eval loops
# ------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    meter = {
        "loss_total": 0.0,
        "loss_future": 0.0,
        "loss_recon": 0.0,
        "loss_presence": 0.0,
    }
    n_samples = 0

    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        feature_mask = batch["feature_mask"].to(device)
        time_delta = batch["time_delta"].to(device)
        presence_flag = batch["presence_flag"].to(device)
        timestep_mask = batch["timestep_mask"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(
                x=x_obs,
                feature_mask=feature_mask,
                time_delta=time_delta,
                presence_flag=presence_flag,
                timestep_mask=timestep_mask,
            )

            loss, loss_dict = compute_losses(outputs, batch, device)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        bs = x_obs.size(0)
        n_samples += bs
        for k in meter:
            meter[k] += loss_dict[k] * bs

    for k in meter:
        meter[k] /= max(1, n_samples)

    return meter


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    set_seed(42)

    DATA_FILE_PATH = os.environ.get(
        "BELIEF_DATA_PATH",
        "/Users/ananygupta/Desktop/PeARL_sync/Data/3solar_2nonsolar_2018-02-01_2018-04-07.csv",
    )
    SAVE_DIR = os.environ.get("BELIEF_SAVE_DIR", "belief_training_runs")
    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data config
    TIME_FREQ = os.environ.get("BELIEF_TIME_FREQ", "15min")
    EPISODE_DAYS = int(os.environ.get("BELIEF_EPISODE_DAYS", "60"))
    SEQ_LEN = int(os.environ.get("BELIEF_SEQ_LEN", "8"))
    PRED_HORIZON = int(os.environ.get("BELIEF_PRED_HORIZON", "8"))
    BATCH_SIZE = int(os.environ.get("BELIEF_BATCH_SIZE", "128"))

    # Training config
    EPOCHS = int(os.environ.get("BELIEF_EPOCHS", "60"))
    LR = float(os.environ.get("BELIEF_LR", "1e-3"))
    WEIGHT_DECAY = float(os.environ.get("BELIEF_WEIGHT_DECAY", "1e-5"))
    STATE = os.environ.get("BELIEF_STATE", "pennsylvania")
    NO_PLOT = str(os.environ.get("BELIEF_NO_PLOT", "0")).strip() in {"1", "true", "TRUE", "yes", "YES"}

    # Corruption settings to mimic async conditions
    dataset_kwargs = dict(
        sensor_dropout_prob=0.10,
        timestep_dropout_prob=0.05,
        stale_prob=0.15,
        max_stale_steps=3,
        absence_prob=0.05,
        seed=42,
    )

    print("Loading data via SolarSys...")
    env = SolarSys(
        data_path=DATA_FILE_PATH,
        state=STATE,
        time_freq=TIME_FREQ,
        episode_days=EPISODE_DAYS,
    )
    env.reset()

    print("Extracting global market series...")
    features, stats = extract_global_market_series(env)
    print(f"Features shape: {features.shape}")
    print(f"Stats: {stats}")

    train_ds, val_ds = make_datasets(
        features=features,
        seq_len=SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        train_frac=0.8,
        **dataset_kwargs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    cfg = MarketBeliefConfig(
        input_dim=3,
        model_dim=128,
        nhead=4,
        num_layers=3,
        ff_dim=256,
        dropout=0.1,
        max_seq_len=SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        output_dim=3,
        latent_dim=128,
        use_feature_mask=True,
        use_time_delta=True,
        use_presence_flag=True,
    )

    model = MarketBeliefTransformer(cfg).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    history = {
        "train_loss_total": [],
        "val_loss_total": [],
        "train_loss_future": [],
        "val_loss_future": [],
        "train_loss_recon": [],
        "val_loss_recon": [],
        "train_loss_presence": [],
        "val_loss_presence": [],
        "lr": [],
    }

    live_plot = None if NO_PLOT else LivePlotter()

    best_val = float("inf")
    best_path = os.path.join(SAVE_DIR, "best_belief_model.pth")
    final_path = os.path.join(SAVE_DIR, "final_belief_model.pth")
    plot_path = os.path.join(SAVE_DIR, "loss_curves.png")
    csv_path = os.path.join(SAVE_DIR, "training_history.csv")
    config_path = os.path.join(SAVE_DIR, "config.txt")

    with open(config_path, "w") as f:
        f.write("=== Model Config ===\n")
        f.write(str(asdict(cfg)) + "\n\n")
        f.write("=== Training Config ===\n")
        f.write(f"SEQ_LEN={SEQ_LEN}\n")
        f.write(f"PRED_HORIZON={PRED_HORIZON}\n")
        f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"EPOCHS={EPOCHS}\n")
        f.write(f"LR={LR}\n")
        f.write(f"WEIGHT_DECAY={WEIGHT_DECAY}\n")
        f.write(f"DATASET_KWARGS={dataset_kwargs}\n")
        f.write(f"STATS={stats}\n")

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        train_meter = run_epoch(model, train_loader, optimizer, DEVICE, train=True)
        val_meter = run_epoch(model, val_loader, optimizer, DEVICE, train=False)

        scheduler.step(val_meter["loss_total"])

        history["train_loss_total"].append(train_meter["loss_total"])
        history["val_loss_total"].append(val_meter["loss_total"])
        history["train_loss_future"].append(train_meter["loss_future"])
        history["val_loss_future"].append(val_meter["loss_future"])
        history["train_loss_recon"].append(train_meter["loss_recon"])
        history["val_loss_recon"].append(val_meter["loss_recon"])
        history["train_loss_presence"].append(train_meter["loss_presence"])
        history["val_loss_presence"].append(val_meter["loss_presence"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Total: {train_meter['loss_total']:.6f} | "
            f"Val Total: {val_meter['loss_total']:.6f} | "
            f"Train Future: {train_meter['loss_future']:.6f} | "
            f"Val Future: {val_meter['loss_future']:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if live_plot is not None:
            live_plot.update(history)

        if val_meter["loss_total"] < best_val:
            best_val = val_meter["loss_total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "best_val_loss": best_val,
                    "stats": stats,
                },
                best_path,
            )
            print(f"  -> Saved best checkpoint to {best_path}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "best_val_loss": best_val,
            "stats": stats,
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")

    if live_plot is not None:
        live_plot.save(plot_path)
        print(f"Saved loss plot to {plot_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss_total",
            "val_loss_total",
            "train_loss_future",
            "val_loss_future",
            "train_loss_recon",
            "val_loss_recon",
            "train_loss_presence",
            "val_loss_presence",
            "lr",
        ])
        for i in range(len(history["train_loss_total"])):
            writer.writerow([
                i + 1,
                history["train_loss_total"][i],
                history["val_loss_total"][i],
                history["train_loss_future"][i],
                history["val_loss_future"][i],
                history["train_loss_recon"][i],
                history["val_loss_recon"][i],
                history["train_loss_presence"][i],
                history["val_loss_presence"][i],
                history["lr"][i],
            ])
    print(f"Saved training history CSV to {csv_path}")

    if live_plot is not None:
        plt.ioff()
        plt.show()
    print("Training complete.")


if __name__ == "__main__":
    main()
