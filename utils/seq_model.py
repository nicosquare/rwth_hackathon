"""B4: sequence model that reads raw quarter-hour history up to the bid cutoff.

Run from the repository root with ``uv run python -m utils.seq_model``.

One sample is (household, delivery day D).  Inputs, all known at the day-ahead
cutoff (``DAY_AHEAD``: 10:00 UTC on D-1):

* a 28 x 96 grid: row k is day D-k, column s is quarter-hour slot s, so column s
  lines up with the target slot.  Values at or after the cutoff (the slots of
  D-1 from 10:00 UTC) and missing readings are 0 with mask 0;
* everything divided by the household's mean load over the 28 days before the
  cutoff (the same scale as the Level 0 features);
* context: target-day calendar, past temperature, household survey fields.

The model (a dilated 1D-CNN over the 96 slots with the 28 history days as input
channels, plus a learned same-slot weighted average as a skip path) outputs all
96 slots of D at once.  One architecture with fixed hyperparameters; early
stopping uses the most recent 42 days of the training period, never the
validation folds.  Its
predictions are mapped onto the frozen ``baseline_v1`` cache rows, so it is
scored on exactly the rows every other Track B model uses.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,4")  # shared server: idle GPUs only

import numpy as np
import pandas as pd

from utils.level0 import FIRST_DAY, FOLDS, HIST_START
from utils.modeling import DAY_AHEAD
from utils.trackb import ALL_FOLDS, THREADS, designs, eval_rows, log_runtime, save_pred


HISTORY_DAYS = 28
SCALE_STEPS = 28 * 96
GAP = DAY_AHEAD.gap_steps
MODEL_NAMES = {"cnn": "seq_cnn", "gru": "seq_gru"}


# ------------------------------------------------------------------ inputs

def household_scale(matrix: np.ndarray, days: np.ndarray, hh: np.ndarray) -> np.ndarray:
    """Mean load over the 28 days before each cutoff (NaN if <50% observed or ~0)."""
    cutoff = days * 96 - GAP
    window = matrix[cutoff[:, None] - SCALE_STEPS + np.arange(SCALE_STEPS)[None, :], hh[:, None]]
    count = np.isfinite(window).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.nansum(window, axis=1) / count
    scale[(count < .5 * SCALE_STEPS) | ~(scale > 1e-3)] = np.nan
    return scale


def history_grid(matrix, days, hh, xp=np):
    """Raw (B, 28, 96) grid of day D-k, slot s, plus mask; nothing at/after the cutoff.

    Works with numpy or torch (``xp``); both paths must agree (tested).
    """
    if xp is np:
        k = np.arange(1, HISTORY_DAYS + 1)[None, :, None]
        s = np.arange(96)[None, None, :]
    else:
        k = xp.arange(1, HISTORY_DAYS + 1, device=days.device)[None, :, None]
        s = xp.arange(96, device=days.device)[None, None, :]
    pos = (days[:, None, None] - k) * 96 + s
    values = matrix[pos, hh[:, None, None]]
    known = (pos < (days * 96 - GAP)[:, None, None]) & xp.isfinite(values)
    return xp.where(known, values, xp.zeros_like(values)), known


def context_table(arrays: dict, n_days: int) -> tuple[np.ndarray, list[str]]:
    """(day, household, feature) context known at each day's cutoff."""
    n_hh = arrays["matrix"].shape[1]
    days = np.arange(n_days)
    local = (HIST_START + pd.to_timedelta(days, unit="D") + pd.Timedelta(hours=12)).tz_convert("Europe/Berlin")
    wd, doy = np.asarray(local.weekday), np.asarray(local.dayofyear)
    calendar = np.concatenate([np.eye(7, dtype=np.float32)[wd], (wd >= 5)[:, None].astype(np.float32),
                               np.sin(2 * np.pi * doy / 365.25)[:, None],
                               np.cos(2 * np.pi * doy / 365.25)[:, None]], axis=1)
    # Past temperature at the household's station, up to the cutoff hour.
    temp = arrays["temperature"]
    cutoff_hour = np.clip(days * 24 - GAP // 4, 168, None)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t24 = np.stack([np.nanmean(temp[h - 24:h], axis=0) for h in cutoff_hour])
            t7d = np.stack([np.nanmean(temp[h - 168:h], axis=0) for h in cutoff_hour])
    weather = np.stack([np.nan_to_num(t24 / 10), np.nan_to_num((t24 - t7d) / 5), np.isnan(t24)], axis=2)
    # Static survey fields: one-hot categoricals, standardised numerics + missing flags.
    static, names = arrays["static"], arrays["static_names"]
    parts = []
    for j, name in enumerate(names):
        col = static[:, j]
        if name in ("pv_status", "building_type", "heatpump_type"):
            codes = np.nan_to_num(col, nan=-1)
            parts.append(np.stack([(codes == v) for v in (-1, 0, 1, 2)], axis=1).astype(np.float32))
        else:
            z = (col - np.nanmean(col)) / (np.nanstd(col) or 1)
            parts.append(np.stack([np.nan_to_num(z), np.isnan(col)], axis=1).astype(np.float32))
    static_feats = np.concatenate(parts, axis=1)
    table = np.concatenate([
        np.broadcast_to(calendar[:, None, :], (n_days, n_hh, calendar.shape[1])),
        weather.astype(np.float32),
        np.broadcast_to(static_feats[None, :, :], (n_days, n_hh, static_feats.shape[1])),
    ], axis=2).astype(np.float32)
    return table, ["calendar"] * calendar.shape[1] + ["weather"] * 3 + ["static"] * static_feats.shape[1]


def sample_inputs(arrays: dict, days: np.ndarray, hh: np.ndarray) -> dict[str, np.ndarray]:
    """Reference (numpy) construction of every model input for (day, household) pairs."""
    m = arrays["matrix"]
    scale = household_scale(m, days, hh)
    values, known = history_grid(m, days, hh)
    ctx, _ = context_table(arrays, int(days.max()) + 1)
    return {"grid": values / scale[:, None, None], "mask": known, "scale": scale,
            "context": ctx[days, hh]}


# ------------------------------------------------------------------- model

def build_model(n_context: int, channels: int = 128, arch: str = "cnn"):
    """1D-CNN over the 96 slots: the 28 history days (values + masks) are the
    input channels, so every output slot sees its own and neighbouring slots on
    all history days.  Fixed hyperparameters, no search."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
            self.norm = nn.GroupNorm(8, channels)

        def forward(self, x):
            return x + nn.functional.gelu(self.norm(self.conv(x)))

    class SeqCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv1d(2 * HISTORY_DAYS, channels, 5, padding=2)
            self.context = nn.Sequential(nn.Linear(n_context + 1, channels), nn.GELU(),
                                         nn.Linear(channels, channels))
            self.slot = nn.Parameter(torch.zeros(channels, 96))  # learned slot-of-day embedding
            self.blocks = nn.Sequential(*[Block(d) for d in (1, 2, 4, 8, 16, 1)])
            self.out = nn.Conv1d(channels, 1, 1)
            # Skip path: learned weights over the 28 history days, same slot, masked.
            self.day_logits = nn.Parameter(torch.linspace(1, -1, HISTORY_DAYS))
            self.skip_gain = nn.Parameter(torch.ones(1))

        def forward(self, grid, mask, context, log_scale):
            x = self.stem(torch.cat([grid, mask], dim=1))                        # (B, C, 96)
            x = x + self.slot + self.context(torch.cat([context, log_scale[:, None]], 1))[:, :, None]
            residual = self.out(self.blocks(x)).squeeze(1)
            w = torch.softmax(self.day_logits, 0)[None, :, None] * mask
            base = (w * grid).sum(1) / w.sum(1).clamp_min(1e-6)
            return self.skip_gain * base + residual

    class SeqGRU(nn.Module):
        """B4b: per-day encoder over the 96 slots (+ mask) -> 28 daily embeddings,
        oldest to newest -> 2-layer GRU -> final state + context -> 96-slot head.
        Same skip path as the CNN; fixed hyperparameters, no search."""

        def __init__(self, hidden: int = 128):
            super().__init__()
            self.day = nn.Sequential(nn.Linear(2 * 96, 256), nn.GELU(), nn.Linear(256, hidden), nn.GELU())
            self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=.1)
            self.context = nn.Sequential(nn.Linear(n_context + 1, 64), nn.GELU())
            self.head = nn.Sequential(nn.Linear(hidden + 64, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 96))
            self.day_logits = nn.Parameter(torch.linspace(1, -1, HISTORY_DAYS))
            self.skip_gain = nn.Parameter(torch.ones(1))

        def forward(self, grid, mask, context, log_scale):
            days = self.day(torch.cat([grid, mask], dim=2)).flip(1)   # row 0 is D-1: oldest first
            _, state = self.gru(days.float())
            ctx = self.context(torch.cat([context, log_scale[:, None]], dim=1))
            residual = self.head(torch.cat([state[-1], ctx], dim=1))
            w = torch.softmax(self.day_logits, 0)[None, :, None] * mask
            base = (w * grid).sum(1) / w.sum(1).clamp_min(1e-6)
            return self.skip_gain * base + residual

    return SeqCNN() if arch == "cnn" else SeqGRU()


class GpuData:
    """Matrix, scales and context on the GPU; batches gathered there."""

    def __init__(self, arrays: dict, device):
        import torch
        self.torch = torch
        m = arrays["matrix"]
        n_days = m.shape[0] // 96
        self.matrix = torch.from_numpy(m).to(device)
        days = np.repeat(np.arange(n_days), m.shape[1])
        hh = np.tile(np.arange(m.shape[1]), n_days)
        ok = days * 96 - GAP - SCALE_STEPS >= 0
        scale = np.full(len(days), np.nan, dtype=np.float32)
        for lo in range(0, ok.sum(), 200_000):  # bounded memory for the window gather
            idx = np.flatnonzero(ok)[lo:lo + 200_000]
            scale[idx] = household_scale(m, days[idx], hh[idx])
        self.scale = torch.from_numpy(scale.reshape(n_days, m.shape[1])).to(device)
        ctx, _ = context_table(arrays, n_days)
        self.context = torch.from_numpy(ctx).to(device)
        self.n_days, self.n_hh = n_days, m.shape[1]
        self.scale_np = scale.reshape(n_days, m.shape[1])

    def batch(self, days, hh):
        torch = self.torch
        scale = self.scale[days, hh]
        values, known = history_grid(self.matrix, days, hh, xp=torch)
        grid = values / scale[:, None, None]
        target_pos = days[:, None] * 96 + torch.arange(96, device=days.device)[None, :]
        target = self.matrix[target_pos, hh[:, None]] / scale[:, None]
        return grid, known.float(), self.context[days, hh], torch.log(scale), target


def training_pairs(data: GpuData, last_day: int) -> tuple[np.ndarray, np.ndarray]:
    """(day, hh) pairs with a valid scale and at least one observed target, day < last_day."""
    first = int((FIRST_DAY - HIST_START) / pd.Timedelta("1D"))
    days = np.arange(first, last_day)
    scale_ok = np.isfinite(data.scale_np[days])
    m = data.matrix.cpu().numpy()
    target_ok = np.isfinite(m[first * 96:last_day * 96].reshape(len(days), 96, -1)).any(axis=1)
    d, h = np.nonzero(scale_ok & target_ok)
    return days[d], h


def fit_predict(data: GpuData, train_days: np.ndarray, train_hh: np.ndarray, pred_days: np.ndarray,
                pred_hh: np.ndarray, seed: int = 42, arch: str = "cnn", max_epochs: int = 30, batch: int = 512,
                holdout_days: int = 42, patience: int = 4) -> tuple[np.ndarray, dict]:
    import torch
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    device = data.matrix.device
    # Early stopping on the most recent training days (still before the cutoff).
    hold = train_days >= train_days.max() - holdout_days + 1
    tr_d = torch.from_numpy(train_days[~hold]).to(device); tr_h = torch.from_numpy(train_hh[~hold]).to(device)
    ho_d = torch.from_numpy(train_days[hold]).to(device); ho_h = torch.from_numpy(train_hh[hold]).to(device)
    model = build_model(data.context.shape[2], arch=arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device=device).manual_seed(seed)

    def loss_on(days, hh):
        grid, mask, ctx, log_scale, target = data.batch(days, hh)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(grid, mask, ctx, log_scale)
        observed = torch.isfinite(target)
        err = torch.where(observed, pred.float() - torch.nan_to_num(target), torch.zeros_like(pred.float()))
        return (err ** 2).sum() / observed.sum().clamp_min(1)

    best, best_state, history, bad = np.inf, None, [], 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(tr_d), device=device, generator=gen)
        for i in range(0, len(perm) - batch + 1, batch):
            j = perm[i:i + batch]
            loss = loss_on(tr_d[j], tr_h[j])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            losses = [float(loss_on(ho_d[i:i + 4096], ho_h[i:i + 4096])) * min(4096, len(ho_d) - i)
                      for i in range(0, len(ho_d), 4096)]
        hold_mse = sum(losses) / len(ho_d)
        history.append(hold_mse)
        print(f"    epoch {epoch}: holdout mse {hold_mse:.4f}", flush=True)
        if hold_mse < best - 1e-4:
            best, bad = hold_mse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    out = []
    pd_t, ph_t = torch.from_numpy(pred_days).to(device), torch.from_numpy(pred_hh).to(device)
    with torch.no_grad():
        for i in range(0, len(pd_t), 4096):
            grid, mask, ctx, log_scale, _ = data.batch(pd_t[i:i + 4096], ph_t[i:i + 4096])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(grid, mask, ctx, log_scale).float().cpu())
    return torch.cat(out).numpy(), {"holdout_mse": history, "best_epoch": int(np.argmin(history)),
                                    "train_pairs": int((~hold).sum()), "holdout_pairs": int(hold.sum())}


def run(seed: int = 42, arch: str = "cnn") -> None:
    name = MODEL_NAMES[arch]
    import torch
    arrays, _, _ = designs()
    data = GpuData(arrays, torch.device("cuda:0"))
    for fold in ALL_FOLDS:
        started = time.perf_counter()
        start, _ = FOLDS[fold]
        last_day = int((DAY_AHEAD.cutoff(start).normalize() - HIST_START) / pd.Timedelta("1D"))
        train_days, train_hh = training_pairs(data, last_day)
        ev = eval_rows(fold)
        row_day = ((ev["index"].delivery_day - HIST_START) / pd.Timedelta("1D")).to_numpy().astype(np.int64)
        row_hh = ev["index"].hh.to_numpy().astype(np.int64)
        pairs, inverse = np.unique(np.stack([row_day, row_hh], axis=1), axis=0, return_inverse=True)
        scaled, info = fit_predict(data, train_days, train_hh, pairs[:, 0], pairs[:, 1], seed=seed, arch=arch)
        # Same scale definition as the cache (checked), so kWh = scaled * cache scale.
        own_scale = data.scale_np[row_day, row_hh]
        assert np.allclose(own_scale, ev["scale"], rtol=1e-3, equal_nan=False), "scale mismatch with cache"
        pred = np.maximum(scaled[inverse.ravel(), ev["index"].horizon_step.to_numpy() - 1], 0) * ev["scale"]
        save_pred(fold, name, pred)
        seconds = time.perf_counter() - started
        log_runtime("b4" if arch == "cnn" else "b4b", name, fold, seconds, **info)
        print(f"  {name} {fold}: {seconds:.0f}s, best epoch {info['best_epoch']}, "
              f"{info['train_pairs']:,} train pairs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch", default="cnn", choices=list(MODEL_NAMES))
    args = parser.parse_args()
    run(args.seed, args.arch)


if __name__ == "__main__":
    main()
