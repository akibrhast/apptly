from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iis3dwb_data import DEFAULT_SAMPLE_RATE_HZ, load_dataset


AXES = ("X", "Y", "Z")
AXIS_COLORS = {
    "X": "tab:blue",
    "Y": "tab:orange",
    "Z": "tab:green",
}


def rolling_dynamic_rms(
    xyz: np.ndarray,
    sample_rate_hz: float,
    window_seconds: float,
    hop_seconds: float,
    offset_seconds: float = 0.0,
    remove_near_clip: bool = False,
    near_clip_threshold: int = 30000,
    winsorize_threshold: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    window_samples = int(round(window_seconds * sample_rate_hz))
    hop_samples = int(round(hop_seconds * sample_rate_hz))
    start_sample = int(round(offset_seconds * sample_rate_hz))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    starts = np.arange(start_sample, xyz.shape[0] - window_samples + 1, hop_samples, dtype=np.int64)
    rms = np.empty((starts.size, 3), dtype=np.float64)

    for idx, start in enumerate(starts):
        block = xyz[start : start + window_samples].astype(np.float64)
        if remove_near_clip:
            clipped_rows = np.any(np.abs(block) >= near_clip_threshold, axis=1)
            block = block[~clipped_rows]
        if winsorize_threshold is not None:
            block = np.clip(block, -winsorize_threshold, winsorize_threshold)
        if block.shape[0] == 0:
            rms[idx] = np.nan
            continue
        centered = block - block.mean(axis=0, keepdims=True)
        rms[idx] = np.sqrt(np.mean(centered * centered, axis=0))

    time_min = (starts + window_samples / 2.0) / sample_rate_hz / 60.0
    return time_min, rms


def near_clip_counts(
    xyz: np.ndarray,
    sample_rate_hz: float,
    window_seconds: float,
    hop_seconds: float,
    threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    window_samples = int(round(window_seconds * sample_rate_hz))
    hop_samples = int(round(hop_seconds * sample_rate_hz))
    starts = np.arange(0, xyz.shape[0] - window_samples + 1, hop_samples, dtype=np.int64)
    counts = np.empty((starts.size, 3), dtype=np.int64)
    for idx, start in enumerate(starts):
        block = xyz[start : start + window_samples]
        counts[idx] = (np.abs(block.astype(np.int32)) >= threshold).sum(axis=0)
    time_min = (starts + window_samples / 2.0) / sample_rate_hz / 60.0
    return time_min, counts


def plot_xyz(ax, time_min: np.ndarray, rms: np.ndarray, title: str, ylabel: str = "Dynamic RMS (raw counts)") -> None:
    for axis_idx, axis_name in enumerate(AXES):
        ax.plot(time_min, rms[:, axis_idx], linewidth=0.85, color=AXIS_COLORS[axis_name], label=axis_name)
    ax.set_title(title)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create rolling-RMS window artifact diagnostic plots.")
    parser.add_argument("--input", type=Path, default=Path("iis3dwb_acc.dat"))
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--offset-bytes", type=int, default=4)
    parser.add_argument("--near-clip-threshold", type=int, default=30000)
    parser.add_argument("--winsorize-threshold", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("outputs/window_artifact_diagnostics.png"))
    args = parser.parse_args()

    dataset = load_dataset(args.input, args.sample_rate_hz, args.offset_bytes)
    xyz = dataset.xyz

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    axes = axes.ravel()

    t, y = rolling_dynamic_rms(xyz, dataset.sample_rate_hz, 5.0, 5.0)
    plot_xyz(axes[0], t, y, "1. 5s window / 5s hop")

    t, y = rolling_dynamic_rms(xyz, dataset.sample_rate_hz, 5.0, 1.0)
    plot_xyz(axes[1], t, y, "2. 5s window / 1s hop")

    t, y = rolling_dynamic_rms(xyz, dataset.sample_rate_hz, 5.0, 5.0, offset_seconds=2.5)
    plot_xyz(axes[2], t, y, "3. 5s window / 5s hop / +2.5s start offset")

    for window_seconds, color in [(4.0, "tab:blue"), (6.0, "tab:orange"), (7.0, "tab:green")]:
        t, y = rolling_dynamic_rms(xyz, dataset.sample_rate_hz, window_seconds, window_seconds)
        vector = np.sqrt(np.sum(y * y, axis=1))
        axes[3].plot(t, vector, linewidth=0.9, color=color, label=f"{window_seconds:g}s window / {window_seconds:g}s hop")
    axes[3].set_title("4. Vector RMS with 4s, 6s, and 7s windows")
    axes[3].set_xlabel("Time (minutes)")
    axes[3].set_ylabel("Dynamic vector RMS (raw counts)")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right", fontsize=8)

    t, normal = rolling_dynamic_rms(xyz, dataset.sample_rate_hz, 5.0, 5.0)
    _, removed = rolling_dynamic_rms(
        xyz,
        dataset.sample_rate_hz,
        5.0,
        5.0,
        remove_near_clip=True,
        near_clip_threshold=args.near_clip_threshold,
    )
    _, winsorized = rolling_dynamic_rms(
        xyz,
        dataset.sample_rate_hz,
        5.0,
        5.0,
        winsorize_threshold=args.winsorize_threshold,
    )
    axes[4].plot(t, np.sqrt(np.sum(normal * normal, axis=1)), color="black", linewidth=0.9, label="Normal")
    axes[4].plot(
        t,
        np.sqrt(np.nansum(removed * removed, axis=1)),
        color="tab:red",
        linewidth=0.9,
        label=f"Rows with abs(sample)>={args.near_clip_threshold} removed",
    )
    axes[4].plot(
        t,
        np.sqrt(np.sum(winsorized * winsorized, axis=1)),
        color="tab:purple",
        linewidth=0.9,
        label=f"Winsorized to +/-{args.winsorize_threshold}",
    )
    axes[4].set_title("5. 5s vector RMS: normal vs outlier-reduced")
    axes[4].set_xlabel("Time (minutes)")
    axes[4].set_ylabel("Dynamic vector RMS (raw counts)")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(loc="upper right", fontsize=8)

    t, counts = near_clip_counts(xyz, dataset.sample_rate_hz, 5.0, 5.0, args.near_clip_threshold)
    for axis_idx, axis_name in enumerate(AXES):
        axes[5].plot(t, counts[:, axis_idx], linewidth=0.85, color=AXIS_COLORS[axis_name], label=axis_name)
    axes[5].plot(t, counts.sum(axis=1), linewidth=1.1, color="black", label="Total")
    axes[5].set_title(f"6. Near-clip sample count per 5s window, threshold={args.near_clip_threshold}")
    axes[5].set_xlabel("Time (minutes)")
    axes[5].set_ylabel("Samples per window")
    axes[5].grid(True, alpha=0.25)
    axes[5].legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Rolling RMS Window Artifact Diagnostics\n"
        "If peaks move with offsets/window sizes, suspect window/alias artifacts. "
        "If peaks align in absolute time, actual captured events exist. "
        "If the pattern disappears after outlier reduction, RMS is dominated by corrupt/near-clipped samples.",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
