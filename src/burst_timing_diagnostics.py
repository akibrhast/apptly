from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iis3dwb_data import DEFAULT_OFFSET_BYTES, DEFAULT_SAMPLE_RATE_HZ, load_dataset
from window_artifact_diagnostics import rolling_dynamic_rms


AXES = ("X", "Y", "Z")
AXIS_COLORS = ("tab:blue", "tab:orange", "tab:green")


def collect_near_clip_rows(
    xyz: np.ndarray,
    threshold: int,
    chunk_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_parts: list[np.ndarray] = []
    axis_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []

    for start in range(0, xyz.shape[0], chunk_samples):
        block = xyz[start : start + chunk_samples]
        rows, axes = np.nonzero(np.abs(block.astype(np.int32)) >= threshold)
        if rows.size:
            sample_parts.append(rows.astype(np.int64) + start)
            axis_parts.append(axes.astype(np.int8))
            value_parts.append(np.asarray(block[rows, axes], dtype=np.int16))

    if not sample_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int16),
        )

    return np.concatenate(sample_parts), np.concatenate(axis_parts), np.concatenate(value_parts)


def binned_axis_counts(
    sample_idx: np.ndarray,
    axis_idx: np.ndarray,
    samples: int,
    sample_rate_hz: float,
    bin_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    bin_samples = int(round(bin_seconds * sample_rate_hz))
    bin_count = int(np.ceil(samples / bin_samples))
    counts = np.zeros((bin_count, 3), dtype=np.int64)
    bins = np.minimum(sample_idx // bin_samples, bin_count - 1).astype(np.int64)
    for axis in range(3):
        counts[:, axis] = np.bincount(bins[axis_idx == axis], minlength=bin_count)
    time_min = ((np.arange(bin_count) + 0.5) * bin_samples) / sample_rate_hz / 60.0
    return time_min, counts


def rolling_sum(values: np.ndarray, window_bins: int) -> np.ndarray:
    if window_bins <= 1:
        return values.astype(np.float64)
    kernel = np.ones(window_bins, dtype=np.float64)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def local_peak_indices(values: np.ndarray, min_prominence: float) -> np.ndarray:
    if values.size < 3:
        return np.empty(0, dtype=np.int64)
    candidates = np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1
    if candidates.size == 0:
        return candidates.astype(np.int64)
    floor = np.nanmedian(values)
    return candidates[(values[candidates] - floor) >= min_prominence].astype(np.int64)


def vector_rms(rms_xyz: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nansum(rms_xyz * rms_xyz, axis=1))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether widening RMS bursts follow near-clip event timing.")
    parser.add_argument("--input", type=Path, default=Path("iis3dwb_acc.dat"))
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--offset-bytes", type=int, default=DEFAULT_OFFSET_BYTES)
    parser.add_argument("--near-clip-threshold", type=int, default=30000)
    parser.add_argument("--winsorize-threshold", type=int, default=10000)
    parser.add_argument("--rms-window-seconds", type=float, default=5.0)
    parser.add_argument("--rms-hop-seconds", type=float, default=1.0)
    parser.add_argument("--event-bin-seconds", type=float, default=1.0)
    parser.add_argument("--event-rate-window-seconds", type=float, default=30.0)
    parser.add_argument("--chunk-samples", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/burst_timing_diagnostics.png"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/burst_timing_summary.txt"))
    args = parser.parse_args()

    dataset = load_dataset(args.input, args.sample_rate_hz, args.offset_bytes)
    xyz = dataset.xyz
    sample_idx, axis_idx, values = collect_near_clip_rows(xyz, args.near_clip_threshold, args.chunk_samples)

    rms_time_min, raw_rms_xyz = rolling_dynamic_rms(
        xyz, dataset.sample_rate_hz, args.rms_window_seconds, args.rms_hop_seconds
    )
    _, cleaned_rms_xyz = rolling_dynamic_rms(
        xyz,
        dataset.sample_rate_hz,
        args.rms_window_seconds,
        args.rms_hop_seconds,
        remove_near_clip=True,
        near_clip_threshold=args.near_clip_threshold,
    )
    _, winsorized_rms_xyz = rolling_dynamic_rms(
        xyz,
        dataset.sample_rate_hz,
        args.rms_window_seconds,
        args.rms_hop_seconds,
        winsorize_threshold=args.winsorize_threshold,
    )
    raw_vector = vector_rms(raw_rms_xyz)
    cleaned_vector = vector_rms(cleaned_rms_xyz)
    winsorized_vector = vector_rms(winsorized_rms_xyz)

    event_time_min, event_counts = binned_axis_counts(
        sample_idx, axis_idx, dataset.samples, dataset.sample_rate_hz, args.event_bin_seconds
    )
    event_total = event_counts.sum(axis=1)
    rate_window_bins = max(1, int(round(args.event_rate_window_seconds / args.event_bin_seconds)))
    event_rate = rolling_sum(event_total, rate_window_bins) / args.event_rate_window_seconds

    rms_bin_counts_time, rms_bin_counts = binned_axis_counts(
        sample_idx, axis_idx, dataset.samples, dataset.sample_rate_hz, args.rms_hop_seconds
    )
    interp_counts = np.interp(rms_time_min, rms_bin_counts_time, rolling_sum(rms_bin_counts.sum(axis=1), int(round(args.rms_window_seconds / args.rms_hop_seconds))))
    raw_event_corr = safe_corr(raw_vector, interp_counts)
    cleaned_event_corr = safe_corr(cleaned_vector, interp_counts)
    raw_cleaned_corr = safe_corr(raw_vector, cleaned_vector)

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    axes = axes.ravel()

    if sample_idx.size:
        max_points = 8000
        if sample_idx.size > max_points:
            pick = np.linspace(0, sample_idx.size - 1, max_points, dtype=np.int64)
        else:
            pick = np.arange(sample_idx.size)
        axes[0].scatter(
            sample_idx[pick] / dataset.sample_rate_hz / 60.0,
            axis_idx[pick],
            c=[AXIS_COLORS[int(i)] for i in axis_idx[pick]],
            s=4,
            alpha=0.55,
            label="Near-clip event",
        )
    axes[0].set_title("1. Near-clip event raster over time")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Axis")
    axes[0].set_yticks([0, 1, 2], AXES)
    axes[0].grid(True, alpha=0.25)

    for axis in range(3):
        axes[1].plot(event_time_min, event_counts[:, axis], color=AXIS_COLORS[axis], linewidth=0.65, label=AXES[axis])
    axes[1].plot(event_time_min, event_total, color="black", linewidth=0.9, label="Total")
    axes[1].set_title(f"2. Near-clip count per {args.event_bin_seconds:g}s bin")
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylabel("Near-clip samples")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(event_time_min, event_rate, color="tab:red", linewidth=0.9, label=f"{args.event_rate_window_seconds:g}s rolling rate")
    axes[2].set_title("3. Rolling near-clip event rate")
    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_ylabel("Near-clips per second")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", fontsize=8)

    event_order = np.argsort(sample_idx)
    event_seconds = sample_idx[event_order] / dataset.sample_rate_hz
    gaps = np.diff(event_seconds)
    gap_times_min = event_seconds[1:] / 60.0
    mask = (gaps > 0) & (gaps <= 20.0)
    axes[3].scatter(gap_times_min[mask], gaps[mask], s=3, color="tab:purple", alpha=0.45, label="Inter-event gap")
    axes[3].set_title("4. Inter-event gaps over time")
    axes[3].set_xlabel("Time (minutes)")
    axes[3].set_ylabel("Gap to previous near-clip (seconds)")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right", fontsize=8)

    ax_rms = axes[4]
    ax_count = ax_rms.twinx()
    raw_line = ax_rms.plot(rms_time_min, raw_vector, color="black", linewidth=0.85, label="Raw vector RMS")
    clean_line = ax_rms.plot(rms_time_min, cleaned_vector, color="tab:red", linewidth=0.85, label="Near-clips removed")
    win_line = ax_rms.plot(rms_time_min, winsorized_vector, color="tab:purple", linewidth=0.85, label="Winsorized")
    count_line = ax_count.plot(rms_time_min, interp_counts, color="tab:green", linewidth=0.65, alpha=0.75, label="Near-clips in RMS window")
    ax_rms.set_title("5. RMS envelope overlaid with near-clip count")
    ax_rms.set_xlabel("Time (minutes)")
    ax_rms.set_ylabel("Dynamic vector RMS (raw counts)")
    ax_count.set_ylabel("Near-clip samples in RMS window")
    ax_rms.grid(True, alpha=0.25)
    lines = raw_line + clean_line + win_line + count_line
    ax_rms.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8)

    delta = raw_vector - cleaned_vector
    axes[5].plot(rms_time_min, delta, color="tab:blue", linewidth=0.85, label="Raw RMS - cleaned RMS")
    axes[5].plot(rms_time_min, interp_counts, color="tab:green", linewidth=0.65, alpha=0.75, label="Near-clips in RMS window")
    axes[5].set_title("6. RMS excess versus near-clip count")
    axes[5].set_xlabel("Time (minutes)")
    axes[5].set_ylabel("Raw-count units / event count")
    axes[5].grid(True, alpha=0.25)
    axes[5].legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Burst Timing Diagnostics\n"
        "If the widening RMS envelope follows near-clip timing and disappears after cleaning, it is a capture artifact rather than vibration.",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)

    peak_prominence = max(1.0, np.nanstd(raw_vector) * 0.35)
    peaks = local_peak_indices(raw_vector, peak_prominence)
    peak_times_min = rms_time_min[peaks]
    peak_gaps_min = np.diff(peak_times_min)
    event_peak_prominence = max(1.0, np.nanstd(event_rate) * 0.35)
    event_peaks = local_peak_indices(event_rate, event_peak_prominence)
    event_peak_times_min = event_time_min[event_peaks]
    event_peak_gaps_min = np.diff(event_peak_times_min)

    lines = [
        f"input={dataset.path}",
        f"samples={dataset.samples}",
        f"duration_minutes={dataset.duration_minutes:.6f}",
        f"sample_rate_hz={dataset.sample_rate_hz:.3f}",
        f"near_clip_threshold={args.near_clip_threshold}",
        f"near_clip_events={sample_idx.size}",
        f"raw_rms_vs_near_clip_window_count_corr={raw_event_corr:.6f}",
        f"cleaned_rms_vs_near_clip_window_count_corr={cleaned_event_corr:.6f}",
        f"raw_rms_vs_cleaned_rms_corr={raw_cleaned_corr:.6f}",
        f"raw_rms_peak_count={peaks.size}",
        f"raw_rms_peak_gap_minutes_median={np.nanmedian(peak_gaps_min) if peak_gaps_min.size else float('nan'):.6f}",
        f"raw_rms_peak_gap_minutes_first={peak_gaps_min[0] if peak_gaps_min.size else float('nan'):.6f}",
        f"raw_rms_peak_gap_minutes_last={peak_gaps_min[-1] if peak_gaps_min.size else float('nan'):.6f}",
        f"near_clip_rate_peak_count={event_peaks.size}",
        f"near_clip_rate_peak_gap_minutes_median={np.nanmedian(event_peak_gaps_min) if event_peak_gaps_min.size else float('nan'):.6f}",
        f"near_clip_rate_peak_gap_minutes_first={event_peak_gaps_min[0] if event_peak_gaps_min.size else float('nan'):.6f}",
        f"near_clip_rate_peak_gap_minutes_last={event_peak_gaps_min[-1] if event_peak_gaps_min.size else float('nan'):.6f}",
        f"output={args.output}",
    ]
    args.summary_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
