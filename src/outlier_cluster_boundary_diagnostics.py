from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iis3dwb_data import DEFAULT_OFFSET_BYTES, DEFAULT_SAMPLE_RATE_HZ, load_dataset
from window_artifact_diagnostics import rolling_dynamic_rms


@dataclass(frozen=True)
class Boundary:
    name: str
    bytes: int


BOUNDARIES = (
    Boundary("sd_sector_512", 512),
    Boundary("iis3dwb_payload_1536", 1536),
    Boundary("estimated_filex_write_48128", 48128),
    Boundary("fat_cluster_32768", 32768),
    Boundary("fat_cluster_65536", 65536),
    Boundary("fat_cluster_131072", 131072),
)


def collect_outlier_rows(
    xyz: np.ndarray,
    threshold: int,
    chunk_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_parts: list[np.ndarray] = []
    axis_count_parts: list[np.ndarray] = []
    for start in range(0, xyz.shape[0], chunk_samples):
        block = xyz[start : start + chunk_samples]
        mask = np.abs(block.astype(np.int32)) >= threshold
        rows = np.nonzero(np.any(mask, axis=1))[0]
        if rows.size:
            row_parts.append(rows.astype(np.int64) + start)
            axis_count_parts.append(mask[rows].sum(axis=1).astype(np.int8))
    if not row_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)
    return np.concatenate(row_parts), np.concatenate(axis_count_parts)


def collect_outlier_axis_events(
    xyz: np.ndarray,
    threshold: int,
    chunk_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_parts: list[np.ndarray] = []
    axis_parts: list[np.ndarray] = []
    for start in range(0, xyz.shape[0], chunk_samples):
        block = xyz[start : start + chunk_samples]
        rows, axes = np.nonzero(np.abs(block.astype(np.int32)) >= threshold)
        if rows.size:
            sample_parts.append(rows.astype(np.int64) + start)
            axis_parts.append(axes.astype(np.int8))
    if not sample_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)
    return np.concatenate(sample_parts), np.concatenate(axis_parts)


def binned_counts(sample_idx: np.ndarray, sample_count: int, sample_rate_hz: float, bin_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    bin_samples = int(round(bin_seconds * sample_rate_hz))
    n_bins = int(np.ceil(sample_count / bin_samples))
    bins = np.minimum(sample_idx // bin_samples, n_bins - 1).astype(np.int64)
    counts = np.bincount(bins, minlength=n_bins)
    time_min = ((np.arange(n_bins) + 0.5) * bin_samples) / sample_rate_hz / 60.0
    return time_min, counts


def moving_sum(values: np.ndarray, window_bins: int) -> np.ndarray:
    if window_bins <= 1:
        return values.astype(np.float64)
    kernel = np.ones(window_bins, dtype=np.float64)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def select_peak_indices(values: np.ndarray, threshold: float, min_separation_bins: int) -> np.ndarray:
    if values.size < 3:
        return np.empty(0, dtype=np.int64)
    candidates = np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]) & (values[1:-1] >= threshold))[0] + 1
    if candidates.size == 0:
        return candidates.astype(np.int64)
    order = candidates[np.argsort(values[candidates])[::-1]]
    selected: list[int] = []
    for idx in order:
        if all(abs(int(idx) - prev) >= min_separation_bins for prev in selected):
            selected.append(int(idx))
    selected.sort()
    return np.asarray(selected, dtype=np.int64)


def boundary_distance(byte_offset: int, boundary: int) -> tuple[int, int]:
    mod = byte_offset % boundary
    return mod, min(mod, boundary - mod)


def cluster_rows_for_peak(
    outlier_rows: np.ndarray,
    peak_time_seconds: float,
    half_width_seconds: float,
    sample_rate_hz: float,
) -> np.ndarray:
    center = int(round(peak_time_seconds * sample_rate_hz))
    half_width = int(round(half_width_seconds * sample_rate_hz))
    lo = np.searchsorted(outlier_rows, center - half_width, side="left")
    hi = np.searchsorted(outlier_rows, center + half_width, side="right")
    return outlier_rows[lo:hi]


def cluster_axis_events_for_peak(
    event_samples: np.ndarray,
    event_axes: np.ndarray,
    peak_time_seconds: float,
    half_width_seconds: float,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    center = int(round(peak_time_seconds * sample_rate_hz))
    half_width = int(round(half_width_seconds * sample_rate_hz))
    lo = np.searchsorted(event_samples, center - half_width, side="left")
    hi = np.searchsorted(event_samples, center + half_width, side="right")
    return event_samples[lo:hi], event_axes[lo:hi]


def write_cluster_csv(
    path: Path,
    clusters: list[dict[str, object]],
) -> None:
    if not clusters:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(clusters[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clusters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare outlier cluster timing against FIFO, SD, and file-boundary offsets.")
    parser.add_argument("--input", type=Path, default=Path("iis3dwb_acc.dat"))
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--offset-bytes", type=int, default=DEFAULT_OFFSET_BYTES)
    parser.add_argument("--threshold", type=int, default=10000)
    parser.add_argument("--bin-seconds", type=float, default=1.0)
    parser.add_argument("--smooth-seconds", type=float, default=5.0)
    parser.add_argument("--cluster-half-width-seconds", type=float, default=2.5)
    parser.add_argument("--cluster-percentile", type=float, default=90.0)
    parser.add_argument("--min-peak-separation-seconds", type=float, default=8.0)
    parser.add_argument("--chunk-samples", type=int, default=2_000_000)
    parser.add_argument("--output-plot", type=Path, default=Path("outputs/outlier_cluster_boundary_diagnostics.png"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/outlier_cluster_boundary_diagnostics.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("outputs/outlier_cluster_boundary_summary.txt"))
    args = parser.parse_args()

    dataset = load_dataset(args.input, args.sample_rate_hz, args.offset_bytes)
    outlier_rows, axis_counts = collect_outlier_rows(dataset.xyz, args.threshold, args.chunk_samples)
    event_samples, event_axes = collect_outlier_axis_events(dataset.xyz, args.threshold, args.chunk_samples)
    if outlier_rows.size == 0:
        raise SystemExit(f"No outlier rows found at abs(sample) >= {args.threshold}")

    bin_time_min, counts = binned_counts(outlier_rows, dataset.samples, dataset.sample_rate_hz, args.bin_seconds)
    smooth_bins = max(1, int(round(args.smooth_seconds / args.bin_seconds)))
    smoothed = moving_sum(counts, smooth_bins)
    peak_threshold = float(np.percentile(smoothed, args.cluster_percentile))
    min_separation_bins = max(1, int(round(args.min_peak_separation_seconds / args.bin_seconds)))
    peak_idx = select_peak_indices(smoothed, peak_threshold, min_separation_bins)
    peak_times_min = bin_time_min[peak_idx]

    clusters: list[dict[str, object]] = []
    prev_time_min: float | None = None
    for cluster_id, idx in enumerate(peak_idx, start=1):
        time_min = float(bin_time_min[idx])
        time_seconds = time_min * 60.0
        rows = cluster_rows_for_peak(outlier_rows, time_seconds, args.cluster_half_width_seconds, dataset.sample_rate_hz)
        event_rows, event_axis_idx = cluster_axis_events_for_peak(
            event_samples,
            event_axes,
            time_seconds,
            args.cluster_half_width_seconds,
            dataset.sample_rate_hz,
        )
        if rows.size:
            center_sample = int(round(float(rows.mean())))
            cluster_outlier_rows = int(rows.size)
        else:
            center_sample = int(round(time_seconds * dataset.sample_rate_hz))
            cluster_outlier_rows = 0
        byte_offset = int(args.offset_bytes + center_sample * 6)
        event_byte_offsets = args.offset_bytes + event_rows * 6 + event_axis_idx * 2
        event_count = int(event_byte_offsets.size)
        event_near_512 = int(((event_byte_offsets % 512) <= 4).sum() + ((event_byte_offsets % 512) >= 508).sum())
        event_near_1536 = int(((event_byte_offsets % 1536) <= 4).sum() + ((event_byte_offsets % 1536) >= 1532).sum())
        event_near_write = int(
            ((event_byte_offsets % 48128) <= 64).sum() + ((event_byte_offsets % 48128) >= (48128 - 64)).sum()
        )
        row = {
            "cluster_id": cluster_id,
            "time_min": f"{time_min:.6f}",
            "gap_from_previous_min": "" if prev_time_min is None else f"{time_min - prev_time_min:.6f}",
            "smoothed_count": f"{float(smoothed[idx]):.3f}",
            "outlier_rows_in_cluster_window": cluster_outlier_rows,
            "outlier_axis_events_in_cluster_window": event_count,
            "axis_events_within_4_bytes_of_512_boundary": event_near_512,
            "axis_events_within_4_bytes_of_1536_boundary": event_near_1536,
            "axis_events_within_64_bytes_of_48128_write_boundary": event_near_write,
            "center_sample": center_sample,
            "file_byte_offset": byte_offset,
            "fifo_payload_number": center_sample // 256,
            "fifo_sample_position": center_sample % 256,
            "estimated_filex_write_number": byte_offset // 48128,
        }
        for boundary in BOUNDARIES:
            mod, dist = boundary_distance(byte_offset, boundary.bytes)
            row[f"{boundary.name}_mod"] = mod
            row[f"{boundary.name}_nearest_boundary_distance"] = dist
        clusters.append(row)
        prev_time_min = time_min

    write_cluster_csv(args.output_csv, clusters)

    t_rms, rms_xyz = rolling_dynamic_rms(dataset.xyz, dataset.sample_rate_hz, 5.0, 5.0)
    vector_rms = np.sqrt(np.nansum(rms_xyz * rms_xyz, axis=1))
    gap_values = np.array([float(c["gap_from_previous_min"]) for c in clusters[1:]], dtype=np.float64)
    gap_times = np.array([float(c["time_min"]) for c in clusters[1:]], dtype=np.float64)
    boundary_dist_512 = np.array([float(c["sd_sector_512_nearest_boundary_distance"]) for c in clusters], dtype=np.float64)
    boundary_dist_48128 = np.array([float(c["estimated_filex_write_48128_nearest_boundary_distance"]) for c in clusters], dtype=np.float64)

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(t_rms, vector_rms, color="black", linewidth=0.8, label="5s rolling vector RMS")
    for time in peak_times_min:
        axes[0].axvline(time, color="tab:red", alpha=0.22, linewidth=0.8)
    axes[0].set_title("1. Rolling RMS with detected outlier-cluster centers")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Dynamic vector RMS (raw counts)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(bin_time_min, counts, color="tab:gray", linewidth=0.55, alpha=0.6, label=f"{args.bin_seconds:g}s outlier rows")
    axes[1].plot(bin_time_min, smoothed, color="tab:red", linewidth=0.9, label=f"{args.smooth_seconds:g}s moving sum")
    axes[1].scatter(peak_times_min, smoothed[peak_idx], color="black", s=14, label="Detected clusters")
    axes[1].axhline(peak_threshold, color="tab:blue", linestyle="--", linewidth=0.8, label=f"{args.cluster_percentile:g}th percentile")
    axes[1].set_title(f"2. Outlier row rate, abs(sample)>={args.threshold}")
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylabel("Outlier rows")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(gap_times, gap_values, marker="o", color="tab:purple", linewidth=0.9, markersize=3)
    axes[2].set_title("3. Gap between detected outlier clusters")
    axes[2].set_xlabel("Time of cluster (minutes)")
    axes[2].set_ylabel("Gap from previous cluster (minutes)")
    axes[2].grid(True, alpha=0.25)

    axes[3].scatter(peak_times_min, boundary_dist_512, s=16, color="tab:green", alpha=0.8, label="distance to 512B boundary")
    axes[3].scatter(peak_times_min, boundary_dist_48128, s=16, color="tab:orange", alpha=0.8, label="distance to 48128B write boundary")
    axes[3].set_title("4. Cluster center distance to storage boundaries")
    axes[3].set_xlabel("Time (minutes)")
    axes[3].set_ylabel("Nearest boundary distance (bytes)")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right", fontsize=8)

    fifo_pos = np.array([int(c["fifo_sample_position"]) for c in clusters], dtype=np.int64)
    axes[4].hist(fifo_pos, bins=np.arange(257), color="tab:blue", alpha=0.85)
    axes[4].set_title("5. FIFO sample position of cluster centers")
    axes[4].set_xlabel("Sample position inside 256-sample FIFO payload")
    axes[4].set_ylabel("Cluster count")
    axes[4].grid(True, axis="y", alpha=0.25)

    filex_mod = np.array([int(c["estimated_filex_write_48128_mod"]) for c in clusters], dtype=np.int64)
    axes[5].hist(filex_mod, bins=80, color="tab:orange", alpha=0.85)
    axes[5].set_title("6. Cluster center byte offset modulo estimated 48128B FileX write")
    axes[5].set_xlabel("Byte offset modulo 48128")
    axes[5].set_ylabel("Cluster count")
    axes[5].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Outlier Cluster Timing vs FIFO/SD/File Boundaries\n"
        "This tests whether the widening RMS peaks follow clusters whose centers align with stream/storage boundaries.",
        fontsize=13,
    )
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=160)

    gap_first = float(gap_values[0]) if gap_values.size else float("nan")
    gap_last = float(gap_values[-1]) if gap_values.size else float("nan")
    gap_median = float(np.median(gap_values)) if gap_values.size else float("nan")
    gap_time_corr = float(np.corrcoef(gap_times, gap_values)[0, 1]) if gap_values.size > 2 else float("nan")
    gap_slope = float(np.polyfit(gap_times, gap_values, 1)[0]) if gap_values.size > 2 else float("nan")
    near_512 = int((boundary_dist_512 <= 4).sum())
    near_write = int((boundary_dist_48128 <= 64).sum())
    cluster_event_near_512 = sum(int(c["axis_events_within_4_bytes_of_512_boundary"]) for c in clusters)
    cluster_event_count = sum(int(c["outlier_axis_events_in_cluster_window"]) for c in clusters)
    summary_lines = [
        f"input={dataset.path}",
        f"samples={dataset.samples}",
        f"duration_minutes={dataset.duration_minutes:.6f}",
        f"threshold={args.threshold}",
        f"outlier_rows={outlier_rows.size}",
        f"bin_seconds={args.bin_seconds}",
        f"smooth_seconds={args.smooth_seconds}",
        f"cluster_percentile={args.cluster_percentile}",
        f"detected_clusters={len(clusters)}",
        f"cluster_gap_minutes_first={gap_first:.6f}",
        f"cluster_gap_minutes_median={gap_median:.6f}",
        f"cluster_gap_minutes_last={gap_last:.6f}",
        f"cluster_gap_vs_time_corr={gap_time_corr:.6f}",
        f"cluster_gap_slope_min_per_min={gap_slope:.9f}",
        f"clusters_within_4_bytes_of_512_boundary={near_512}",
        f"clusters_within_64_bytes_of_estimated_48128B_write_boundary={near_write}",
        f"axis_events_inside_clusters={cluster_event_count}",
        f"axis_events_inside_clusters_within_4_bytes_of_512_boundary={cluster_event_near_512}",
        f"output_plot={args.output_plot}",
        f"output_csv={args.output_csv}",
    ]
    args.output_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_plot}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_summary}")


if __name__ == "__main__":
    main()
