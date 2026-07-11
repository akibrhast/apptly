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

from iis3dwb_data import DEFAULT_OFFSET_BYTES, DEFAULT_SAMPLE_RATE_HZ, dynamic_vector_rms, load_dataset, welch_psd
from window_artifact_diagnostics import rolling_dynamic_rms


AXES = ("X", "Y", "Z")
BANDS = (
    (0.0, 20.0),
    (20.0, 100.0),
    (100.0, 500.0),
    (500.0, 2000.0),
    (2000.0, DEFAULT_SAMPLE_RATE_HZ / 2.0),
)
COLORS = {
    "raw": "black",
    "winsorized": "tab:purple",
    "cleaned_removed": "tab:red",
}


@dataclass(frozen=True)
class VersionSpec:
    name: str
    path: Path


@dataclass
class VersionMetrics:
    name: str
    path: Path
    samples: int
    duration_minutes: float
    axis_mean: np.ndarray
    axis_std: np.ndarray
    axis_dynamic_rms: np.ndarray
    vector_dynamic_rms: float
    p95_abs: float
    p99_abs: float
    p999_abs: float
    max_abs: int
    rows_ge_10000: int
    rows_ge_15000: int
    rows_ge_20000: int
    rows_ge_25000: int
    rows_ge_30000: int
    rolling_time_min: np.ndarray
    rolling_vector_rms: np.ndarray
    quartile_vector_rms: np.ndarray
    spectrum_freq_hz: np.ndarray
    spectrum_amp: np.ndarray
    quartile_band_rms: np.ndarray


def row_count_at_threshold(xyz: np.ndarray, threshold: int, chunk_samples: int = 2_000_000) -> int:
    count = 0
    for start in range(0, xyz.shape[0], chunk_samples):
        block = xyz[start : start + chunk_samples].astype(np.int32)
        count += int(np.any(np.abs(block) >= threshold, axis=1).sum())
    return count


def abs_percentiles(xyz: np.ndarray) -> tuple[float, float, float, int]:
    abs_values = np.abs(xyz.astype(np.int32)).reshape(-1)
    return (
        float(np.percentile(abs_values, 95)),
        float(np.percentile(abs_values, 99)),
        float(np.percentile(abs_values, 99.9)),
        int(abs_values.max()),
    )


def representative_vector_spectrum(xyz: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    nfft = min(1_048_576, 1 << int(np.floor(np.log2(xyz.shape[0]))))
    start = max(0, (xyz.shape[0] - nfft) // 2)
    segment = xyz[start : start + nfft].astype(np.float64)
    segment -= segment.mean(axis=0, keepdims=True)
    window = np.hanning(nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate_hz)
    amp_sq = np.zeros(freqs.size, dtype=np.float64)
    scale = 2.0 / np.sum(window)
    for axis_idx in range(3):
        amp = scale * np.abs(np.fft.rfft(segment[:, axis_idx] * window))
        amp_sq += amp * amp
    return freqs, np.sqrt(amp_sq)


def quartile_band_rms(xyz: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    edges = np.linspace(0, xyz.shape[0], 5, dtype=np.int64)
    output = np.zeros((4, len(BANDS)), dtype=np.float64)
    for quartile_idx in range(4):
        start = edges[quartile_idx]
        end = edges[quartile_idx + 1]
        for axis_idx in range(3):
            freqs, psd = welch_psd(xyz[start:end, axis_idx], sample_rate_hz, nperseg=65536, max_segments=96)
            for band_idx, (low, high) in enumerate(BANDS):
                band_high = min(high, sample_rate_hz / 2.0)
                mask = (freqs >= low) & (freqs < band_high)
                if mask.sum() > 1:
                    output[quartile_idx, band_idx] += float(np.trapezoid(psd[mask], freqs[mask]))
    return np.sqrt(output)


def compute_metrics(spec: VersionSpec, sample_rate_hz: float, offset_bytes: int) -> VersionMetrics:
    dataset = load_dataset(spec.path, sample_rate_hz, offset_bytes)
    xyz = dataset.xyz
    axis_dynamic, vector_dynamic = dynamic_vector_rms(xyz)
    p95, p99, p999, max_abs = abs_percentiles(xyz)
    rolling_t, rolling_xyz = rolling_dynamic_rms(xyz, sample_rate_hz, 5.0, 5.0)
    rolling_vector = np.sqrt(np.nansum(rolling_xyz * rolling_xyz, axis=1))
    quartile_edges = np.linspace(0, xyz.shape[0], 5, dtype=np.int64)
    quartile_rms = np.empty(4, dtype=np.float64)
    for idx in range(4):
        _, quartile_rms[idx] = dynamic_vector_rms(xyz[quartile_edges[idx] : quartile_edges[idx + 1]])
    spectrum_freq, spectrum_amp = representative_vector_spectrum(xyz, sample_rate_hz)
    return VersionMetrics(
        name=spec.name,
        path=spec.path,
        samples=dataset.samples,
        duration_minutes=dataset.duration_minutes,
        axis_mean=xyz.astype(np.float64).mean(axis=0),
        axis_std=xyz.astype(np.float64).std(axis=0),
        axis_dynamic_rms=axis_dynamic,
        vector_dynamic_rms=vector_dynamic,
        p95_abs=p95,
        p99_abs=p99,
        p999_abs=p999,
        max_abs=max_abs,
        rows_ge_10000=row_count_at_threshold(xyz, 10000),
        rows_ge_15000=row_count_at_threshold(xyz, 15000),
        rows_ge_20000=row_count_at_threshold(xyz, 20000),
        rows_ge_25000=row_count_at_threshold(xyz, 25000),
        rows_ge_30000=row_count_at_threshold(xyz, 30000),
        rolling_time_min=rolling_t,
        rolling_vector_rms=rolling_vector,
        quartile_vector_rms=quartile_rms,
        spectrum_freq_hz=spectrum_freq,
        spectrum_amp=spectrum_amp,
        quartile_band_rms=quartile_band_rms(xyz, sample_rate_hz),
    )


def write_summary_csv(metrics: list[VersionMetrics], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "version",
                "path",
                "samples",
                "duration_minutes",
                "axis_mean_x",
                "axis_mean_y",
                "axis_mean_z",
                "axis_dynamic_rms_x",
                "axis_dynamic_rms_y",
                "axis_dynamic_rms_z",
                "vector_dynamic_rms",
                "p95_abs",
                "p99_abs",
                "p999_abs",
                "max_abs",
                "rows_ge_10000",
                "rows_ge_15000",
                "rows_ge_20000",
                "rows_ge_25000",
                "rows_ge_30000",
            ]
        )
        for item in metrics:
            writer.writerow(
                [
                    item.name,
                    item.path,
                    item.samples,
                    f"{item.duration_minutes:.6f}",
                    *[f"{v:.9g}" for v in item.axis_mean],
                    *[f"{v:.9g}" for v in item.axis_dynamic_rms],
                    f"{item.vector_dynamic_rms:.9g}",
                    f"{item.p95_abs:.9g}",
                    f"{item.p99_abs:.9g}",
                    f"{item.p999_abs:.9g}",
                    item.max_abs,
                    item.rows_ge_10000,
                    item.rows_ge_15000,
                    item.rows_ge_20000,
                    item.rows_ge_25000,
                    item.rows_ge_30000,
                ]
            )


def write_markdown(metrics: list[VersionMetrics], output: Path, image_path: Path, csv_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Raw vs Winsorized vs Cleaned Comparison")
    lines.append("")
    lines.append("This report compares the original raw capture with the derived artifact-suppressed data files.")
    lines.append("")
    lines.append(f"- Figure: `{image_path}`")
    lines.append(f"- CSV: `{csv_path}`")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append(
        "| version | samples | duration min | vector RMS | p99.9 abs | max abs | rows >=10000 | rows >=30000 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in metrics:
        lines.append(
            f"| {item.name} | {item.samples} | {item.duration_minutes:.6f} | "
            f"{item.vector_dynamic_rms:.3f} | {item.p999_abs:.1f} | {item.max_abs} | "
            f"{item.rows_ge_10000} | {item.rows_ge_30000} |"
        )
    lines.append("")
    lines.append("## Current Interpretation")
    lines.append("")
    lines.append(
        "The winsorized file suppresses the broad high-amplitude outlier population while preserving the original timebase. "
        "The cleaned-removed file only removes rows above the configured near-clip threshold, so lower-amplitude outliers can remain."
    )
    lines.append("")
    lines.append("For time-aligned vibration visualization, prefer `derived_data/iis3dwb_acc_winsorized.dat`.")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def plot_comparison(metrics: list[VersionMetrics], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    axes = axes.ravel()

    for item in metrics:
        axes[0].plot(
            item.rolling_time_min,
            item.rolling_vector_rms,
            color=COLORS.get(item.name, None),
            linewidth=0.9,
            label=item.name,
        )
    axes[0].set_title("1. 5s rolling dynamic vector RMS")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("RMS amplitude (raw counts)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    x = np.arange(4)
    width = 0.24
    for idx, item in enumerate(metrics):
        axes[1].bar(x + (idx - 1) * width, item.quartile_vector_rms, width=width, label=item.name, color=COLORS.get(item.name, None))
    axes[1].set_title("2. Dynamic vector RMS by quartile")
    axes[1].set_xlabel("Quartile")
    axes[1].set_ylabel("RMS amplitude (raw counts)")
    axes[1].set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    for item in metrics:
        mask = (item.spectrum_freq_hz >= 1.0) & (item.spectrum_freq_hz <= 200.0)
        axes[2].semilogy(
            item.spectrum_freq_hz[mask],
            item.spectrum_amp[mask],
            color=COLORS.get(item.name, None),
            linewidth=0.85,
            label=item.name,
        )
    axes[2].set_title("3. Representative vector amplitude spectrum, 1-200 Hz")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Amplitude (raw counts)")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend(loc="upper right", fontsize=8)

    thresholds = np.array([10000, 15000, 20000, 25000, 30000])
    for item in metrics:
        counts = np.array(
            [
                item.rows_ge_10000,
                item.rows_ge_15000,
                item.rows_ge_20000,
                item.rows_ge_25000,
                item.rows_ge_30000,
            ]
        )
        axes[3].plot(thresholds, counts, marker="o", linewidth=1.0, color=COLORS.get(item.name, None), label=item.name)
    axes[3].set_title("4. Rows with any axis above threshold")
    axes[3].set_xlabel("Absolute raw-count threshold")
    axes[3].set_ylabel("Rows")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right", fontsize=8)

    band_labels = ["0-20", "20-100", "100-500", "500-2000", "2000-Nyq"]
    for item in metrics:
        band_mean = item.quartile_band_rms.mean(axis=0)
        axes[4].plot(band_labels, band_mean, marker="o", linewidth=1.0, color=COLORS.get(item.name, None), label=item.name)
    axes[4].set_title("5. Mean quartile band RMS")
    axes[4].set_xlabel("Frequency band (Hz)")
    axes[4].set_ylabel("Band RMS (raw counts)")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(loc="upper right", fontsize=8)

    table_rows = []
    for item in metrics:
        table_rows.append(
            [
                item.name,
                f"{item.vector_dynamic_rms:.1f}",
                f"{item.p999_abs:.0f}",
                f"{item.max_abs}",
                f"{item.rows_ge_10000}",
                f"{item.rows_ge_30000}",
            ]
        )
    axes[5].axis("off")
    table = axes[5].table(
        cellText=table_rows,
        colLabels=["version", "vector RMS", "p99.9 abs", "max abs", "rows >=10k", "rows >=30k"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    axes[5].set_title("6. Numeric comparison")

    fig.suptitle(
        "Raw vs Winsorized vs Cleaned IIS3DWB Data\n"
        "Winsorized preserves time alignment and suppresses high-amplitude outliers; cleaned_removed deletes only near-clip rows.",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw, winsorized, and cleaned IIS3DWB data versions.")
    parser.add_argument("--raw", type=Path, default=Path("iis3dwb_acc.dat"))
    parser.add_argument("--winsorized", type=Path, default=Path("derived_data/iis3dwb_acc_winsorized.dat"))
    parser.add_argument("--cleaned", type=Path, default=Path("derived_data/iis3dwb_acc_cleaned_removed.dat"))
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--offset-bytes", type=int, default=DEFAULT_OFFSET_BYTES)
    parser.add_argument("--output-image", type=Path, default=Path("outputs/data_version_comparison.png"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/data_version_comparison_summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/data_version_comparison_report.md"))
    args = parser.parse_args()

    specs = [
        VersionSpec("raw", args.raw),
        VersionSpec("winsorized", args.winsorized),
        VersionSpec("cleaned_removed", args.cleaned),
    ]
    metrics = [compute_metrics(spec, args.sample_rate_hz, args.offset_bytes) for spec in specs]
    plot_comparison(metrics, args.output_image)
    write_summary_csv(metrics, args.output_csv)
    write_markdown(metrics, args.output_md, args.output_image, args.output_csv)
    print(f"Wrote {args.output_image}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
