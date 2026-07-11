from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from iis3dwb_data import (
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_OFFSET_BYTES,
    counts_to_g,
    dynamic_vector_rms,
    load_dataset,
    top_spectral_peaks,
    welch_psd,
)


AXES = ("X", "Y", "Z")


def format_row(values: list[object]) -> str:
    return ",".join(str(value) for value in values)


def analyze(args: argparse.Namespace) -> str:
    dataset = load_dataset(args.input, args.sample_rate_hz, args.offset_bytes)
    xyz = dataset.xyz
    unit = "counts"
    scale_xyz = xyz

    if args.full_scale_g:
        scale_xyz = counts_to_g(xyz, args.full_scale_g)
        unit = "g"

    lines: list[str] = []
    lines.append("IIS3DWB analysis")
    lines.append(f"input,{dataset.path}")
    lines.append(f"samples,{dataset.samples}")
    lines.append(f"sample_rate_hz,{dataset.sample_rate_hz}")
    lines.append(f"duration_seconds,{dataset.duration_seconds:.6f}")
    lines.append(f"duration_minutes,{dataset.duration_minutes:.6f}")
    lines.append(f"offset_bytes,{dataset.offset_bytes}")
    lines.append(f"trailing_bytes_ignored,{dataset.trailing_bytes_ignored}")
    lines.append(f"unit,{unit}")
    lines.append("")

    lines.append("overall_axis_stats")
    lines.append("axis,mean,std,rms,min,max,p95_abs,p99_abs,p999_abs,near_clip_pct")
    values = scale_xyz.astype(np.float64)
    raw_abs = np.abs(xyz.astype(np.float64))
    for axis_idx, axis_name in enumerate(AXES):
        axis = values[:, axis_idx]
        lines.append(
            format_row(
                [
                    axis_name,
                    f"{axis.mean():.9g}",
                    f"{axis.std():.9g}",
                    f"{np.sqrt(np.mean(axis * axis)):.9g}",
                    f"{axis.min():.9g}",
                    f"{axis.max():.9g}",
                    f"{np.percentile(np.abs(axis), 95):.9g}",
                    f"{np.percentile(np.abs(axis), 99):.9g}",
                    f"{np.percentile(np.abs(axis), 99.9):.9g}",
                    f"{(raw_abs[:, axis_idx] >= 30000).mean() * 100:.9g}",
                ]
            )
        )
    lines.append("")

    quartile_edges = np.linspace(0, dataset.samples, 5, dtype=np.int64)
    lines.append("quartile_dynamic_rms")
    lines.append("quartile,start_min,end_min,axis_rms_x_counts,axis_rms_y_counts,axis_rms_z_counts,vector_rms_counts")
    for idx in range(4):
        start = quartile_edges[idx]
        end = quartile_edges[idx + 1]
        axis_rms, vector_rms = dynamic_vector_rms(xyz[start:end])
        lines.append(
            format_row(
                [
                    idx + 1,
                    f"{start / dataset.sample_rate_hz / 60:.6f}",
                    f"{end / dataset.sample_rate_hz / 60:.6f}",
                    f"{axis_rms[0]:.9g}",
                    f"{axis_rms[1]:.9g}",
                    f"{axis_rms[2]:.9g}",
                    f"{vector_rms:.9g}",
                ]
            )
        )
    lines.append("")

    half_edges = np.linspace(0, dataset.samples, 3, dtype=np.int64)
    lines.append("half_dynamic_rms")
    lines.append("half,start_min,end_min,axis_rms_x_counts,axis_rms_y_counts,axis_rms_z_counts,vector_rms_counts")
    for idx in range(2):
        start = half_edges[idx]
        end = half_edges[idx + 1]
        axis_rms, vector_rms = dynamic_vector_rms(xyz[start:end])
        lines.append(
            format_row(
                [
                    idx + 1,
                    f"{start / dataset.sample_rate_hz / 60:.6f}",
                    f"{end / dataset.sample_rate_hz / 60:.6f}",
                    f"{axis_rms[0]:.9g}",
                    f"{axis_rms[1]:.9g}",
                    f"{axis_rms[2]:.9g}",
                    f"{vector_rms:.9g}",
                ]
            )
        )
    lines.append("")

    lines.append("quartile_spectral_summary_counts")
    lines.append("axis,quartile,band_0_20,band_20_100,band_100_500,band_500_2000,band_2000_nyquist,top_peak_hz")
    bands = [(0.0, 20.0), (20.0, 100.0), (100.0, 500.0), (500.0, 2000.0), (2000.0, dataset.sample_rate_hz / 2)]
    for axis_idx, axis_name in enumerate(AXES):
        for idx in range(4):
            start = quartile_edges[idx]
            end = quartile_edges[idx + 1]
            freqs, psd = welch_psd(xyz[start:end, axis_idx], dataset.sample_rate_hz)
            band_rms = []
            for low, high in bands:
                mask = (freqs >= low) & (freqs < high)
                band_rms.append(float(np.sqrt(np.trapezoid(psd[mask], freqs[mask]))))
            peaks = top_spectral_peaks(freqs, psd, count=8)
            lines.append(
                format_row(
                    [
                        axis_name,
                        idx + 1,
                        *[f"{value:.9g}" for value in band_rms],
                        " ".join(f"{freq:.2f}" for freq, _ in peaks),
                    ]
                )
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze IIS3DWB raw accelerometer data.")
    parser.add_argument("--input", type=Path, default=Path("iis3dwb_acc.dat"))
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--offset-bytes", type=int, default=DEFAULT_OFFSET_BYTES)
    parser.add_argument("--full-scale-g", type=int, choices=(2, 4, 8, 16))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis_summary.csv"))
    args = parser.parse_args()

    text = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
