from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iis3dwb_data import DEFAULT_OFFSET_BYTES, DEFAULT_SAMPLE_RATE_HZ, counts_to_g, load_dataset


AXES = ("X", "Y", "Z")


def block_rms(values: np.ndarray, block_size: int) -> np.ndarray:
    n_blocks = values.shape[0] // block_size
    trimmed = values[: n_blocks * block_size].astype(np.float32)
    blocks = trimmed.reshape(n_blocks, block_size, 3)
    return np.sqrt(np.mean(blocks * blocks, axis=1))


def plot_amplitude_vs_time(values: np.ndarray, sample_rate_hz: float, unit: str, output: Path) -> None:
    block_size = max(1, values.shape[0] // 20000)
    reduced = block_rms(values, block_size)
    time_s = np.arange(reduced.shape[0]) * block_size / sample_rate_hz

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for axis_idx, axis_name in enumerate(AXES):
        ax.plot(time_s, reduced[:, axis_idx], linewidth=0.65, label=axis_name)
    ax.set_title("Amplitude vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"RMS amplitude per {block_size}-sample block ({unit})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_frequency_vs_amplitude(values: np.ndarray, sample_rate_hz: float, unit: str, output: Path) -> None:
    n = min(values.shape[0], 4_194_304)
    n_fft = 1 << int(np.floor(np.log2(n)))
    start = max(0, (values.shape[0] - n_fft) // 2)
    segment = values[start : start + n_fft].astype(np.float32)
    segment -= segment.mean(axis=0, keepdims=True)
    window = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for axis_idx, axis_name in enumerate(AXES):
        amplitude = (2.0 / np.sum(window)) * np.abs(np.fft.rfft(segment[:, axis_idx] * window))
        ax.plot(freqs, amplitude, linewidth=0.75, label=axis_name)
    ax.set_title("Frequency vs Amplitude")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Amplitude ({unit})")
    ax.set_xlim(0, sample_rate_hz / 2)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_frequency_vs_time(values: np.ndarray, sample_rate_hz: float, unit: str, output: Path, axis_index: int) -> None:
    decimation = max(1, int(round(sample_rate_hz / 4000.0)))
    axis_values = values[::decimation, axis_index].astype(np.float32)
    decimated_rate = sample_rate_hz / decimation
    axis_values -= axis_values.mean()

    nfft = 4096
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    _, _, _, image = ax.specgram(
        axis_values,
        NFFT=nfft,
        Fs=decimated_rate,
        noverlap=nfft // 2,
        window=np.hanning(nfft),
        scale="dB",
        cmap="magma",
    )
    ax.set_title(f"Frequency vs Time Spectrogram ({AXES[axis_index]} axis)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, decimated_rate / 2)
    fig.colorbar(image, ax=ax, label=f"Power/frequency ({unit}^2/Hz, dB)")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IIS3DWB raw accelerometer data.")
    parser.add_argument("--input", type=Path, default=Path("iis3dwb_acc.dat"))
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--offset-bytes", type=int, default=DEFAULT_OFFSET_BYTES)
    parser.add_argument("--full-scale-g", type=int, choices=(2, 4, 8, 16))
    parser.add_argument("--spectrogram-axis", choices=AXES, default="X")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    dataset = load_dataset(args.input, args.sample_rate_hz, args.offset_bytes)
    values = dataset.xyz
    unit = "raw counts"
    if args.full_scale_g:
        values = counts_to_g(values, args.full_scale_g)
        unit = "g"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_amplitude_vs_time(values, dataset.sample_rate_hz, unit, args.output_dir / "amplitude_vs_time.png")
    plot_frequency_vs_time(
        values,
        dataset.sample_rate_hz,
        unit,
        args.output_dir / "frequency_vs_time_spectrogram.png",
        AXES.index(args.spectrogram_axis),
    )
    plot_frequency_vs_amplitude(values, dataset.sample_rate_hz, unit, args.output_dir / "frequency_vs_amplitude.png")

    print(f"Samples: {dataset.samples}")
    print(f"Duration: {dataset.duration_minutes:.3f} min")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
