from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button
import numpy as np

from iis3dwb_data import DEFAULT_OFFSET_BYTES, DEFAULT_SAMPLE_RATE_HZ, SENSITIVITY_MG_PER_LSB, load_dataset


AXES = ("X", "Y", "Z")
AXIS_COLORS = {
    "X": "tab:blue",
    "Y": "tab:orange",
    "Z": "tab:green",
}


@dataclass(frozen=True)
class DisplayValues:
    xyz: np.ndarray
    unit: str
    scale: float
    sample_rate_hz: float


@dataclass(frozen=True)
class GraphSpec:
    title: str
    draw: Callable[[Axes], None]


@dataclass(frozen=True)
class WindowStats:
    time_min: np.ndarray
    rms: np.ndarray
    vector_rms: np.ndarray
    peak_abs: np.ndarray
    crest: np.ndarray
    kurtosis: np.ndarray


@dataclass(frozen=True)
class FftWindowStats:
    time_min: np.ndarray
    band_rms: np.ndarray
    dominant_frequency_hz: np.ndarray
    spectral_centroid_hz: np.ndarray
    spectral_bandwidth_hz: np.ndarray


def block_rms(values: np.ndarray, block_size: int) -> np.ndarray:
    n_blocks = values.shape[0] // block_size
    trimmed = values[: n_blocks * block_size].astype(np.float32)
    blocks = trimmed.reshape(n_blocks, block_size, 3)
    return np.sqrt(np.mean(blocks * blocks, axis=1))


def next_power_of_two_at_most(value: int, limit: int) -> int:
    return 1 << int(np.floor(np.log2(max(2, min(value, limit)))))


def block_dynamic_vector_rms(values: DisplayValues, target_points: int) -> tuple[np.ndarray, np.ndarray]:
    block_size = max(1, values.xyz.shape[0] // target_points)
    n_blocks = values.xyz.shape[0] // block_size
    means = values.xyz.mean(axis=0).astype(np.float64)
    output = np.empty(n_blocks, dtype=np.float64)
    for block_idx in range(n_blocks):
        start = block_idx * block_size
        end = start + block_size
        block = values.xyz[start:end].astype(np.float64) - means
        output[block_idx] = np.sqrt(np.mean(np.sum(block * block, axis=1))) * values.scale
    time_min = np.arange(n_blocks) * block_size / values.sample_rate_hz / 60.0
    return time_min, output


class AnalysisCache:
    def __init__(self, values: DisplayValues, window_seconds: float) -> None:
        self.values = values
        self.window_seconds = window_seconds
        self.window_samples = max(1024, int(round(window_seconds * values.sample_rate_hz)))
        self.bands = [
            (0.0, 20.0),
            (20.0, 100.0),
            (100.0, 500.0),
            (500.0, 2000.0),
            (2000.0, values.sample_rate_hz / 2.0),
        ]
        self._window_stats: WindowStats | None = None
        self._fft_stats: dict[int, FftWindowStats] = {}
        self._spectrum: tuple[np.ndarray, np.ndarray] | None = None
        self._quartile_psd: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._coherence: tuple[np.ndarray, dict[str, np.ndarray]] | None = None

    def window_stats(self) -> WindowStats:
        if self._window_stats is not None:
            return self._window_stats

        n_windows = self.values.xyz.shape[0] // self.window_samples
        rms = np.empty((n_windows, 3), dtype=np.float64)
        vector_rms = np.empty(n_windows, dtype=np.float64)
        peak_abs = np.empty((n_windows, 3), dtype=np.float64)
        crest = np.empty((n_windows, 3), dtype=np.float64)
        kurtosis = np.empty((n_windows, 3), dtype=np.float64)

        for window_idx in range(n_windows):
            start = window_idx * self.window_samples
            end = start + self.window_samples
            block = self.values.xyz[start:end].astype(np.float64)
            centered = block - block.mean(axis=0, keepdims=True)
            block_rms_values = np.sqrt(np.mean(centered * centered, axis=0))
            peaks = np.max(np.abs(centered), axis=0)
            fourth = np.mean(centered**4, axis=0)
            denom = np.maximum(block_rms_values**4, np.finfo(np.float64).eps)

            rms[window_idx] = block_rms_values * self.values.scale
            vector_rms[window_idx] = np.sqrt(np.mean(np.sum(centered * centered, axis=1))) * self.values.scale
            peak_abs[window_idx] = peaks * self.values.scale
            crest[window_idx] = peaks / np.maximum(block_rms_values, np.finfo(np.float64).eps)
            kurtosis[window_idx] = fourth / denom

        time_min = (np.arange(n_windows) + 0.5) * self.window_samples / self.values.sample_rate_hz / 60.0
        self._window_stats = WindowStats(time_min, rms, vector_rms, peak_abs, crest, kurtosis)
        return self._window_stats

    def fft_window_stats(self, axis_index: int = 0) -> FftWindowStats:
        if axis_index in self._fft_stats:
            return self._fft_stats[axis_index]

        n_windows = self.values.xyz.shape[0] // self.window_samples
        nfft = next_power_of_two_at_most(self.window_samples, 65536)
        window = np.hanning(nfft).astype(np.float64)
        freqs = np.fft.rfftfreq(nfft, 1.0 / self.values.sample_rate_hz)
        band_rms = np.empty((n_windows, len(self.bands)), dtype=np.float64)
        dominant_frequency_hz = np.empty(n_windows, dtype=np.float64)
        spectral_centroid_hz = np.empty(n_windows, dtype=np.float64)
        spectral_bandwidth_hz = np.empty(n_windows, dtype=np.float64)

        for window_idx in range(n_windows):
            start = window_idx * self.window_samples
            segment = self.values.xyz[start : start + nfft, axis_index].astype(np.float64)
            segment = (segment - segment.mean()) * self.values.scale
            spectrum = np.fft.rfft(segment * window)
            power = np.abs(spectrum) ** 2
            power_sum = max(float(power[1:].sum()), np.finfo(np.float64).eps)
            spectral_centroid_hz[window_idx] = float(np.sum(freqs[1:] * power[1:]) / power_sum)
            spectral_bandwidth_hz[window_idx] = float(
                np.sqrt(np.sum(((freqs[1:] - spectral_centroid_hz[window_idx]) ** 2) * power[1:]) / power_sum)
            )

            psd = power / (self.values.sample_rate_hz * np.sum(window * window))
            if psd.size > 2:
                psd[1:-1] *= 2.0
            for band_idx, (low, high) in enumerate(self.bands):
                mask = (freqs >= low) & (freqs < high)
                band_rms[window_idx, band_idx] = np.sqrt(np.trapezoid(psd[mask], freqs[mask]))

            mask = freqs >= 1.0
            dominant_frequency_hz[window_idx] = float(freqs[mask][np.argmax(power[mask])])

        time_min = (np.arange(n_windows) + 0.5) * self.window_samples / self.values.sample_rate_hz / 60.0
        self._fft_stats[axis_index] = FftWindowStats(
            time_min,
            band_rms,
            dominant_frequency_hz,
            spectral_centroid_hz,
            spectral_bandwidth_hz,
        )
        return self._fft_stats[axis_index]

    def representative_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        if self._spectrum is not None:
            return self._spectrum

        nfft = min(4_194_304, next_power_of_two_at_most(self.values.xyz.shape[0], 4_194_304))
        start = max(0, (self.values.xyz.shape[0] - nfft) // 2)
        segment = self.values.xyz[start : start + nfft].astype(np.float32) * self.values.scale
        segment -= segment.mean(axis=0, keepdims=True)
        window = np.hanning(nfft).astype(np.float32)
        freqs = np.fft.rfftfreq(nfft, 1.0 / self.values.sample_rate_hz)
        amplitudes = np.empty((freqs.size, 3), dtype=np.float64)
        for axis_idx in range(3):
            amplitudes[:, axis_idx] = (2.0 / np.sum(window)) * np.abs(np.fft.rfft(segment[:, axis_idx] * window))
        self._spectrum = (freqs, amplitudes)
        return self._spectrum

    def quartile_psd(self, axis_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if axis_index in self._quartile_psd:
            return self._quartile_psd[axis_index]

        quartile_len = self.values.xyz.shape[0] // 4
        nfft = next_power_of_two_at_most(quartile_len, 65536)
        window = np.hanning(nfft).astype(np.float64)
        freqs = np.fft.rfftfreq(nfft, 1.0 / self.values.sample_rate_hz)
        quartile_edges = np.linspace(0, self.values.xyz.shape[0], 5, dtype=np.int64)
        psds = np.empty((4, freqs.size), dtype=np.float64)
        scale = self.values.sample_rate_hz * np.sum(window * window)
        for quartile_idx in range(4):
            start = quartile_edges[quartile_idx]
            end = quartile_edges[quartile_idx + 1]
            possible = max(1, (end - start - nfft) // nfft + 1)
            n_segments = min(128, possible)
            offsets = np.linspace(0, max(0, possible - 1), n_segments, dtype=np.int64) * nfft
            psd = np.zeros(freqs.size, dtype=np.float64)
            mean = float(self.values.xyz[start:end, axis_index].mean())
            for offset in offsets:
                segment = (self.values.xyz[start + offset : start + offset + nfft, axis_index].astype(np.float64) - mean)
                segment *= self.values.scale
                spectrum = np.fft.rfft(segment * window)
                psd += (np.abs(spectrum) ** 2) / scale
            psd /= n_segments
            if psd.size > 2:
                psd[1:-1] *= 2.0
            psds[quartile_idx] = psd
        self._quartile_psd[axis_index] = (freqs, psds)
        return self._quartile_psd[axis_index]

    def coherence(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if self._coherence is not None:
            return self._coherence

        nfft = next_power_of_two_at_most(self.values.xyz.shape[0], 65536)
        step = nfft
        window = np.hanning(nfft).astype(np.float64)
        freqs = np.fft.rfftfreq(nfft, 1.0 / self.values.sample_rate_hz)
        possible = max(1, (self.values.xyz.shape[0] - nfft) // step + 1)
        n_segments = min(256, possible)
        starts = np.linspace(0, max(0, possible - 1), n_segments, dtype=np.int64) * step
        pairs = {(0, 1): "X-Y", (0, 2): "X-Z", (1, 2): "Y-Z"}
        pxx = np.zeros((3, freqs.size), dtype=np.float64)
        pxy = {name: np.zeros(freqs.size, dtype=np.complex128) for name in pairs.values()}

        means = self.values.xyz.mean(axis=0).astype(np.float64)
        for start in starts:
            block = (self.values.xyz[start : start + nfft].astype(np.float64) - means) * self.values.scale
            spectra = np.fft.rfft(block * window[:, None], axis=0)
            for axis_idx in range(3):
                pxx[axis_idx] += np.abs(spectra[:, axis_idx]) ** 2
            for (left, right), name in pairs.items():
                pxy[name] += spectra[:, left] * np.conj(spectra[:, right])

        coherence = {}
        for (left, right), name in pairs.items():
            denom = np.maximum(pxx[left] * pxx[right], np.finfo(np.float64).eps)
            coherence[name] = np.clip((np.abs(pxy[name]) ** 2) / denom, 0.0, 1.0)
        self._coherence = (freqs, coherence)
        return self._coherence

    def export_rolling_metrics(self, output: Path) -> None:
        stats = self.window_stats()
        fft_stats = self.fft_window_stats(0)
        output.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "time_start_s",
            "time_end_s",
            "rms_x",
            "rms_y",
            "rms_z",
            "vector_rms",
            "peak_abs_x",
            "peak_abs_y",
            "peak_abs_z",
            "crest_x",
            "crest_y",
            "crest_z",
            "kurtosis_x",
            "kurtosis_y",
            "kurtosis_z",
            "dominant_frequency_hz",
            "dominant_rpm",
            "band_rms_0_20",
            "band_rms_20_100",
            "band_rms_100_500",
            "band_rms_500_2000",
            "band_rms_2000_nyquist",
        ]
        with output.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for idx in range(stats.time_min.size):
                center_s = stats.time_min[idx] * 60.0
                half = self.window_seconds / 2.0
                writer.writerow(
                    [
                        f"{center_s - half:.6f}",
                        f"{center_s + half:.6f}",
                        *[f"{value:.9g}" for value in stats.rms[idx]],
                        f"{stats.vector_rms[idx]:.9g}",
                        *[f"{value:.9g}" for value in stats.peak_abs[idx]],
                        *[f"{value:.9g}" for value in stats.crest[idx]],
                        *[f"{value:.9g}" for value in stats.kurtosis[idx]],
                        f"{fft_stats.dominant_frequency_hz[idx]:.9g}",
                        f"{fft_stats.dominant_frequency_hz[idx] * 60.0:.9g}",
                        *[f"{value:.9g}" for value in fft_stats.band_rms[idx]],
                    ]
                )


def add_description(ax: Axes, text: str) -> None:
    ax.text(
        0.01,
        -0.22,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        wrap=True,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.80",
            "alpha": 0.92,
        },
    )


class InteractiveAnalysisBrowser:
    def __init__(self, graphs: list[GraphSpec]) -> None:
        if not graphs:
            raise ValueError("At least one graph is required")
        self.graphs = graphs
        self.index = 0
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_axes((0.08, 0.22, 0.88, 0.68))
        self.prev_ax = self.fig.add_axes((0.72, 0.055, 0.10, 0.045))
        self.next_ax = self.fig.add_axes((0.84, 0.055, 0.10, 0.045))
        self.prev_button = Button(self.prev_ax, "Previous")
        self.next_button = Button(self.next_ax, "Next")
        self.prev_button.on_clicked(self.previous_graph)
        self.next_button.on_clicked(self.next_graph)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def draw_current(self) -> None:
        for extra_ax in list(self.fig.axes):
            if extra_ax not in {self.ax, self.prev_ax, self.next_ax}:
                extra_ax.remove()
        self.ax.clear()
        graph = self.graphs[self.index]
        graph.draw(self.ax)
        self.fig.suptitle(f"{self.index + 1}/{len(self.graphs)} - {graph.title}", fontsize=14)
        self.fig.canvas.draw_idle()

    def next_graph(self, _event=None) -> None:
        self.index = (self.index + 1) % len(self.graphs)
        self.draw_current()

    def previous_graph(self, _event=None) -> None:
        self.index = (self.index - 1) % len(self.graphs)
        self.draw_current()

    def on_key_press(self, event) -> None:
        if event.key in {"right", "n"}:
            self.next_graph()
        elif event.key in {"left", "p"}:
            self.previous_graph()
        elif event.key == "q":
            plt.close(self.fig)

    def show(self) -> None:
        self.draw_current()
        plt.show()


def make_amplitude_vs_time_graph(values: DisplayValues, target_points: int) -> GraphSpec:
    def draw(ax: Axes) -> None:
        block_size = max(1, values.xyz.shape[0] // target_points)
        reduced = block_rms(values.xyz, block_size) * values.scale
        time_min = np.arange(reduced.shape[0]) * block_size / values.sample_rate_hz / 60.0

        for axis_index, axis_name in enumerate(AXES):
            ax.plot(
                time_min,
                reduced[:, axis_index],
                linewidth=0.75,
                color=AXIS_COLORS[axis_name],
                label=axis_name,
            )

        ax.set_title("Amplitude vs Time")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(f"Block RMS amplitude ({values.unit})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", title="Axis")
        add_description(
            ax,
            "This graph compresses the full recording into block RMS values so the long-duration "
            "vibration trend is visible without plotting all raw samples. Rising or falling lines "
            "indicate broad changes in vibration level over time.",
        )

    return GraphSpec("Amplitude vs Time", draw)


def make_frequency_vs_time_spectrogram_graph(values: DisplayValues, axis_name: str = "X") -> GraphSpec:
    axis_index = AXES.index(axis_name)

    def draw(ax: Axes) -> None:
        decimation = max(1, int(round(values.sample_rate_hz / 4000.0)))
        decimated_rate = values.sample_rate_hz / decimation
        axis_values = values.xyz[::decimation, axis_index].astype(np.float32) * values.scale
        axis_values -= float(axis_values.mean())

        nfft = 4096
        _, _, _, image = ax.specgram(
            axis_values,
            NFFT=nfft,
            Fs=decimated_rate,
            noverlap=nfft // 2,
            window=np.hanning(nfft),
            scale="dB",
            cmap="magma",
        )
        ax.xaxis.set_major_formatter(FuncFormatter(lambda seconds, _pos: f"{seconds / 60.0:g}"))
        ax.set_title(f"Frequency vs Time Spectrogram ({axis_name} axis)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, decimated_rate / 2)
        ax.grid(False)
        colorbar = ax.figure.colorbar(image, ax=ax, pad=0.015)
        colorbar.set_label(f"Power/frequency ({values.unit}^2/Hz, dB)")
        add_description(
            ax,
            "This spectrogram shows how vibration energy is distributed across frequency as time "
            "passes. Bright horizontal bands indicate persistent vibration frequencies; movement "
            "or fading of bands indicates changing speed or changing vibration energy.",
        )

    return GraphSpec(f"Frequency vs Time Spectrogram ({axis_name})", draw)


def make_frequency_vs_amplitude_graph(cache: AnalysisCache) -> GraphSpec:
    def draw(ax: Axes) -> None:
        freqs, amplitudes = cache.representative_spectrum()
        for axis_idx, axis_name in enumerate(AXES):
            ax.plot(freqs, amplitudes[:, axis_idx], linewidth=0.75, color=AXIS_COLORS[axis_name], label=axis_name)
        ax.set_title("Frequency vs Amplitude")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"Amplitude ({cache.values.unit})")
        ax.set_xlim(0, cache.values.sample_rate_hz / 2.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", title="Axis")
        add_description(
            ax,
            "This spectrum uses a representative central segment of the recording. Tall peaks mark "
            "dominant vibration frequencies; matching peaks on all axes suggest a shared mechanical source.",
        )

    return GraphSpec("Frequency vs Amplitude", draw)


def make_rolling_rms_graph(cache: AnalysisCache) -> GraphSpec:
    def draw(ax: Axes) -> None:
        stats = cache.window_stats()
        for axis_idx, axis_name in enumerate(AXES):
            ax.plot(stats.time_min, stats.rms[:, axis_idx], color=AXIS_COLORS[axis_name], label=axis_name)
        ax.plot(stats.time_min, stats.vector_rms, color="black", linewidth=1.25, label="Vector")
        ax.set_title("Rolling RMS Over Time")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(f"Dynamic RMS amplitude ({cache.values.unit})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_description(
            ax,
            f"RMS is calculated in {cache.window_seconds:g}-second windows after removing each window's "
            "mean. It shows slow changes in vibration energy more clearly than raw samples.",
        )

    return GraphSpec("Rolling RMS Over Time", draw)


def make_band_limited_rms_graph(cache: AnalysisCache, axis_name: str = "X") -> GraphSpec:
    axis_index = AXES.index(axis_name)

    def draw(ax: Axes) -> None:
        stats = cache.fft_window_stats(axis_index)
        for band_idx, (low, high) in enumerate(cache.bands):
            label = f"{low:g}-{high:g} Hz" if high < cache.values.sample_rate_hz / 2.0 else f"{low:g}-Nyquist Hz"
            ax.plot(stats.time_min, stats.band_rms[:, band_idx], label=label)
        ax.set_title(f"Band-Limited RMS Over Time ({axis_name} axis)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(f"Band RMS amplitude ({cache.values.unit})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_description(
            ax,
            "Each line tracks vibration energy in a frequency band. Diverging bands show whether "
            "low-frequency motion, harmonics, structural vibration, or high-frequency content is changing.",
        )

    return GraphSpec(f"Band-Limited RMS ({axis_name})", draw)


def make_peak_frequency_graph(cache: AnalysisCache, axis_name: str = "X") -> GraphSpec:
    axis_index = AXES.index(axis_name)

    def draw(ax: Axes) -> None:
        stats = cache.fft_window_stats(axis_index)
        line = ax.plot(stats.time_min, stats.dominant_frequency_hz, color="tab:purple", label="Dominant frequency")[0]
        ax.set_title(f"Peak Frequency / RPM Over Time ({axis_name} axis)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Dominant frequency (Hz)")
        ax.grid(True, alpha=0.25)
        rpm_ax = ax.twinx()
        rpm_ax.set_ylabel("Estimated RPM")
        rpm_ax.set_ylim(ax.get_ylim()[0] * 60.0, ax.get_ylim()[1] * 60.0)
        ax.legend([line], ["Dominant frequency"], loc="upper right")
        add_description(
            ax,
            "For each time window, this graph finds the strongest spectral peak above 1 Hz. If the "
            "peak is rotational, the right axis estimates RPM as frequency times 60.",
        )

    return GraphSpec(f"Peak Frequency / RPM ({axis_name})", draw)


def make_harmonic_analysis_graph(cache: AnalysisCache) -> GraphSpec:
    fundamental_hz = 7.74
    orders = np.array([1, 2, 3, 4, 5, 6, 8, 10], dtype=np.float64)

    def draw(ax: Axes) -> None:
        freqs, amplitudes = cache.representative_spectrum()
        targets = fundamental_hz * orders
        for axis_idx, axis_name in enumerate(AXES):
            harmonic_amplitudes = []
            for target in targets:
                idx = int(np.argmin(np.abs(freqs - target)))
                harmonic_amplitudes.append(amplitudes[idx, axis_idx])
            ax.plot(orders, harmonic_amplitudes, marker="o", color=AXIS_COLORS[axis_name], label=axis_name)
        ax.set_title("Harmonic Analysis")
        ax.set_xlabel("Order / harmonic of 7.74 Hz")
        ax.set_ylabel(f"Amplitude ({cache.values.unit})")
        ax.set_xticks(orders)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", title="Axis")
        add_description(
            ax,
            "This compares vibration amplitude at the presumed 1x rotational component near 7.74 Hz "
            "and selected harmonics. Strong harmonics are common in rotating or reciprocating machinery.",
        )

    return GraphSpec("Harmonic Analysis", draw)


def make_shock_detection_graph(cache: AnalysisCache) -> GraphSpec:
    threshold_counts = 30000.0

    def draw(ax: Axes) -> None:
        stats = cache.window_stats()
        for axis_idx, axis_name in enumerate(AXES):
            peaks_counts = stats.peak_abs[:, axis_idx] / cache.values.scale
            mask = peaks_counts >= threshold_counts
            ax.scatter(
                stats.time_min[mask],
                stats.peak_abs[mask, axis_idx],
                s=18,
                color=AXIS_COLORS[axis_name],
                label=axis_name,
                alpha=0.8,
            )
        threshold = threshold_counts * cache.values.scale
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, label="30000-count threshold")
        ax.set_title("Shock / Impulse Detection")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(f"Peak absolute dynamic amplitude ({cache.values.unit})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_description(
            ax,
            "Markers show windows where at least one axis approaches the high-amplitude threshold. "
            "Clusters suggest repeated impacts; isolated points may be transient shocks or artifacts.",
        )

    return GraphSpec("Shock / Impulse Detection", draw)


def make_clipping_map_graph(cache: AnalysisCache) -> GraphSpec:
    bin_seconds = 10.0

    def draw(ax: Axes) -> None:
        bin_samples = max(1, int(round(bin_seconds * cache.values.sample_rate_hz)))
        n_bins = cache.values.xyz.shape[0] // bin_samples
        time_min = (np.arange(n_bins) + 0.5) * bin_samples / cache.values.sample_rate_hz / 60.0
        counts = np.empty((n_bins, 3), dtype=np.int64)
        for idx in range(n_bins):
            start = idx * bin_samples
            end = start + bin_samples
            counts[idx] = (np.abs(cache.values.xyz[start:end].astype(np.int32)) >= 32760).sum(axis=0)
        for axis_idx, axis_name in enumerate(AXES):
            ax.plot(time_min, counts[:, axis_idx], color=AXIS_COLORS[axis_name], label=axis_name)
        ax.set_title("Clipping / Saturation Map")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(f"Near-saturation samples per {bin_seconds:g}-second bin")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", title="Axis")
        add_description(
            ax,
            "This counts samples very close to signed int16 limits. Nonzero bins indicate possible "
            "sensor saturation, hard impacts, or logging artifacts.",
        )

    return GraphSpec("Clipping / Saturation Map", draw)


def make_axis_correlation_graph(cache: AnalysisCache) -> GraphSpec:
    def draw(ax: Axes) -> None:
        stride = max(1, cache.values.xyz.shape[0] // 300000)
        sample = cache.values.xyz[::stride].astype(np.float64)
        corr = np.corrcoef(sample.T)
        image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_title("Axis Correlation Matrix")
        ax.set_xlabel("Axis")
        ax.set_ylabel("Axis")
        ax.set_xticks(range(3), AXES)
        ax.set_yticks(range(3), AXES)
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f"{corr[row, col]:.3f}", ha="center", va="center", color="black")
        colorbar = ax.figure.colorbar(image, ax=ax, pad=0.015)
        colorbar.set_label("Correlation coefficient")
        add_description(
            ax,
            "Correlation summarizes how similarly the three axes move. Values near 1 or -1 indicate "
            "strong shared motion; values near 0 indicate weaker linear coupling.",
        )

    return GraphSpec("Axis Correlation", draw)


def make_axis_coherence_graph(cache: AnalysisCache) -> GraphSpec:
    def draw(ax: Axes) -> None:
        freqs, coherence = cache.coherence()
        for label, values in coherence.items():
            ax.plot(freqs, values, linewidth=0.85, label=label)
        ax.set_title("Axis Coherence")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude-squared coherence")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 500)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_description(
            ax,
            "Coherence shows whether pairs of axes share frequency-specific vibration content. "
            "Values near 1 mean two axes are strongly coupled at that frequency.",
        )

    return GraphSpec("Axis Coherence", draw)


def make_dynamic_vector_magnitude_graph(cache: AnalysisCache, target_points: int) -> GraphSpec:
    def draw(ax: Axes) -> None:
        time_min, vector = block_dynamic_vector_rms(cache.values, target_points)
        ax.plot(time_min, vector, color="black", linewidth=0.8, label="Dynamic vector magnitude")
        ax.set_title("Dynamic Vector Magnitude")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(f"Block RMS vector magnitude ({cache.values.unit})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_description(
            ax,
            "After removing each axis mean, this combines X/Y/Z into one direction-independent "
            "vibration magnitude. It is useful when vibration direction is less important than total energy.",
        )

    return GraphSpec("Dynamic Vector Magnitude", draw)


def make_histogram_graph(cache: AnalysisCache) -> GraphSpec:
    def draw(ax: Axes) -> None:
        stride = max(1, cache.values.xyz.shape[0] // 1000000)
        sample = cache.values.xyz[::stride].astype(np.float64) * cache.values.scale
        for axis_idx, axis_name in enumerate(AXES):
            ax.hist(
                sample[:, axis_idx],
                bins=180,
                density=True,
                histtype="step",
                linewidth=1.2,
                color=AXIS_COLORS[axis_name],
                label=axis_name,
            )
        ax.set_title("Amplitude Histogram / Probability Distribution")
        ax.set_xlabel(f"Amplitude ({cache.values.unit})")
        ax.set_ylabel("Probability density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", title="Axis")
        add_description(
            ax,
            "This distribution shows how often different amplitudes occur. Heavy tails indicate "
            "more impulses or rare large excursions than a simple Gaussian-like vibration signal.",
        )

    return GraphSpec("Amplitude Histogram", draw)


def make_crest_factor_graph(cache: AnalysisCache) -> GraphSpec:
    def draw(ax: Axes) -> None:
        stats = cache.window_stats()
        for axis_idx, axis_name in enumerate(AXES):
            ax.plot(stats.time_min, stats.crest[:, axis_idx], color=AXIS_COLORS[axis_name], label=axis_name)
        ax.set_title("Crest Factor Over Time")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Crest factor (peak / RMS)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", title="Axis")
        add_description(
            ax,
            "Crest factor compares peak amplitude to RMS in each window. Higher values indicate "
            "sharper impacts or more impulsive vibration.",
        )

    return GraphSpec("Crest Factor Over Time", draw)


def make_kurtosis_graph(cache: AnalysisCache) -> GraphSpec:
    def draw(ax: Axes) -> None:
        stats = cache.window_stats()
        for axis_idx, axis_name in enumerate(AXES):
            ax.plot(stats.time_min, stats.kurtosis[:, axis_idx], color=AXIS_COLORS[axis_name], label=axis_name)
        ax.axhline(3.0, color="black", linestyle="--", linewidth=1.0, label="Gaussian kurtosis = 3")
        ax.set_title("Kurtosis Over Time")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Kurtosis")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_description(
            ax,
            "Kurtosis is calculated with the standard non-excess convention, where a Gaussian signal "
            "is near 3. Values above 3 indicate heavier tails and more impulsive behavior.",
        )

    return GraphSpec("Kurtosis Over Time", draw)


def make_spectral_centroid_graph(cache: AnalysisCache, axis_name: str = "X") -> GraphSpec:
    axis_index = AXES.index(axis_name)

    def draw(ax: Axes) -> None:
        stats = cache.fft_window_stats(axis_index)
        ax.plot(stats.time_min, stats.spectral_centroid_hz, label="Spectral centroid", color="tab:blue")
        ax.plot(stats.time_min, stats.spectral_bandwidth_hz, label="Spectral bandwidth", color="tab:red")
        ax.set_title(f"Spectral Centroid / Bandwidth Over Time ({axis_name} axis)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Frequency (Hz)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_description(
            ax,
            "The centroid is the spectrum's center of mass; bandwidth shows spread around it. "
            "Together they show whether energy shifts toward higher or lower frequencies over time.",
        )

    return GraphSpec(f"Spectral Centroid / Bandwidth ({axis_name})", draw)


def make_waterfall_spectrum_graph(cache: AnalysisCache, axis_name: str = "X") -> GraphSpec:
    axis_index = AXES.index(axis_name)

    def draw(ax: Axes) -> None:
        nfft = next_power_of_two_at_most(cache.values.xyz.shape[0], 32768)
        n_lines = 24
        window = np.hanning(nfft).astype(np.float64)
        freqs = np.fft.rfftfreq(nfft, 1.0 / cache.values.sample_rate_hz)
        starts = np.linspace(0, cache.values.xyz.shape[0] - nfft, n_lines, dtype=np.int64)
        max_amp = 0.0
        spectra = []
        for start in starts:
            segment = cache.values.xyz[start : start + nfft, axis_index].astype(np.float64)
            segment = (segment - segment.mean()) * cache.values.scale
            amp = (2.0 / np.sum(window)) * np.abs(np.fft.rfft(segment * window))
            spectra.append(amp)
            max_amp = max(max_amp, float(np.percentile(amp, 99.5)))
        offset = max(max_amp, np.finfo(np.float64).eps)
        for idx, amp in enumerate(spectra):
            label = f"{starts[idx] / cache.values.sample_rate_hz / 60:.1f} min"
            ax.plot(freqs, amp + idx * offset, linewidth=0.65, label=label if idx in {0, n_lines - 1} else None)
        ax.set_title(f"Waterfall Spectrum ({axis_name} axis)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"Amplitude + time offset ({cache.values.unit})")
        ax.set_xlim(0, 500)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_description(
            ax,
            "This stacks spectra from multiple times in the recording. Consistent peak positions "
            "show steady frequencies; changing line shapes reveal evolving spectral content.",
        )

    return GraphSpec(f"Waterfall Spectrum ({axis_name})", draw)


def make_quartile_psd_graph(cache: AnalysisCache, axis_name: str = "X") -> GraphSpec:
    axis_index = AXES.index(axis_name)

    def draw(ax: Axes) -> None:
        freqs, psds = cache.quartile_psd(axis_index)
        for quartile_idx in range(4):
            ax.semilogy(freqs, psds[quartile_idx], linewidth=0.8, label=f"Q{quartile_idx + 1}")
        ax.set_title(f"Quartile PSD Overlay ({axis_name} axis)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"PSD ({cache.values.unit}^2/Hz)")
        ax.set_xlim(0, 500)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_description(
            ax,
            "The recording is split into four equal time quartiles. Overlaid PSD curves make it "
            "easy to compare frequency content early, middle, and late in the capture.",
        )

    return GraphSpec(f"Quartile PSD Overlay ({axis_name})", draw)


def make_frequency_band_heatmap_graph(cache: AnalysisCache, axis_name: str = "X") -> GraphSpec:
    axis_index = AXES.index(axis_name)

    def draw(ax: Axes) -> None:
        stats = cache.fft_window_stats(axis_index)
        labels = [
            f"{low:g}-{high:g} Hz" if high < cache.values.sample_rate_hz / 2.0 else f"{low:g}-Nyquist Hz"
            for low, high in cache.bands
        ]
        image = ax.imshow(
            stats.band_rms.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(stats.time_min[0], stats.time_min[-1], -0.5, len(labels) - 0.5),
        )
        ax.set_title(f"Frequency-Band Heatmap ({axis_name} axis)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Frequency band")
        ax.set_yticks(range(len(labels)), labels)
        colorbar = ax.figure.colorbar(image, ax=ax, pad=0.015)
        colorbar.set_label(f"Band RMS amplitude ({cache.values.unit})")
        add_description(
            ax,
            "This compact heatmap summarizes band energy over time. Brighter cells identify time "
            "periods and frequency ranges with stronger vibration.",
        )

    return GraphSpec(f"Frequency-Band Heatmap ({axis_name})", draw)


def make_largest_event_waveform_graph(cache: AnalysisCache) -> GraphSpec:
    def draw(ax: Axes) -> None:
        stats = cache.window_stats()
        window_idx, axis_idx = np.unravel_index(np.argmax(stats.peak_abs), stats.peak_abs.shape)
        start = max(0, window_idx * cache.window_samples - int(cache.values.sample_rate_hz * 0.25))
        end = min(cache.values.xyz.shape[0], start + int(cache.values.sample_rate_hz * 0.5))
        segment = cache.values.xyz[start:end].astype(np.float64) * cache.values.scale
        time_s = (np.arange(segment.shape[0]) + start) / cache.values.sample_rate_hz
        for idx, axis_name in enumerate(AXES):
            ax.plot(time_s, segment[:, idx], color=AXIS_COLORS[axis_name], linewidth=0.8, label=axis_name)
        ax.set_title("Largest Detected Event Waveform")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(f"Amplitude ({cache.values.unit})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        add_description(
            ax,
            "This zooms into the highest peak found in the rolling windows. Use toolbar zoom/pan "
            "to inspect the local waveform shape around the largest transient.",
        )

    return GraphSpec("Largest Event Waveform", draw)


def build_graphs(values: DisplayValues, cache: AnalysisCache, target_points: int) -> list[GraphSpec]:
    return [
        make_amplitude_vs_time_graph(values, target_points),
        make_frequency_vs_time_spectrogram_graph(values, "X"),
        make_frequency_vs_amplitude_graph(cache),
        make_rolling_rms_graph(cache),
        make_band_limited_rms_graph(cache, "X"),
        make_peak_frequency_graph(cache, "X"),
        make_harmonic_analysis_graph(cache),
        make_shock_detection_graph(cache),
        make_largest_event_waveform_graph(cache),
        make_clipping_map_graph(cache),
        make_axis_correlation_graph(cache),
        make_axis_coherence_graph(cache),
        make_dynamic_vector_magnitude_graph(cache, target_points),
        make_histogram_graph(cache),
        make_crest_factor_graph(cache),
        make_kurtosis_graph(cache),
        make_spectral_centroid_graph(cache, "X"),
        make_waterfall_spectrum_graph(cache, "X"),
        make_quartile_psd_graph(cache, "X"),
        make_frequency_band_heatmap_graph(cache, "X"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive matplotlib browser for IIS3DWB vibration analysis.")
    parser.add_argument("--input", type=Path, default=Path("iis3dwb_acc.dat"))
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--offset-bytes", type=int, default=DEFAULT_OFFSET_BYTES)
    parser.add_argument("--full-scale-g", type=int, choices=(2, 4, 8, 16))
    parser.add_argument(
        "--target-points",
        type=int,
        default=20000,
        help="Approximate number of plotted points for full-duration downsampled graphs.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=5.0,
        help="Window size for rolling and FFT-based metrics.",
    )
    parser.add_argument(
        "--export-rolling-metrics",
        type=Path,
        default=None,
        help="Optional CSV output path for rolling metrics, for example outputs/rolling_metrics.csv.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Compute requested exports and exit without opening the interactive matplotlib window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.input, args.sample_rate_hz, args.offset_bytes)

    unit = "raw counts"
    scale = 1.0
    if args.full_scale_g:
        unit = "g"
        scale = SENSITIVITY_MG_PER_LSB[args.full_scale_g] / 1000.0

    values = DisplayValues(
        xyz=dataset.xyz,
        unit=unit,
        scale=scale,
        sample_rate_hz=dataset.sample_rate_hz,
    )
    cache = AnalysisCache(values, args.window_seconds)
    if args.export_rolling_metrics:
        cache.export_rolling_metrics(args.export_rolling_metrics)
        print(f"Wrote {args.export_rolling_metrics}")
    if args.no_show:
        return
    graphs = build_graphs(values, cache, args.target_points)
    browser = InteractiveAnalysisBrowser(graphs)
    browser.show()


if __name__ == "__main__":
    main()
