from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SAMPLE_RATE_HZ = 26667.0
DEFAULT_OFFSET_BYTES = 4
SENSITIVITY_MG_PER_LSB = {
    2: 0.061,
    4: 0.122,
    8: 0.244,
    16: 0.488,
}


@dataclass(frozen=True)
class Iis3dwbDataset:
    path: Path
    xyz: np.ndarray
    sample_rate_hz: float
    offset_bytes: int
    trailing_bytes_ignored: int

    @property
    def samples(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.samples / self.sample_rate_hz

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0


def load_dataset(
    path: str | Path = "iis3dwb_acc.dat",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    offset_bytes: int = DEFAULT_OFFSET_BYTES,
) -> Iis3dwbDataset:
    path = Path(path)
    if offset_bytes % 2:
        raise ValueError("offset_bytes must be even because data is int16 aligned")

    raw = np.memmap(path, dtype="<i2", mode="r")
    offset_words = offset_bytes // 2
    usable_words = ((raw.size - offset_words) // 3) * 3
    xyz = raw[offset_words : offset_words + usable_words].reshape(-1, 3)
    trailing_bytes = (raw.size - offset_words - usable_words) * 2
    return Iis3dwbDataset(
        path=path,
        xyz=xyz,
        sample_rate_hz=sample_rate_hz,
        offset_bytes=offset_bytes,
        trailing_bytes_ignored=trailing_bytes,
    )


def counts_to_g(values: np.ndarray, full_scale_g: int) -> np.ndarray:
    try:
        sensitivity = SENSITIVITY_MG_PER_LSB[full_scale_g]
    except KeyError as exc:
        raise ValueError("full_scale_g must be one of 2, 4, 8, or 16") from exc
    return values.astype(np.float64) * sensitivity / 1000.0


def dynamic_vector_rms(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    centered = xyz.astype(np.float64) - xyz.astype(np.float64).mean(axis=0, keepdims=True)
    axis_rms = np.sqrt(np.mean(centered * centered, axis=0))
    vector_rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    return axis_rms, vector_rms


def welch_psd(
    values: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 65536,
    max_segments: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    values = values.astype(np.float64)
    if values.size < nperseg:
        nperseg = 1 << int(np.floor(np.log2(values.size)))
    step = nperseg
    possible = max(1, (values.size - nperseg) // step + 1)
    n_segments = min(max_segments, possible)
    starts = np.linspace(0, max(0, possible - 1), n_segments, dtype=np.int64) * step

    window = np.hanning(nperseg).astype(np.float64)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sample_rate_hz)
    scale = sample_rate_hz * np.sum(window * window)
    psd = np.zeros(freqs.size, dtype=np.float64)
    mean = float(values.mean())

    for start in starts:
        segment = values[start : start + nperseg] - mean
        fft = np.fft.rfft(segment * window)
        psd += (np.abs(fft) ** 2) / scale

    psd /= n_segments
    if psd.size > 2:
        psd[1:-1] *= 2.0
    return freqs, psd


def top_spectral_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    min_frequency_hz: float = 1.0,
    min_spacing_hz: float = 3.0,
    count: int = 8,
) -> list[tuple[float, float]]:
    mask = freqs >= min_frequency_hz
    ff = freqs[mask]
    pp = psd[mask]
    if pp.size < 3:
        return []

    local = np.where((pp[1:-1] > pp[:-2]) & (pp[1:-1] >= pp[2:]))[0] + 1
    ordered = local[np.argsort(pp[local])[::-1]]
    chosen: list[int] = []
    for idx in ordered:
        if all(abs(ff[idx] - ff[prev]) >= min_spacing_hz for prev in chosen):
            chosen.append(int(idx))
        if len(chosen) >= count:
            break
    return [(float(ff[idx]), float(pp[idx])) for idx in chosen]
