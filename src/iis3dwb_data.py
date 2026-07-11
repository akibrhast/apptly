from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


DEFAULT_SAMPLE_RATE_HZ = 26667.0
DEFAULT_OFFSET_BYTES = 0
DEFAULT_SD_ITEM_BYTES = 48128
DEFAULT_SD_ITEM_HEADER_BYTES = 4
DEFAULT_SAMPLES_PER_TIMESTAMP = 1000
DEFAULT_TIMESTAMP_BYTES = 8
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
    framing: str = "raw"
    sd_item_bytes: int | None = None
    sd_item_header_bytes: int | None = None
    samples_per_timestamp: int | None = None
    timestamp_bytes: int | None = None
    sd_headers_removed: int = 0
    timestamp_records_removed: int = 0

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
    framing: Literal["auto", "raw", "datalog2"] = "auto",
    sd_item_bytes: int = DEFAULT_SD_ITEM_BYTES,
    sd_item_header_bytes: int = DEFAULT_SD_ITEM_HEADER_BYTES,
    samples_per_timestamp: int = DEFAULT_SAMPLES_PER_TIMESTAMP,
    timestamp_bytes: int = DEFAULT_TIMESTAMP_BYTES,
) -> Iis3dwbDataset:
    path = Path(path)
    if framing == "auto":
        framing = "datalog2" if looks_like_datalog2_stream(path, sd_item_bytes, sd_item_header_bytes) else "raw"

    if framing == "datalog2":
        return load_datalog2_dataset(
            path=path,
            sample_rate_hz=sample_rate_hz,
            sd_item_bytes=sd_item_bytes,
            sd_item_header_bytes=sd_item_header_bytes,
            samples_per_timestamp=samples_per_timestamp,
            timestamp_bytes=timestamp_bytes,
        )

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
        framing="raw",
    )


def looks_like_datalog2_stream(
    path: str | Path,
    sd_item_bytes: int = DEFAULT_SD_ITEM_BYTES,
    sd_item_header_bytes: int = DEFAULT_SD_ITEM_HEADER_BYTES,
) -> bool:
    path = Path(path)
    size = path.stat().st_size
    if size < sd_item_bytes or size % sd_item_bytes != 0:
        return False

    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n_items = size // sd_item_bytes
    sample_count = min(n_items, 16)
    for item_idx in range(sample_count):
        start = item_idx * sd_item_bytes
        header = raw[start : start + sd_item_header_bytes]
        value = int(np.frombuffer(header, dtype="<u4", count=1)[0])
        expected = item_idx * (sd_item_bytes - sd_item_header_bytes)
        if value != expected:
            return False
    return True


def load_datalog2_dataset(
    path: str | Path,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    sd_item_bytes: int = DEFAULT_SD_ITEM_BYTES,
    sd_item_header_bytes: int = DEFAULT_SD_ITEM_HEADER_BYTES,
    samples_per_timestamp: int = DEFAULT_SAMPLES_PER_TIMESTAMP,
    timestamp_bytes: int = DEFAULT_TIMESTAMP_BYTES,
) -> Iis3dwbDataset:
    path = Path(path)
    xyz, trailing_bytes, n_items, n_timestamps = decode_datalog2_xyz(
        path=path,
        sd_item_bytes=sd_item_bytes,
        sd_item_header_bytes=sd_item_header_bytes,
        samples_per_timestamp=samples_per_timestamp,
        timestamp_bytes=timestamp_bytes,
    )
    return Iis3dwbDataset(
        path=path,
        xyz=xyz,
        sample_rate_hz=sample_rate_hz,
        offset_bytes=0,
        trailing_bytes_ignored=trailing_bytes,
        framing="datalog2",
        sd_item_bytes=sd_item_bytes,
        sd_item_header_bytes=sd_item_header_bytes,
        samples_per_timestamp=samples_per_timestamp,
        timestamp_bytes=timestamp_bytes,
        sd_headers_removed=n_items,
        timestamp_records_removed=n_timestamps,
    )


def decode_datalog2_xyz(
    path: str | Path,
    sd_item_bytes: int = DEFAULT_SD_ITEM_BYTES,
    sd_item_header_bytes: int = DEFAULT_SD_ITEM_HEADER_BYTES,
    samples_per_timestamp: int = DEFAULT_SAMPLES_PER_TIMESTAMP,
    timestamp_bytes: int = DEFAULT_TIMESTAMP_BYTES,
) -> tuple[np.ndarray, int, int, int]:
    path = Path(path)
    if sd_item_header_bytes != 4:
        raise ValueError("Only 4-byte DATALOG2 SD item headers are supported")

    raw = np.memmap(path, dtype=np.uint8, mode="r")
    size = int(raw.size)
    n_items = size // sd_item_bytes
    remainder = size % sd_item_bytes
    payload_per_item = sd_item_bytes - sd_item_header_bytes
    if n_items <= 0:
        raise ValueError("File is too small to contain a DATALOG2 SD item")

    payload_bytes = n_items * payload_per_item + max(0, remainder - sd_item_header_bytes)
    payload = np.empty(payload_bytes, dtype=np.uint8)
    dst = 0
    for item_idx in range(n_items):
        src_start = item_idx * sd_item_bytes + sd_item_header_bytes
        src_stop = (item_idx + 1) * sd_item_bytes
        payload[dst : dst + payload_per_item] = raw[src_start:src_stop]
        dst += payload_per_item
    if remainder > sd_item_header_bytes:
        src_start = n_items * sd_item_bytes + sd_item_header_bytes
        src_stop = n_items * sd_item_bytes + remainder
        n = src_stop - src_start
        payload[dst : dst + n] = raw[src_start:src_stop]

    sample_payload_bytes = samples_per_timestamp * 3 * np.dtype("<i2").itemsize
    frame_bytes = sample_payload_bytes + timestamp_bytes
    n_frames = payload_bytes // frame_bytes
    frame_remainder = payload_bytes % frame_bytes
    data_bytes = n_frames * sample_payload_bytes + min(frame_remainder, sample_payload_bytes)
    usable_data_bytes = data_bytes - (data_bytes % 6)

    decoded = np.empty(usable_data_bytes, dtype=np.uint8)
    dst = 0
    for frame_idx in range(n_frames):
        src_start = frame_idx * frame_bytes
        src_stop = src_start + sample_payload_bytes
        decoded[dst : dst + sample_payload_bytes] = payload[src_start:src_stop]
        dst += sample_payload_bytes
    if frame_remainder:
        src_start = n_frames * frame_bytes
        n = min(frame_remainder, sample_payload_bytes, usable_data_bytes - dst)
        if n > 0:
            decoded[dst : dst + n] = payload[src_start : src_start + n]

    trailing_bytes = payload_bytes - n_frames * timestamp_bytes - usable_data_bytes
    xyz = decoded.view("<i2").reshape(-1, 3)
    return xyz, int(trailing_bytes), int(n_items), int(n_frames)


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
