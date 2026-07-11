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
AXIS_COLORS = ("tab:blue", "tab:orange", "tab:green")


def collect_near_clip_events(
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

    return (
        np.concatenate(sample_parts),
        np.concatenate(axis_parts),
        np.concatenate(value_parts),
    )


def plot_axis_bars(ax: plt.Axes, x: np.ndarray, counts_by_axis: np.ndarray, title: str, xlabel: str) -> None:
    bottom = np.zeros(x.size, dtype=np.int64)
    for axis_idx, axis_name in enumerate(AXES):
        ax.bar(
            x,
            counts_by_axis[axis_idx],
            bottom=bottom,
            width=1.0,
            color=AXIS_COLORS[axis_idx],
            label=axis_name,
            linewidth=0,
        )
        bottom += counts_by_axis[axis_idx]
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Near-clip samples")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def counts_by_mod(sample_idx: np.ndarray, axis_idx: np.ndarray, modulo: int, byte_offset: int) -> np.ndarray:
    byte_pos = byte_offset + sample_idx * 6 + axis_idx * 2
    bins = (byte_pos % modulo).astype(np.int64)
    counts = np.zeros((3, modulo), dtype=np.int64)
    for axis in range(3):
        counts[axis] = np.bincount(bins[axis_idx == axis], minlength=modulo)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Check IIS3DWB near-clip spikes against FIFO and file block boundaries.")
    parser.add_argument("--input", type=Path, default=Path("iis3dwb_acc.dat"))
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--offset-bytes", type=int, default=4)
    parser.add_argument("--near-clip-threshold", type=int, default=30000)
    parser.add_argument("--fifo-samples", type=int, default=256)
    parser.add_argument("--chunk-samples", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/fifo_boundary_diagnostics.png"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/fifo_boundary_summary.txt"))
    args = parser.parse_args()

    dataset = load_dataset(args.input, args.sample_rate_hz, args.offset_bytes)
    sample_idx, axis_idx, values = collect_near_clip_events(dataset.xyz, args.near_clip_threshold, args.chunk_samples)

    if sample_idx.size == 0:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(f"No samples found at abs(value) >= {args.near_clip_threshold}\n", encoding="utf-8")
        raise SystemExit(f"No samples found at abs(value) >= {args.near_clip_threshold}")

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    axes = axes.ravel()

    fifo_pos = (sample_idx % args.fifo_samples).astype(np.int64)
    fifo_counts = np.zeros((3, args.fifo_samples), dtype=np.int64)
    for axis in range(3):
        fifo_counts[axis] = np.bincount(fifo_pos[axis_idx == axis], minlength=args.fifo_samples)
    plot_axis_bars(
        axes[0],
        np.arange(args.fifo_samples),
        fifo_counts,
        f"1. Near-clips by position inside {args.fifo_samples}-sample FIFO payload",
        "Sample position within FIFO payload",
    )

    counts_1536 = counts_by_mod(sample_idx, axis_idx, args.fifo_samples * 6, dataset.offset_bytes)
    plot_axis_bars(
        axes[1],
        np.arange(args.fifo_samples * 6),
        counts_1536,
        "2. Near-clips by byte offset modulo 1536-byte payload",
        "Byte offset modulo 1536",
    )

    counts_512 = counts_by_mod(sample_idx, axis_idx, 512, dataset.offset_bytes)
    plot_axis_bars(
        axes[2],
        np.arange(512),
        counts_512,
        "3. Near-clips by byte offset modulo 512-byte SD sector",
        "Byte offset modulo 512",
    )

    counts_4096 = counts_by_mod(sample_idx, axis_idx, 4096, dataset.offset_bytes)
    plot_axis_bars(
        axes[3],
        np.arange(4096),
        counts_4096,
        "4. Near-clips by byte offset modulo 4096 bytes",
        "Byte offset modulo 4096",
    )

    fifo_block_idx = sample_idx // args.fifo_samples
    block_counts = np.bincount(fifo_block_idx, minlength=int(dataset.samples // args.fifo_samples) + 1)
    block_time_min = (np.arange(block_counts.size) * args.fifo_samples) / dataset.sample_rate_hz / 60.0
    axes[4].plot(block_time_min, block_counts, color="black", linewidth=0.7, label="All axes")
    axes[4].set_title("5. Near-clip count by FIFO payload over acquisition time")
    axes[4].set_xlabel("Time (minutes)")
    axes[4].set_ylabel("Near-clip samples per FIFO payload")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(loc="upper right", fontsize=8)

    order = np.argsort(sample_idx)
    event_times = sample_idx[order] / dataset.sample_rate_hz
    deltas = np.diff(event_times)
    deltas = deltas[(deltas > 0) & (deltas <= 60.0)]
    axes[5].hist(deltas, bins=120, color="tab:purple", alpha=0.85, label="Near-clip event gaps")
    axes[5].set_title("6. Time gaps between near-clip events")
    axes[5].set_xlabel("Gap between events (seconds)")
    axes[5].set_ylabel("Count")
    axes[5].grid(True, axis="y", alpha=0.25)
    axes[5].legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "IIS3DWB Firmware-Aware Boundary Diagnostics\n"
        "Concentration at FIFO sample positions, 1536-byte payload boundaries, or SD block offsets points toward logger/FIFO artifacts.",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)

    total_by_axis = np.bincount(axis_idx, minlength=3)
    worst_fifo_pos = int(np.argmax(fifo_counts.sum(axis=0)))
    worst_fifo_count = int(fifo_counts.sum(axis=0)[worst_fifo_pos])
    worst_512_pos = int(np.argmax(counts_512.sum(axis=0)))
    worst_512_count = int(counts_512.sum(axis=0)[worst_512_pos])
    worst_1536_pos = int(np.argmax(counts_1536.sum(axis=0)))
    worst_1536_count = int(counts_1536.sum(axis=0)[worst_1536_pos])
    max_block_count = int(block_counts.max())
    nonzero_blocks = int((block_counts > 0).sum())
    lines = [
        f"input={dataset.path}",
        f"samples={dataset.samples}",
        f"duration_minutes={dataset.duration_minutes:.6f}",
        f"sample_rate_hz={dataset.sample_rate_hz:.3f}",
        f"offset_bytes={dataset.offset_bytes}",
        f"near_clip_threshold={args.near_clip_threshold}",
        f"near_clip_events={sample_idx.size}",
        f"near_clip_by_axis X/Y/Z={total_by_axis[0]}/{total_by_axis[1]}/{total_by_axis[2]}",
        f"worst_fifo_position={worst_fifo_pos} count={worst_fifo_count}",
        f"worst_byte_mod_1536={worst_1536_pos} count={worst_1536_count}",
        f"worst_byte_mod_512={worst_512_pos} count={worst_512_count}",
        f"fifo_blocks_with_near_clip={nonzero_blocks}",
        f"max_near_clip_events_in_one_fifo_payload={max_block_count}",
        f"output={args.output}",
    ]
    args.summary_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
