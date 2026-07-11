from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iis3dwb_data import DEFAULT_OFFSET_BYTES, DEFAULT_SAMPLE_RATE_HZ


def chunk_starts(total_samples: int, chunk_samples: int) -> range:
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    return range(0, total_samples, chunk_samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create derived IIS3DWB .dat files with near-clipped samples treated as contamination. "
            "The original file is not modified."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("iis3dwb_acc.dat"))
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--offset-bytes", type=int, default=DEFAULT_OFFSET_BYTES)
    parser.add_argument("--near-clip-threshold", type=int, default=30000)
    parser.add_argument("--winsorize-threshold", type=int, default=10000)
    parser.add_argument("--chunk-samples", type=int, default=2_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("derived_data"))
    parser.add_argument("--write-cleaned-removed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-winsorized", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.offset_bytes % 2:
        raise ValueError("offset_bytes must be even because data is int16 aligned")
    if args.near_clip_threshold <= 0:
        raise ValueError("near_clip_threshold must be positive")
    if args.winsorize_threshold <= 0:
        raise ValueError("winsorize_threshold must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    winsorized_path = args.output_dir / "iis3dwb_acc_winsorized.dat"
    cleaned_path = args.output_dir / "iis3dwb_acc_cleaned_removed.dat"
    mask_path = args.output_dir / "iis3dwb_acc_contamination_mask.npy"
    metadata_path = args.output_dir / "iis3dwb_acc_cleaning_metadata.json"

    raw = np.memmap(args.input, dtype="<i2", mode="r")
    offset_words = args.offset_bytes // 2
    usable_words = ((raw.size - offset_words) // 3) * 3
    total_samples = usable_words // 3
    trailing_words = raw.size - offset_words - usable_words
    xyz = raw[offset_words : offset_words + usable_words].reshape(total_samples, 3)

    contamination_mask = np.lib.format.open_memmap(mask_path, mode="w+", dtype=bool, shape=(total_samples,))
    if args.write_winsorized:
        winsorized = np.memmap(winsorized_path, dtype="<i2", mode="w+", shape=raw.shape)
        if offset_words:
            winsorized[:offset_words] = raw[:offset_words]
        if trailing_words:
            winsorized[offset_words + usable_words :] = raw[offset_words + usable_words :]
    else:
        winsorized = None

    cleaned_file = cleaned_path.open("wb") if args.write_cleaned_removed else None
    if cleaned_file is not None and args.offset_bytes:
        cleaned_file.write(raw[:offset_words].tobytes())

    contaminated_rows = 0
    contaminated_axis_samples = np.zeros(3, dtype=np.int64)
    cleaned_rows = 0
    clipped_axis_samples = 0

    for start in chunk_starts(total_samples, args.chunk_samples):
        stop = min(total_samples, start + args.chunk_samples)
        block = np.asarray(xyz[start:stop])
        contamination = np.any(np.abs(block.astype(np.int32)) >= args.near_clip_threshold, axis=1)
        contamination_mask[start:stop] = contamination

        contaminated_rows += int(contamination.sum())
        contaminated_axis_samples += (np.abs(block.astype(np.int32)) >= args.near_clip_threshold).sum(axis=0)

        if winsorized is not None:
            clipped = np.clip(block, -args.winsorize_threshold, args.winsorize_threshold).astype("<i2", copy=False)
            winsorized[offset_words + start * 3 : offset_words + stop * 3] = clipped.reshape(-1)
            clipped_axis_samples += int((block != clipped).sum())

        if cleaned_file is not None:
            clean_block = block[~contamination].astype("<i2", copy=False)
            cleaned_file.write(clean_block.tobytes())
            cleaned_rows += int(clean_block.shape[0])

    contamination_mask.flush()
    if winsorized is not None:
        winsorized.flush()
    if cleaned_file is not None:
        if trailing_words:
            cleaned_file.write(raw[offset_words + usable_words :].tobytes())
        cleaned_file.close()

    metadata = {
        "input": str(args.input),
        "sample_rate_hz": args.sample_rate_hz,
        "offset_bytes": args.offset_bytes,
        "near_clip_threshold": args.near_clip_threshold,
        "winsorize_threshold": args.winsorize_threshold,
        "original_samples": int(total_samples),
        "original_duration_minutes": total_samples / args.sample_rate_hz / 60.0,
        "contaminated_rows_removed": int(contaminated_rows),
        "contaminated_rows_percent": contaminated_rows / total_samples * 100.0,
        "contaminated_axis_samples_x_y_z": [int(v) for v in contaminated_axis_samples],
        "cleaned_removed_samples": int(cleaned_rows) if args.write_cleaned_removed else None,
        "cleaned_removed_duration_minutes_if_retimed": cleaned_rows / args.sample_rate_hz / 60.0
        if args.write_cleaned_removed
        else None,
        "winsorized_axis_samples_clipped_total": int(clipped_axis_samples) if args.write_winsorized else None,
        "outputs": {
            "winsorized_dat": str(winsorized_path) if args.write_winsorized else None,
            "cleaned_removed_dat": str(cleaned_path) if args.write_cleaned_removed else None,
            "contamination_mask_npy": str(mask_path),
            "metadata_json": str(metadata_path),
        },
        "notes": [
            "The winsorized .dat preserves the original sample count, timing, header, and trailing bytes.",
            "The cleaned_removed .dat drops any XYZ row where any axis is near-clipped; use the mask to map back to original time.",
            "Near-clipped rows are treated as contamination, not vibration.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {metadata_path}")
    if args.write_winsorized:
        print(f"Wrote {winsorized_path}")
    if args.write_cleaned_removed:
        print(f"Wrote {cleaned_path}")
    print(f"Wrote {mask_path}")


if __name__ == "__main__":
    main()
