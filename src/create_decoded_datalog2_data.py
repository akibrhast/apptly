from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iis3dwb_data import (
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SAMPLES_PER_TIMESTAMP,
    DEFAULT_SD_ITEM_BYTES,
    DEFAULT_SD_ITEM_HEADER_BYTES,
    DEFAULT_TIMESTAMP_BYTES,
    decode_datalog2_xyz,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode an ST DATALOG2 IIS3DWB .dat stream into a pure int16 XYZ .dat file."
    )
    parser.add_argument("--input", type=Path, default=Path("iis3dwb_acc.dat"))
    parser.add_argument("--output-dir", type=Path, default=Path("derived_data"))
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--sd-item-bytes", type=int, default=DEFAULT_SD_ITEM_BYTES)
    parser.add_argument("--sd-item-header-bytes", type=int, default=DEFAULT_SD_ITEM_HEADER_BYTES)
    parser.add_argument("--samples-per-timestamp", type=int, default=DEFAULT_SAMPLES_PER_TIMESTAMP)
    parser.add_argument("--timestamp-bytes", type=int, default=DEFAULT_TIMESTAMP_BYTES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "iis3dwb_acc_decoded_xyz.dat"
    metadata_path = args.output_dir / "iis3dwb_acc_decoded_metadata.json"

    xyz, trailing_bytes, sd_items, timestamp_records = decode_datalog2_xyz(
        path=args.input,
        sd_item_bytes=args.sd_item_bytes,
        sd_item_header_bytes=args.sd_item_header_bytes,
        samples_per_timestamp=args.samples_per_timestamp,
        timestamp_bytes=args.timestamp_bytes,
    )

    decoded = np.memmap(output_path, dtype="<i2", mode="w+", shape=xyz.size)
    decoded[:] = xyz.reshape(-1)
    decoded.flush()

    max_abs = int(np.max(np.abs(xyz.astype(np.int32))))
    rows_ge_10000 = int(np.any(np.abs(xyz.astype(np.int32)) >= 10000, axis=1).sum())
    rows_ge_30000 = int(np.any(np.abs(xyz.astype(np.int32)) >= 30000, axis=1).sum())
    vector_rms = float(np.sqrt(np.mean(xyz.astype(np.float64) ** 2)))

    metadata = {
        "input": str(args.input),
        "output": str(output_path),
        "sample_rate_hz": args.sample_rate_hz,
        "samples": int(xyz.shape[0]),
        "duration_minutes": float(xyz.shape[0] / args.sample_rate_hz / 60.0),
        "sd_item_bytes": args.sd_item_bytes,
        "sd_item_header_bytes": args.sd_item_header_bytes,
        "sd_items_decoded": sd_items,
        "samples_per_timestamp": args.samples_per_timestamp,
        "timestamp_bytes": args.timestamp_bytes,
        "timestamp_records_removed": timestamp_records,
        "trailing_bytes_ignored": trailing_bytes,
        "decoded_file_is_pure_int16_xyz": True,
        "max_abs_count": max_abs,
        "rows_abs_ge_10000": rows_ge_10000,
        "rows_abs_ge_30000": rows_ge_30000,
        "raw_count_vector_rms": vector_rms,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {output_path}")
    print(f"Wrote {metadata_path}")
    print(f"samples={metadata['samples']}")
    print(f"duration_minutes={metadata['duration_minutes']:.6f}")
    print(f"removed_sd_headers={sd_items}")
    print(f"removed_timestamp_records={timestamp_records}")
    print(f"rows_abs_ge_10000={rows_ge_10000}")
    print(f"rows_abs_ge_30000={rows_ge_30000}")
    print(f"raw_count_vector_rms={vector_rms:.6f}")


if __name__ == "__main__":
    main()
