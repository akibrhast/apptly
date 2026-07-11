# Apptly IIS3DWB Analysis Project

This project analyzes `iis3dwb_acc.dat`, a raw binary IIS3DWB accelerometer capture.

## Layout

- `iis3dwb_acc.dat` - copied raw accelerometer data file.
- `src/iis3dwb_data.py` - shared binary loader and constants.
- `src/analyze_iis3dwb.py` - summary, quartile, and spectral analysis.
- `src/plot_iis3dwb.py` - amplitude/time, spectrogram, and spectrum plots.
- `src/window_artifact_diagnostics.py` - six-panel rolling-RMS window sensitivity checks.
- `src/fifo_boundary_diagnostics.py` - firmware-aware near-clipping checks against FIFO and SD block offsets.
- `src/burst_timing_diagnostics.py` - compares widening RMS bursts against near-clip event timing.
- `src/create_cleaned_data.py` - creates derived cleaned/winsorized `.dat` files without modifying the original.
- `derived_data/` - derived data files and contamination metadata.
- `outputs/` - generated reports and images.
- `.venv/` - local Python environment.

## Assumptions

The parser assumes:

- sample rate: `26667 Hz`
- first `4` bytes are a leading marker/header
- data is little-endian signed `int16`
- samples are interleaved as `X, Y, Z`

The scripts report raw counts by default. Use `--full-scale-g 2`, `4`, `8`, or `16` to additionally convert counts to `g`.

## Run

```bash
source .venv/bin/activate
python src/analyze_iis3dwb.py
python src/plot_iis3dwb.py
```

Generated files are written to `outputs/`.

## Interactive Browser

```bash
source .venv/bin/activate
python src/interactive_analysis_browser.py
```

The browser opens a matplotlib window with toolbar zoom/pan support. Use the on-screen `Previous` and `Next` buttons, or keyboard shortcuts: right arrow/`n` for next, left arrow/`p` for previous, and `q` to close.

The browser includes amplitude, spectrogram, spectrum, rolling RMS, band RMS, RPM, harmonic, shock/clipping, correlation/coherence, vector magnitude, histogram, crest factor, kurtosis, spectral centroid, waterfall, quartile PSD, and band heatmap views.

To export rolling metrics without opening the GUI:

```bash
python src/interactive_analysis_browser.py --export-rolling-metrics outputs/rolling_metrics.csv --no-show
```

## Window Artifact Diagnostics

Generate the six-panel rolling-RMS diagnostic page:

```bash
python src/window_artifact_diagnostics.py
```

The output is written to `outputs/window_artifact_diagnostics.png`.

## FIFO Boundary Diagnostics

Generate the firmware-aware boundary diagnostic page:

```bash
python src/fifo_boundary_diagnostics.py
```

The outputs are written to `outputs/fifo_boundary_diagnostics.png` and `outputs/fifo_boundary_summary.txt`. This checks whether near-clipping samples concentrate at the 256-sample IIS3DWB FIFO payload, 1536-byte compact XYZ payload, 512-byte SD sector, or 4096-byte block offsets.

## Burst Timing Diagnostics

Generate the timing page that compares RMS bursts against near-clip event timing:

```bash
python src/burst_timing_diagnostics.py
```

The outputs are written to `outputs/burst_timing_diagnostics.png` and `outputs/burst_timing_summary.txt`. This is useful when the same widening RMS pattern appears with an unconnected sensor, because it shows whether the raw RMS envelope follows corrupt/near-clipped samples and whether the pattern remains after outlier removal.

## Derived Cleaned Data

Create derived data files that treat near-clipped rows as contamination:

```bash
python src/create_cleaned_data.py
```

The original `iis3dwb_acc.dat` is not modified. Outputs are written to `derived_data/`:

- `iis3dwb_acc_winsorized.dat` preserves the original sample count and timing, but clamps values to `+/-10000` raw counts.
- `iis3dwb_acc_cleaned_removed.dat` removes any XYZ row where any axis has `abs(sample) >= 30000`.
- `iis3dwb_acc_contamination_mask.npy` maps each original sample row to clean/contaminated status.
- `iis3dwb_acc_cleaning_metadata.json` records thresholds, row counts, and output paths.

For time-aligned plots, prefer the winsorized file. For statistics where dropping contaminated rows is acceptable, use the cleaned-removed file and metadata.
