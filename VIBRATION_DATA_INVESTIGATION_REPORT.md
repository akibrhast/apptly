# IIS3DWB Vibration Data Investigation Report

Date: 2026-07-11

## Scope

This report summarizes the investigation of `iis3dwb_acc.dat`, a 51.56 minute IIS3DWB accelerometer capture. The goal was to understand whether the observed rolling RMS burst pattern, including increasing burst spacing over time, represents real vibration or a data acquisition/storage artifact.

The current working conclusion is that the raw rolling RMS pattern is dominated by corrupt/outlier samples, not physical vibration. The strongest supporting observation is that the same widening pattern was reported when the sensor was not connected to anything.

## Data And Parser Assumptions

The dataset is currently parsed as:

- file: `iis3dwb_acc.dat`
- sample rate: `26667 Hz`
- leading offset/header: `4 bytes`
- data type: little-endian signed `int16`
- sample layout: interleaved `X, Y, Z`
- full-scale setting from firmware: `16g`
- original sample rows: `82,499,412`
- duration: `51.561488 minutes`

The parser and constants are in `src/iis3dwb_data.py`.

## Initial Observations

The first plots showed a repeating RMS burst pattern over the full capture. The burst spacing appeared to slowly increase over time.

Initial time-domain and frequency-domain plots included:

- `outputs/amplitude_vs_time.png`
- `outputs/frequency_vs_time_spectrogram.png`
- `outputs/frequency_vs_amplitude.png`

Early spectral analysis showed a strong low-frequency component around `7.7 Hz`, with additional peaks. At that stage it was still possible that the signal represented a machine vibration or rotating system.

## Initial Guesses

The first plausible explanations were:

- actual vibration amplitude was changing over time
- a rotating system was drifting in speed
- rolling RMS windowing was creating a visual beat/alias pattern
- rare high-amplitude samples were driving the RMS envelope
- the data parser assumptions were wrong

The rolling RMS graph raised concern because its periodic-looking peaks were much stronger than expected and did not look like a simple stationary vibration.

## Initial Steps Taken

The project folder was created with:

- `iis3dwb_acc.dat`
- Python environment `.venv`
- shared loader `src/iis3dwb_data.py`
- analysis script `src/analyze_iis3dwb.py`
- plotting script `src/plot_iis3dwb.py`
- interactive matplotlib browser `src/interactive_analysis_browser.py`

The interactive browser was built to inspect:

- amplitude vs time
- frequency vs time
- frequency vs amplitude
- rolling RMS
- band RMS
- spectrograms
- quartile PSDs
- clipping/shock metrics
- crest factor
- kurtosis
- spectral centroid
- waterfall plots
- correlation/coherence

The browser can be launched with:

```bash
python src/interactive_analysis_browser.py
```

## Firmware Code Investigation

The originally suspected codebase was later replaced by a more relevant local firmware path:

`/Users/arahman/Documents/fp-sns-datalog2/Projects/STM32L4R9ZI-STWIN`

Important code findings:

- IIS3DWB is modeled as `iis3dwb_acc` in `Applications/DATALOG2/PnPL/AppModel/Src/App_model_Iis3dwb_Acc.c`.
- The model sets full-scale to `16g`.
- The data format is `int16`.
- The firmware ODR is `26667 Hz`, not `26700 Hz`.
- IIS3DWB FIFO records are read as `7 bytes/sample`: `tag + X + Y + Z`.
- The firmware then strips the 1-byte FIFO tag and compacts data into `6 bytes/sample`: `X/Y/Z int16`.
- The SD write layer writes stream bytes directly to `.dat` through FileX.

Relevant firmware path details:

- IIS3DWB FIFO read and tag removal:
  `/Users/arahman/Documents/fp-sns-datalog2/Projects/eLooM_Components/SensorManager/Src/IIS3DWBTask.c`
- SD data write:
  `/Users/arahman/Documents/fp-sns-datalog2/Projects/STM32L4R9ZI-STWIN/Applications/DATALOG2/FileX/App/filex_dctrl_class.c`
- SD chunk size calculation:
  `/Users/arahman/Documents/fp-sns-datalog2/Projects/STM32L4R9ZI-STWIN/Applications/DATALOG2/PnPL/AppModel/Src/App_model.c`
- FileX SD driver:
  `/Users/arahman/Documents/fp-sns-datalog2/Projects/STM32L4R9ZI-STWIN/Applications/DATALOG2/FileX/Target/fx_stm32_sd_driver.h`

## SD Card And Storage Findings

The card was reported as a `120 GB` SD card. That matters because large cards are usually SDXC-class media, often formatted as exFAT by default.

Firmware observations:

- FileX logical sector size is configured as `512 bytes`.
- FileX writes use `HAL_SD_WriteBlocks_DMA`.
- SD stream buffer sizes are rounded to multiples of `512` for efficiency.
- `FX_ENABLE_EXFAT` is commented out in `fx_user.h`.

This means the firmware is strongly tied to 512-byte logical sector writes. If the 120 GB card was used successfully, it was likely formatted in a FileX-compatible way, or the active firmware/library setup supports the mounted volume despite the visible config. The observed data artifact still lines up with 512-byte boundaries.

## Window Artifact Diagnostic

A six-panel rolling RMS diagnostic was created:

- script: `src/window_artifact_diagnostics.py`
- output: `outputs/window_artifact_diagnostics.png`

It compared:

- 5-second window / 5-second hop
- 5-second window / 1-second hop
- 5-second window / 5-second hop shifted by 2.5 seconds
- 4-, 6-, and 7-second windows
- raw RMS vs near-clip-removed RMS vs winsorized RMS
- near-clip count over time

Purpose:

- if peaks move when window start shifts, suspect window-boundary artifact
- if spacing changes with window size, suspect beat/alias artifact
- if peaks stay at absolute timestamps, actual captured events exist
- if pattern disappears after removing or suppressing outliers, RMS is dominated by corrupt samples

Observation:

The RMS pattern is strongly influenced by high-amplitude samples. Suppressing outliers materially changes the visible RMS envelope.

## FIFO And Boundary Diagnostic

A firmware-aware boundary diagnostic was created:

- script: `src/fifo_boundary_diagnostics.py`
- plot: `outputs/fifo_boundary_diagnostics.png`
- summary: `outputs/fifo_boundary_summary.txt`

Important numeric results:

```text
near_clip_threshold=30000
near_clip_events=21894
near_clip_by_axis X/Y/Z=7286/7297/7311
worst_fifo_position=170 count=385
worst_byte_mod_1536=0 count=343
worst_byte_mod_512=0 count=1001
fifo_blocks_with_near_clip=20011
max_near_clip_events_in_one_fifo_payload=76
```

Interpretation:

The near-clipped samples are not randomly distributed. They concentrate at byte positions tied to:

- 512-byte SD sector boundaries
- 1536-byte compact IIS3DWB payload boundaries
- 256-sample FIFO group positions

Real vibration should not care about byte position inside a file. Storage, DMA, FIFO packing, and buffer handling can.

This diagnostic explains where corruption tends to land in the stream.

## Burst Timing Diagnostic

A timing diagnostic was created to test why the visible RMS bursts become farther apart over time:

- script: `src/burst_timing_diagnostics.py`
- plot: `outputs/burst_timing_diagnostics.png`
- summary: `outputs/burst_timing_summary.txt`

Important numeric results:

```text
near_clip_events=21894
raw_rms_vs_near_clip_window_count_corr=0.659152
cleaned_rms_vs_near_clip_window_count_corr=-0.341157
raw_rms_vs_cleaned_rms_corr=0.478133
raw_rms_peak_count=298
raw_rms_peak_gap_minutes_median=0.066667
raw_rms_peak_gap_minutes_first=0.050000
raw_rms_peak_gap_minutes_last=0.883333
```

Interpretation:

The raw RMS envelope rises when near-clipped sample count rises. After removing the near-clipped rows, the relationship disappears and even flips sign.

This suggests:

- the raw RMS bursts are driven by corrupt/outlier samples
- the increasing apparent burst spacing is mainly changing timing/count of those corrupt samples
- the widening pattern is not reliable evidence of changing physical vibration

This diagnostic explains when enough corruption accumulates to make the rolling RMS spike.

## Cleaned And Winsorized Data

A script was created to derive cleaned data files while preserving the original raw file:

- script: `src/create_cleaned_data.py`
- metadata: `derived_data/iis3dwb_acc_cleaning_metadata.json`

Outputs:

- `derived_data/iis3dwb_acc_winsorized.dat`
- `derived_data/iis3dwb_acc_cleaned_removed.dat`
- `derived_data/iis3dwb_acc_contamination_mask.npy`
- `derived_data/iis3dwb_acc_cleaning_metadata.json`

Metadata summary:

```text
original_samples=82499412
original_duration_minutes=51.561488
near_clip_threshold=30000
winsorize_threshold=10000
contaminated_rows_removed=20929
contaminated_rows_percent=0.0253687
contaminated_axis_samples_x_y_z=7286/7297/7311
cleaned_removed_samples=82478483
winsorized_axis_samples_clipped_total=262147
```

Definitions:

- winsorized data preserves timing and sample count but clamps extreme values to `+/-10000`
- cleaned-removed data deletes any XYZ row where any axis has `abs(sample) >= 30000`

The original file is not modified.

## Important Later Observation

When the interactive browser was run on:

```bash
python src/interactive_analysis_browser.py --input derived_data/iis3dwb_acc_winsorized.dat
```

the increasing periodic RMS gaps disappeared.

When it was run on:

```bash
python src/interactive_analysis_browser.py --input derived_data/iis3dwb_acc_cleaned_removed.dat
```

the pattern was still visible.

This was an important refinement. It showed that the pattern is not only caused by the most obvious near-clipping samples above `30000`. It is also driven by a larger population of high-amplitude outliers below clipping.

Counts by threshold in the original file:

```text
abs(sample) >= 10000: 160963 contaminated rows, 0.195108%
abs(sample) >= 15000: 149835 contaminated rows, 0.181619%
abs(sample) >= 20000: 81312 contaminated rows, 0.098561%
abs(sample) >= 25000: 53832 contaminated rows, 0.065251%
abs(sample) >= 30000: 20929 contaminated rows, 0.025369%
```

This explains why removing only `>=30000` rows was insufficient. Winsorizing at `+/-10000` suppresses a much broader outlier population.

## Current Understanding

The current understanding is:

1. The raw `.dat` file contains a real IIS3DWB accelerometer stream, but it is contaminated by sparse high-amplitude outliers.
2. The most extreme outliers appear as near-clipped values around `+/-30000` to `+/-32768`.
3. A larger outlier population exists between `10000` and `30000` raw counts.
4. These outliers are strong enough to dominate rolling RMS.
5. The outliers show boundary correlations with 512-byte SD sectors and 1536-byte IIS3DWB payload structure.
6. The increasing RMS burst spacing is mostly a visualization of changing outlier timing/count, not a reliable vibration trend.
7. The observation that the same pattern appears with an unconnected sensor strongly supports a logger/storage/firmware artifact.

## Current Working Hypotheses

The most likely artifact mechanisms are:

### Partial Byte Corruption

Each axis sample is a signed 16-bit value. If one byte is stale, shifted, or overwritten, the decoded value can become very large without necessarily reaching full-scale clipping.

This could explain the `10000-30000` outliers.

### FIFO Tag Or Packing Error

The IIS3DWB FIFO data starts as `7 bytes/sample`:

```text
tag + X_L X_H + Y_L Y_H + Z_L Z_H
```

The firmware compacts this to:

```text
X_L X_H + Y_L Y_H + Z_L Z_H
```

If tag stripping or in-place compaction is occasionally off, stale, or interrupted, false high-amplitude `int16` values can appear.

### SD DMA Or Cache Coherency Issue

The SD path uses DMA writes. If buffer ownership or cache coherency is imperfect, the SD card may receive partially stale data.

This would naturally create boundary-correlated corruption and a range of amplitudes, not only full-scale clips.

### SD Card Latency Or Queue Pressure

A 120 GB SD card may have variable internal write latency. If the card stalls or the FileX/SD write task falls behind, sensor buffers may pile up or be reused under pressure.

This could explain why corruption timing changes over the 51-minute run.

### FileX/FAT Allocation Or Sector Boundary Effects

The firmware writes 512-byte sector-oriented chunks. The strongest diagnostic signal appears at byte offset `0` modulo `512`.

This does not prove FileX itself is corrupting data, but it strongly suggests the artifact is coupled to the storage/write boundary.

## What Seems Less Likely Now

The following explanations are now less likely:

- actual vibration amplitude slowly changing over time
- rotating equipment gradually changing speed
- normal sensor noise
- purely mathematical rolling-window artifact without real outlier events
- a single wrong parser setting for the whole file

A parser issue would usually create continuous misinterpretation, not sparse high-amplitude events concentrated at storage/FIFO boundaries.

## Practical Analysis Guidance

Use the original raw file for forensic diagnostics only.

Use the winsorized file for time-aligned visualization:

```bash
python src/interactive_analysis_browser.py --input derived_data/iis3dwb_acc_winsorized.dat
```

Use the cleaned-removed file for statistics where dropping contaminated rows is acceptable:

```bash
python src/interactive_analysis_browser.py --input derived_data/iis3dwb_acc_cleaned_removed.dat
```

However, note that `cleaned_removed` with threshold `30000` still leaves many `10000-30000` outliers. For stricter row removal:

```bash
python src/create_cleaned_data.py --near-clip-threshold 10000 --output-dir derived_data_t10000
```

For most time-based plots, the winsorized file is currently the better practical choice because it preserves the original timebase and suppresses the outlier-driven RMS envelope.

## Current Summary

The dataset should not be interpreted using raw rolling RMS alone.

The current best interpretation is:

> The observed increasing periodic RMS gaps are primarily caused by sparse, boundary-correlated high-amplitude outliers in the captured data stream. The 512-byte/SD alignment explains where corruption tends to appear, and the burst timing diagnostic shows that the visible raw RMS envelope follows the changing timing/count of those outliers. Since a similar pattern appears with an unconnected sensor, the artifact is likely in acquisition, buffering, packing, DMA, FileX, SD card behavior, or their interaction, rather than in physical vibration.

## Generated Artifacts

Main scripts:

- `src/iis3dwb_data.py`
- `src/analyze_iis3dwb.py`
- `src/plot_iis3dwb.py`
- `src/interactive_analysis_browser.py`
- `src/window_artifact_diagnostics.py`
- `src/fifo_boundary_diagnostics.py`
- `src/burst_timing_diagnostics.py`
- `src/create_cleaned_data.py`

Main outputs:

- `outputs/window_artifact_diagnostics.png`
- `outputs/fifo_boundary_diagnostics.png`
- `outputs/fifo_boundary_summary.txt`
- `outputs/burst_timing_diagnostics.png`
- `outputs/burst_timing_summary.txt`
- `derived_data/iis3dwb_acc_winsorized.dat`
- `derived_data/iis3dwb_acc_cleaned_removed.dat`
- `derived_data/iis3dwb_acc_contamination_mask.npy`
- `derived_data/iis3dwb_acc_cleaning_metadata.json`

