# IIS3DWB Vibration Data Investigation Report

Date: 2026-07-11

## Scope

This report summarizes the investigation of `iis3dwb_acc.dat`, a DATALOG2 IIS3DWB accelerometer capture. The goal was to understand whether the observed rolling RMS burst pattern, including increasing burst spacing over time, represents real vibration or a data acquisition/storage artifact.

The current confirmed conclusion is that the original file was initially misparsed as a pure `int16 XYZ` stream. The `.dat` file contains DATALOG2 framing metadata: repeated SD buffer byte counters and repeated timestamp records. When those metadata bytes are interpreted as accelerometer samples, they create the high-amplitude outlier population that drives the widening rolling RMS pattern.

The strongest supporting observations are:

- firmware code explicitly writes the metadata records into the stream
- the high-amplitude samples line up with those metadata byte positions
- after decoding the DATALOG2 framing, the large widening RMS envelope disappears
- a similar pattern was reported when the sensor was not connected to anything

## Data And Parser Assumptions

The initial parser assumption was:

- file: `iis3dwb_acc.dat`
- sample rate: `26667 Hz`
- data type: little-endian signed `int16`
- sample layout: interleaved `X, Y, Z`
- full-scale setting from firmware: `16g`
- leading offset/header: `4 bytes`
- sample rows under this incorrect parse: `82,499,412`
- duration under this incorrect parse: `51.561488 minutes`

That assumption is now known to be incomplete.

The confirmed DATALOG2 framing is:

- each SD circular-buffer item is `48128 bytes`
- each item starts with a `4-byte byte_counter`
- the remaining `48124 bytes` are stream payload
- inside the payload, every `1000` XYZ samples are followed by an `8-byte timestamp`
- one XYZ sample is `6 bytes`: `X/Y/Z int16`

The corrected decoded stream is:

- decoded file: `derived_data/iis3dwb_acc_decoded_xyz.dat`
- decoded sample rows: `82,382,714`
- decoded duration: `51.488553 minutes`
- removed SD item headers: `10,285`
- removed timestamp records: `82,382`
- decoded output format: pure little-endian `int16 X/Y/Z`, no leading offset

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
- `iis3dwb_acc_set_samples_per_ts(1000, NULL)` enables one timestamp record per `1000` samples.
- IIS3DWB FIFO records are read as `7 bytes/sample`: `tag + X + Y + Z`.
- The firmware then strips the 1-byte FIFO tag and compacts data into `6 bytes/sample`: `X/Y/Z int16`.
- The DatalogAppTask stream layer writes `8` timestamp bytes into the FileX stream whenever the `samples_per_ts` counter is reached.
- The FileX circular buffer packs stream bytes into fixed-size SD items and writes a `4-byte byte_counter` at the start of each item.
- The SD write layer writes those framed stream bytes directly to `.dat` through FileX.

Relevant firmware path details:

- IIS3DWB FIFO read and tag removal:
  `/Users/arahman/Documents/fp-sns-datalog2/Projects/eLooM_Components/SensorManager/Src/IIS3DWBTask.c`
- IIS3DWB `samples_per_ts = 1000`:
  `/Users/arahman/Documents/fp-sns-datalog2/Projects/STM32L4R9ZI-STWIN/Applications/DATALOG2/PnPL/AppModel/Src/App_model_Iis3dwb_Acc.c`
- Timestamp insertion into USB/FileX stream:
  `/Users/arahman/Documents/fp-sns-datalog2/Projects/STM32L4R9ZI-STWIN/Applications/DATALOG2/Core/Src/DatalogAppTask.c`
- DATALOG2 circular-buffer item header:
  `/Users/arahman/Documents/fp-sns-datalog2/Middlewares/ST/CircularBufferDL2/CircularBufferDL2.c`
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
- if pattern disappears after removing or suppressing outliers, RMS is dominated by non-signal high-amplitude samples

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

Real vibration should not care about byte position inside a file. This made the initial RMS pattern look like a storage, DMA, FIFO packing, or buffer-handling artifact.

This diagnostic was the clue that the apparent high-amplitude samples were tied to file framing, not physical vibration.

## DATALOG2 Framing Confirmation

The boundary diagnostic originally pointed toward 512-byte and SD-buffer alignment. Later firmware inspection clarified that these were not arbitrary corruption sites; they were DATALOG2 framing positions.

Two metadata layers were confirmed:

1. SD item header:

```text
48128-byte SD item =
  4-byte byte_counter
  48124-byte stream payload
```

2. Timestamp records inside the payload:

```text
6008-byte payload frame =
  6000 bytes of samples = 1000 XYZ samples * 6 bytes/sample
  8-byte timestamp
```

This explains the earlier outlier findings:

- high-amplitude events were strongly concentrated at byte offset `0 mod 48128` before SD headers were removed
- after removing SD headers, the remaining high-amplitude events concentrated at byte offsets `6000`, `6002`, `6004`, and `6006 mod 6008`
- those offsets are exactly the four `int16` words that make up an 8-byte timestamp when misread as accelerometer samples

So the large outlier population was mostly metadata interpreted as `int16 XYZ`, not physical sensor output and not random SD corruption.

A decoder was added:

- script: `src/create_decoded_datalog2_data.py`
- decoded output: `derived_data/iis3dwb_acc_decoded_xyz.dat`
- metadata: `derived_data/iis3dwb_acc_decoded_metadata.json`

Decoder summary:

```text
raw_file_bytes=494996480
sd_item_bytes=48128
sd_item_header_bytes=4
sd_items_decoded=10285
samples_per_timestamp=1000
timestamp_bytes=8
timestamp_records_removed=82382
decoded_samples=82382714
decoded_duration_minutes=51.488553
```

Outlier collapse after correct decoding:

```text
naive_parse_rows_abs_ge_10000=160963
decoded_rows_abs_ge_10000=779

naive_parse_rows_abs_ge_30000=20929
decoded_rows_abs_ge_30000=150

naive_parse_vector_rms_counts≈2450
decoded_vector_rms_counts=1390.220512
```

This is the strongest confirmation in the investigation. The RMS burst artifact was primarily caused by parsing DATALOG2 metadata as vibration data.

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

Interpretation at that stage:

The raw RMS envelope rises when near-clipped sample count rises. After removing the near-clipped rows, the relationship disappears and even flips sign.

This suggested:

- the raw RMS bursts were driven by high-amplitude non-signal samples
- the increasing apparent burst spacing was mainly changing timing/count of those samples
- the widening pattern is not reliable evidence of changing physical vibration

The later DATALOG2 framing confirmation explains what those samples were: SD byte-counter and timestamp metadata decoded as `int16 XYZ`.

## Cleaned And Winsorized Data

A script was created before the framing issue was fully understood to derive cleaned data files while preserving the original raw file:

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

These files are now best understood as diagnostic artifacts. They showed that suppressing high-amplitude values removed the rolling RMS envelope, but they did not solve the root issue. The root issue was that DATALOG2 metadata bytes were being parsed as samples.

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

This was an important refinement at the time. It showed that the pattern was not only caused by the most obvious near-clipping samples above `30000`; it was also driven by a larger population of high-amplitude values below clipping.

Counts by threshold in the original file:

```text
abs(sample) >= 10000: 160963 contaminated rows, 0.195108%
abs(sample) >= 15000: 149835 contaminated rows, 0.181619%
abs(sample) >= 20000: 81312 contaminated rows, 0.098561%
abs(sample) >= 25000: 53832 contaminated rows, 0.065251%
abs(sample) >= 30000: 20929 contaminated rows, 0.025369%
```

The later DATALOG2 decoder explains why removing only `>=30000` rows was insufficient: timestamp bytes and SD byte-counter bytes can decode into many different signed `int16` amplitudes, not only full-scale clips. Winsorizing at `+/-10000` suppressed that broad metadata-derived population, but proper decoding removes the metadata instead.

## Current Understanding

The current understanding is:

1. The original `.dat` file contains a valid DATALOG2 IIS3DWB stream, not a pure sample-only binary stream.
2. The first analysis incorrectly treated DATALOG2 metadata as accelerometer samples.
3. The apparent high-amplitude outlier population was mostly made of SD byte-counter headers and timestamp bytes.
4. These metadata bytes were strong enough to dominate rolling RMS.
5. The increasing RMS burst spacing was a parser/framing artifact, not reliable evidence of changing physical vibration.
6. After removing DATALOG2 framing, the decoded quartile dynamic RMS is stable:

```text
Q1 vector RMS=1404.08396 counts
Q2 vector RMS=1407.26991 counts
Q3 vector RMS=1407.24249 counts
Q4 vector RMS=1399.53798 counts
```

7. The decoded first half vs second half dynamic RMS is also stable:

```text
first_half_vector_rms=1405.67786 counts
second_half_vector_rms=1403.39568 counts
```

8. The observation that the same widening pattern appears with an unconnected sensor is consistent with a parser/framing artifact.

## Current Working Hypotheses

The primary hypothesis is now confirmed:

### Confirmed: DATALOG2 Metadata Was Parsed As Sample Data

The file contains repeated metadata records that must be removed before interpreting the stream as `int16 XYZ` samples:

- `4-byte byte_counter` at the start of each `48128-byte` SD item
- `8-byte timestamp` after each `1000` XYZ samples

When these bytes are interpreted as signed accelerometer counts, they naturally create values across the `10000-30000` range and sometimes near full-scale. The rolling RMS formula then amplifies those sparse large values into visible bursts.

### Residual High-Amplitude Samples

After correct decoding, a small number of high-amplitude samples remain:

```text
decoded_rows_abs_ge_10000=779
decoded_rows_abs_ge_30000=150
```

Those residual samples may be real transient sensor values, startup/shutdown effects, remaining framing edge cases, or true rare corruption. They are now small enough that they do not explain the original widening RMS pattern.

## What Seems Less Likely Now

The following explanations are now less likely:

- actual vibration amplitude slowly changing over time
- rotating equipment gradually changing speed
- normal sensor noise
- purely mathematical rolling-window artifact without real outlier events
- random SD corruption as the main explanation
- FIFO tag stripping as the main explanation
- SD DMA/cache coherency as the main explanation

The file still needs correct parsing, but the problem is no longer an unknown parser setting. The relevant format details have been identified in the firmware and confirmed against the data.

## Practical Analysis Guidance

Use the decoded pure XYZ file for normal vibration analysis:

```bash
python src/interactive_analysis_browser.py --input derived_data/iis3dwb_acc_decoded_xyz.dat
```

The loader can also auto-detect and decode the original DATALOG2 file:

```bash
python src/interactive_analysis_browser.py --input iis3dwb_acc.dat
```

Use the original raw file directly only if the script uses the updated `src/iis3dwb_data.py` loader. Older scripts or external tools that assume pure XYZ will recreate the false RMS bursts.

The winsorized and cleaned-removed files are retained as diagnostic artifacts:

```bash
python src/interactive_analysis_browser.py --input derived_data/iis3dwb_acc_winsorized.dat
python src/interactive_analysis_browser.py --input derived_data/iis3dwb_acc_cleaned_removed.dat
```

They are no longer preferred for primary analysis because they suppress symptoms instead of decoding the actual file format.

## Current Summary

The dataset should not be interpreted by treating the original `.dat` bytes as a pure `int16 XYZ` stream.

The current confirmed interpretation is:

> The observed increasing periodic RMS gaps are primarily caused by DATALOG2 metadata bytes being interpreted as accelerometer samples. The 512-byte/SD alignment and `48128`-byte FileX buffer alignment identify the repeated byte-counter headers. After those headers are removed, the remaining large samples align with the 8-byte timestamp records inserted every `1000` samples. Once both metadata layers are removed, the quartile and half-duration RMS values are stable, so the widening RMS pattern should not be treated as physical vibration.

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
- `src/create_decoded_datalog2_data.py`

Main outputs:

- `derived_data/iis3dwb_acc_decoded_xyz.dat`
- `derived_data/iis3dwb_acc_decoded_metadata.json`
- `outputs/decoded_analysis_summary.csv`
- `outputs/decoded_window_artifact_diagnostics.png`
- `outputs/decoded_rolling_metrics.csv`
- `outputs/decoded_plots/`
- `outputs/window_artifact_diagnostics.png`
- `outputs/fifo_boundary_diagnostics.png`
- `outputs/fifo_boundary_summary.txt`
- `outputs/burst_timing_diagnostics.png`
- `outputs/burst_timing_summary.txt`
- `derived_data/iis3dwb_acc_winsorized.dat`
- `derived_data/iis3dwb_acc_cleaned_removed.dat`
- `derived_data/iis3dwb_acc_contamination_mask.npy`
- `derived_data/iis3dwb_acc_cleaning_metadata.json`
