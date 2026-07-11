# AGENTS.md

## Project Goal

Build and maintain a Python/matplotlib analysis tool for `iis3dwb_acc.dat`, a raw IIS3DWB 3-axis accelerometer vibration capture.

The tool must let the user inspect all analysis graphs interactively in matplotlib. The user should be able to zoom, pan, inspect data visually, and click through graphs using matplotlib controls/buttons.

## Data Assumptions

Use the shared loader in `src/iis3dwb_data.py`.

- Input file: `iis3dwb_acc.dat`
- Sample rate: `26700 Hz`
- First `4` bytes are a leading marker/header
- Data format: little-endian signed `int16`
- Sample layout: interleaved `X, Y, Z`
- Default units: raw counts
- Optional engineering units: convert to `g` using IIS3DWB full-scale sensitivity

Do not load the whole file into unnecessary duplicate arrays. Prefer `numpy.memmap`, chunked calculations, and downsampled views for visualization.

## Required Interactive Graph Browser

Create a matplotlib-based graph browser, preferably as:

```bash
python src/interactive_analysis_browser.py
```

Required behavior:

- Open one matplotlib window.
- Show one graph at a time.
- Provide clickable `Previous` and `Next` buttons to move through all graphs.
- Keep matplotlib's standard toolbar available for zoom, pan, and save.
- Keyboard shortcuts should also work:
  - Right arrow / `n`: next graph
  - Left arrow / `p`: previous graph
  - `q`: close
- Every graph must include:
  - clear title
  - x-axis label with units
  - y-axis label with units
  - legend when more than one line/series is shown
  - grid where useful
  - a short visible description of what the graph visualizes and how to interpret it
- Descriptions should be placed as a small text box inside the figure or as wrapped text below the axes.
- Use stable, readable colors for X/Y/Z and maintain them across plots.

## Required Graphs

Implement all graphs below in matplotlib. They can be computed on demand when the user navigates to them, but expensive metrics should be cached in memory during one run.

### 1. Amplitude vs Time

Plot downsampled/block RMS amplitude over the full 51-minute recording.

- X-axis: time in minutes
- Y-axis: RMS amplitude in raw counts or `g`
- Series: X, Y, Z
- Purpose: show broad vibration level changes over the full capture.

### 2. Frequency vs Time Spectrogram

Plot a spectrogram for at least the X axis. Ideally allow X/Y/Z selection.

- X-axis: time in minutes
- Y-axis: frequency in Hz
- Colorbar: power/frequency in dB
- Purpose: show how frequency content changes over time.

### 3. Frequency vs Amplitude Spectrum

Plot the amplitude spectrum from a representative central segment of the recording.

- X-axis: frequency in Hz
- Y-axis: amplitude in raw counts or `g`
- Series: X, Y, Z
- Purpose: show dominant vibration frequencies.

### 4. Rolling RMS Over Time

Compute rolling RMS in fixed windows, e.g. 1-second or 5-second windows.

- X-axis: time in minutes
- Y-axis: rolling RMS amplitude
- Series: X, Y, Z and optionally dynamic vector RMS
- Purpose: show slow vibration level trends and operational changes.

### 5. Band-Limited RMS Over Time

Compute RMS energy over time for frequency bands:

- `0-20 Hz`
- `20-100 Hz`
- `100-500 Hz`
- `500-2000 Hz`
- `2000 Hz-Nyquist`

Use Welch or FFT windows per time block.

- X-axis: time in minutes
- Y-axis: band RMS amplitude
- Series: frequency bands
- Purpose: show which frequency ranges change over time.

### 6. Peak Frequency / RPM Over Time

For each time window, find the strongest spectral peak above `1 Hz`.

- X-axis: time in minutes
- Left Y-axis: dominant frequency in Hz
- Optional right Y-axis: RPM, using `RPM = Hz * 60`
- Purpose: show whether the dominant vibration frequency/speed drifts.

### 7. Harmonic Analysis

Track the fundamental near `7.74 Hz` and its harmonics.

Recommended harmonics:

- `1x`
- `2x`
- `3x`
- `4x`
- `5x`
- `6x`
- `8x`
- `10x`

- X-axis: harmonic/order
- Y-axis: amplitude
- Series: X, Y, Z or quartiles
- Purpose: show rotating-machine harmonic structure.

### 8. Shock / Impulse Detection

Detect samples or windows with unusually high absolute amplitude.

Recommended thresholds:

- `abs(sample) >= 30000 counts`
- optional percentile threshold, e.g. `99.9th` or `99.99th`

Plots:

- event timeline: event time vs peak amplitude
- optional zoomed waveform around the largest event

Purpose: identify transient impacts or artifacts.

### 9. Clipping / Saturation Map

Plot where samples approach or hit sensor/raw `int16` limits.

- X-axis: time in minutes
- Y-axis: axis or event count per time bin
- Series: X, Y, Z
- Purpose: reveal clipping, sensor saturation, or logging artifacts.

### 10. Axis Correlation

Show how X/Y/Z relate to each other.

Required plots:

- correlation matrix heatmap
- optional scatter or density plot of X vs Y, X vs Z, Y vs Z using downsampled points

Purpose: show whether axes share a common vibration source.

### 11. Axis Coherence

Compute magnitude-squared coherence between axis pairs if `scipy` is available.

If `scipy` is not available, implement this with Welch cross-spectral density in NumPy.

Pairs:

- X-Y
- X-Z
- Y-Z

- X-axis: frequency in Hz
- Y-axis: coherence, `0-1`
- Purpose: show frequency-specific coupling between axes.

### 12. Dynamic Vector Magnitude

Remove per-axis mean, then compute:

```python
sqrt(x_dynamic**2 + y_dynamic**2 + z_dynamic**2)
```

Plots:

- vector magnitude over time, downsampled/block RMS
- histogram of dynamic vector magnitude

Purpose: direction-independent vibration energy.

### 13. Histogram / Probability Distribution

Plot amplitude distributions.

- X-axis: amplitude
- Y-axis: count or probability density
- Series: X, Y, Z
- Use enough bins to show tails without making the plot slow.
- Purpose: show skew, heavy tails, and impulsive behavior.

### 14. Crest Factor Over Time

For each time window:

```python
crest_factor = peak_abs / rms
```

- X-axis: time in minutes
- Y-axis: crest factor
- Series: X, Y, Z and optionally vector magnitude
- Purpose: identify impulsive periods; higher values mean sharper peaks relative to RMS.

### 15. Kurtosis Over Time

For each time window, compute kurtosis.

Use normal-distribution kurtosis convention consistently and state it in the description.

- X-axis: time in minutes
- Y-axis: kurtosis
- Series: X, Y, Z
- Purpose: identify non-Gaussian impulsive vibration.

### 16. Spectral Centroid / Bandwidth Over Time

Compute spectral centroid and spectral bandwidth for fixed windows.

- X-axis: time in minutes
- Y-axis: frequency in Hz
- Series: centroid and bandwidth, or separate graphs
- Purpose: show whether vibration energy shifts toward higher or lower frequencies.

### 17. Waterfall Spectrum

Plot stacked spectra over time.

- X-axis: frequency in Hz
- Y-axis: amplitude
- Stack/offset: time windows
- Purpose: make time-evolving spectra easier to inspect than a dense spectrogram.

### 18. Quartile PSD Overlay

Split the full capture into four equal time quartiles and overlay PSD curves.

- X-axis: frequency in Hz
- Y-axis: PSD or RMS spectral density
- Series: Q1, Q2, Q3, Q4
- Make one plot per axis or provide a selector.
- Purpose: visually compare first/second/third/fourth portions of the recording.

### 19. Frequency-Band Heatmap

Compute band RMS in time windows and plot a compact heatmap.

- X-axis: time in minutes
- Y-axis: frequency band
- Colorbar: RMS amplitude
- Purpose: summarize changes across frequency ranges without a full-resolution spectrogram.

### 20. Rolling Metrics Export

Export a CSV with time-windowed metrics.

Recommended columns:

- `time_start_s`
- `time_end_s`
- `rms_x`
- `rms_y`
- `rms_z`
- `vector_rms`
- `peak_abs_x`
- `peak_abs_y`
- `peak_abs_z`
- `crest_x`
- `crest_y`
- `crest_z`
- `kurtosis_x`
- `kurtosis_y`
- `kurtosis_z`
- `dominant_frequency_hz`
- `dominant_rpm`
- band RMS columns for all defined bands

Purpose: allow inspection in spreadsheets or downstream tools.

## Output Files

The interactive browser should not replace static output generation. Keep or extend static scripts that write plots to `outputs/`.

Recommended outputs:

- `outputs/analysis_summary.csv`
- `outputs/rolling_metrics.csv`
- `outputs/*.png` for each static plot

## Implementation Guidance

- Keep shared parsing and math helpers in `src/iis3dwb_data.py`.
- Add new reusable analysis functions rather than duplicating FFT/window logic.
- Prefer time units in minutes for full-record plots.
- Use raw counts unless the user passes `--full-scale-g`.
- If full-scale is provided, state the conversion in the figure description.
- Use `26700 Hz` as the default sample rate.
- Make expensive defaults reasonable. For interactive plots, do not attempt to render all 82 million samples directly.
- Use downsampling, block RMS, or windowed summaries for full-duration plots.
- For FFT-based time-window analysis, start with 1-second or 5-second windows and make the window size configurable.

## Validation

Before finishing changes, run:

```bash
.venv/bin/python src/analyze_iis3dwb.py
.venv/bin/python src/plot_iis3dwb.py
```

If an interactive browser is changed, also run:

```bash
.venv/bin/python src/interactive_analysis_browser.py --help
```

Do not require internet access at runtime.
