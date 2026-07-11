# Raw vs Winsorized vs Cleaned Comparison

This report compares the original raw capture with the derived artifact-suppressed data files.

- Figure: `outputs/data_version_comparison.png`
- CSV: `outputs/data_version_comparison_summary.csv`

## Summary Table

| version | samples | duration min | vector RMS | p99.9 abs | max abs | rows >=10000 | rows >=30000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 82499412 | 51.561488 | 2450.433 | 11832.0 | 32768 | 160963 | 20929 |
| winsorized | 82499412 | 51.561488 | 2225.429 | 10000.0 | 10000 | 160963 | 0 |
| cleaned_removed | 82478483 | 51.548408 | 2375.687 | 7007.0 | 29999 | 140034 | 0 |

## Current Interpretation

The winsorized file suppresses the broad high-amplitude outlier population while preserving the original timebase. The cleaned-removed file only removes rows above the configured near-clip threshold, so lower-amplitude outliers can remain.

For time-aligned vibration visualization, prefer `derived_data/iis3dwb_acc_winsorized.dat`.
