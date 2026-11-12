# Provenance of the files in this directory

Written because the paper argues that an artifact should be checked rather than
trusted, and it would be inconsistent not to say where these came from.

## Direct script output

These were written by the scripts and copied here unchanged:

| file | produced by |
|---|---|
| `paired_n1n3.csv` | `run_13_revision.py` |
| `paired_n1only.csv` | `run_13_revision.py` |
| `stratified_enrichment.csv` | `run_13_revision.py` |
| `bit_collisions.csv` | `run_13_revision.py` |
| `revision_log.txt` | `run_13_revision.py` |
| `tokenizer_alignment.csv` | `run_17_tokenizer_alignment.py` |

`tokenizer_alignment_log.txt` is that script's console output, captured verbatim
from the run rather than redirected to a file.

## Reconstructed from run logs

The analyses were executed in a hosted notebook whose runtime was reset before
these files were copied out. The console output had been retained in full, so
the tables below were rebuilt from it rather than by re-running several hours of
GPU computation. The logs they were rebuilt from are included alongside them, so
every value can be checked against the output that produced it.

| file | rebuilt from |
|---|---|
| `matched_null.csv` | `matched_null_log.txt` (`run_15_null_and_inventory.py`) |
| `ig_sensitivity.csv` | `ig_sensitivity_log.txt` (`run_16_ig_sensitivity.py`) |
| `tf_faithfulness.csv` | `ig_sensitivity_log.txt` (`run_16_ig_sensitivity.py`) |
| `dataset_stats.txt` | `run_14_figure_and_stats.py` console output |

One column was renamed after publication. In `ig_sensitivity.csv` the column
originally called `selectivity` is the ratio of the two means
(`mean(narrow) / mean(broad)`), whereas Table 1 of the paper and
`run_13_revision.py` use the mean of the within-molecule ratios. These are
different quantities and should not be compared with each other. The column is
now named `selectivity_ratio_of_means` to say which one it is, and
`run_16_ig_sensitivity.py` has been changed to emit both, so a re-run produces
a column directly comparable with Table 1. The numbers themselves are
unchanged. Nothing in the paper quotes a selectivity value from this file.

**This file therefore predates that script change, and cannot be brought into
line with it without re-running the sweep.** `selectivity_paired` is a mean
over per-molecule ratios; the original run retained only the aggregates, so the
paired value cannot be derived from anything kept here. Adding the column would
mean inventing a number that was never computed. The file stays as the record
of the run that was performed. A reader who wants the paired column runs
`run_16_ig_sensitivity.py` and gets both.

Two consequences of that reconstruction, stated rather than hidden:

- `tf_faithfulness.csv` omits the per-row `n` column, because the console
  summary did not print it. Every other column is as printed.
- `ig_completeness.csv` is not included. It held the median and 90th-percentile
  completeness residuals; only the median was printed to the console, and it is
  already a column of `ig_sensitivity.csv`. Rather than publish a file with a
  column reconstructed and a column missing, it is left out.

Re-running `run_15_null_and_inventory.py` and `run_16_ig_sensitivity.py`
regenerates all of these directly. `run_16` needs a GPU and takes roughly two
and a half hours for the full sweep.
