# Structure-grounded validation of attribution methods in EGFR bioactivity prediction

Code and data definitions for the AICSET'26 paper *Explainable AI for Anticancer
Drug Discovery: Structure-Grounded Validation of Attribution Methods in EGFR
Bioactivity Prediction* (Paper 222).

The paper asks a narrow question: when an atom-level attribution map says a
model "focused on" part of a molecule, does that agree with the binding
chemistry crystallography already established for EGFR? To answer it without
circularity, the structural reference is fixed before any model is trained, two
negative controls calibrate the metric, and the same attribution is scored
against two references that differ only in precision.

## What is here

```
scripts/     the pipeline, in run order
results/     the tables behind every number in the paper
smarts.md    the complete pharmacophore definitions
```

## Reproducing the paper

The pipeline runs in order. Steps 01-07 build the dataset and train; 08-12
produce attributions; 13-16 are the analyses added during revision.

| script | what it does | needs |
|---|---|---|
| `01_build_dataset.py` | ChEMBL download, filtering, wild-type/mutant separation, standardisation, scaffold split | CPU |
| `02_annotate_pharmacophore.py` | SMARTS annotation, broad and narrow targets | CPU |
| `03_finalize_dataset.py` | leak-free split, writes `egfr_final.csv` | CPU |
| `run_06_gnn.py` | AttentiveFP | GPU |
| `run_07_transformers.py` | ChemBERTa variants, MoLFormer-XL | GPU |
| `run_08_xai.py` | attributions: random forest, AttentiveFP (gradient and occlusion), MoLFormer (integrated gradients), both controls | GPU |
| `run_08b_rf_fix.py` | random forest attribution by bit occlusion, replacing TreeSHAP | CPU |
| `run_11_transformer_occlusion.py` | MoLFormer token occlusion | GPU |
| `run_12_seeds.py` | five seeds, 42-46 | GPU |
| `run_13_revision.py` | paired enrichment, bootstrap CIs, bit collisions, stratification | CPU |
| `run_14_figure_and_stats.py` | Figure 1, assay provenance statistics | CPU |
| `run_15_null_and_inventory.py` | molecule-matched null comparisons | CPU |
| `run_16_ig_sensitivity.py` | integrated-gradients baseline and step sensitivity, MoLFormer faithfulness | GPU |
| `run_17_tokenizer_alignment.py` | tokenizer-offset to heavy-atom alignment check | CPU |

Every analysis script takes `--data egfr_final.csv` and writes to `--outdir`.

## A note on TreeSHAP

The random forest attribution in the paper is computed by **bit occlusion**, not
TreeSHAP. TreeSHAP was tried first and discarded: `shap.TreeExplainer` returned
per-molecule attribution totals of order 10^11 where the model's own output
scale is a few pIC50 units, which cannot be correct, since SHAP values must sum
to the prediction minus the base value. The additivity check that would have
caught this had been disabled. `run_08b_rf_fix.py --verify-shap` reproduces the
failure; the occlusion replacement is in the same file.

The lesson generalises beyond this package: an attribution method can fail
silently while still producing a visually plausible saliency map. Attribution
magnitudes should be checked against the model's output scale before they are
interpreted.

## Data split

The split is Bemis-Murcko scaffold-based, computed once and reused unchanged
across every experiment and every seed. Scaffold families are assigned greedily
in descending size to 80/10/10 folds, so the largest families go to training and
the test fold is enriched in rarer chemotypes. Partition integrity is asserted
programmatically: each scaffold occurs in exactly one fold.

Per-compound fold identifiers are the `split` column of `egfr_final.csv`
(6,385 train / 798 validation / 799 test).

## Attribution, in one place

All attributions are projected onto heavy atoms so that representations can be
compared.

- **Fingerprints.** Each bit is mapped to its contributing circular atom
  environments through the RDKit bit-information map, and its attribution is
  divided among those environments and among the atoms of each, which conserves
  attribution mass. Across the dataset 2,039 of the 2,047 bits used correspond
  to more than one substructure (8.58 on average), but attribution is computed
  per molecule, where a set bit corresponds to 1.56 environments on average.
- **Chemical language models.** Tokens are associated with the atoms they span
  using the tokenizer's offset mappings; multi-character tokens divide their
  attribution among those atoms. `run_17_tokenizer_alignment.py` verifies the
  mapping on all 7,982 curated molecules: the atom-span count matches the
  heavy-atom count in every molecule, no molecule exceeds the 256-token limit,
  and **every heavy atom is covered by exactly one token**, so no atom can
  receive zero attribution or be counted twice.

  That last claim is checkable from `results/tokenizer_alignment.csv` without
  re-running anything. Every row has `atoms_without_token == 0`, so each atom
  has at least one token; every row has `mean_tokens_per_atom == 1.000` and the
  largest molecule has 103 atoms, so a single extra token anywhere would have
  pushed that mean to at least 1.010. The two together force exactly one.

  What the published CSV does *not* settle is whether any single token spans
  more than one atom — the averages are consistent with it. `run_17` now
  records the per-molecule maxima that would decide it. No claim about it is
  made in the paper.
- **Graph models.** Atoms are indexed directly.
- **Signed attributions.** Gradient methods return signed values. Since the
  enrichment metric asks where a model places weight and is defined for
  non-negative attribution, signed values are mapped to magnitude before
  projection and before enrichment is computed. Perturbation attributions are
  non-negative by construction.

## Seeds and determinism

Three models were trained under five seeds, 42-46, and are reported as mean +/-
standard deviation: **random forest, XGBoost and AttentiveFP**. The chemical
language models (ChemBERTa x3, MoLFormer-XL) are reported from a **single
selected run**, not a seed average, and carry no error bar. The paper says so;
it is repeated here because the distinction is easy to lose when reading the
tables.

The scaffold partition is held fixed across seeds because it defines the
benchmark rather than a source of variance. Controls use seed 0; bootstrap
resampling uses seed 42 with 2,000 resamples.

## Software

Versions for the re-analysis environment (Linux, CUDA 13), which ran `run_13`
through `run_16`:

| package | version |
|---|---|
| Python | 3.13.16 |
| torch | 2.14.1+cu130 |
| transformers | 5.22.1 |
| scikit-learn | 1.6.1 |
| xgboost | 3.4.1 |
| numpy | 2.4.6 |
| scipy | 1.16.3 |
| pandas | 2.2.3 |
| rdkit | installed from PyPI at run time |

Scripts `01` through `run_12`, which produced the model checkpoints and the
archived attribution files, were run on Windows in an environment whose package
versions were not recorded.

## Results

`results/` holds every number behind the paper. `results/PROVENANCE.md` records
which files are direct script output and which were rebuilt from run logs after
a notebook runtime reset.

| file | contents |
|---|---|
| `paired_n1n3.csv` | Table 1: broad, narrow and selectivity enrichment with bootstrap 95% CIs, on the matched 236 compounds |
| `paired_n1only.csv` | the same, against the strict N1-plus-warhead target |
| `matched_null.csv` | every model against each molecule-matched null control, paired |
| `stratified_enrichment.csv` | enrichment by chemotype and by nearest-neighbour similarity to training |
| `bit_collisions.csv` | per-bit collision counts across the dataset |
| `ig_sensitivity.csv` | integrated gradients across baselines and step counts, with completeness residuals |
| `egfr_final.csv` | the curated dataset: 7,982 structures with pIC50, pharmacophore annotation and fold assignment |
| `tf_faithfulness.csv` | MoLFormer ablation, token occlusion and integrated gradients |
| `tokenizer_alignment.csv` | per-molecule tokenizer-offset coverage check |
| `dataset_stats.txt` | assay provenance and replicate agreement, with two caveats on how the numbers were computed |
| `fig1_attribution_maps.png` | Figure 1 at full resolution — the panels are small at print size, so read them here |
| `revision_log.txt`, `matched_null_log.txt`, `ig_sensitivity_log.txt`, `tokenizer_alignment_log.txt` | full console output of each analysis |

Two columns need a word of warning. In `ig_sensitivity.csv`,
`selectivity_ratio_of_means` is `mean(narrow) / mean(broad)` — **not** the
quantity in Table 1 of the paper, which is the mean of the within-molecule
ratios. The two are different and should not be compared; `run_16` now emits
both under distinct names, so a re-run gives a column that is comparable.
Nothing in the paper quotes a selectivity value from this file.

Two numbers in `dataset_stats.txt` deserve their caveats read before use: the
"cell-based" assay-format count is dominated by false positives (its pattern
matches "growth" in *epidermal growth factor receptor*) and is not used in the
paper, and the replicate statistics describe the exact-relation records before
variant separation and standardisation rather than the final curated set.

## Citation

To be added once the proceedings are published.

## Licence

MIT — see [LICENSE](LICENSE).
