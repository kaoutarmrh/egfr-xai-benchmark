#!/usr/bin/env python
"""
check_rf.py — is the Random Forest attribution pathological?

Section 3 of run_09 printed single atoms carrying values like 274. SHAP values
for a pIC50 model should sum to roughly +/- 2 across the whole molecule, so a
single atom at 274 is not plausible and would explain the 0.0002 narrow
enrichment as an artefact rather than a chemical statement.

    python check_rf.py
    python check_rf.py --xai-dir results_xai
"""

import sys, os
import numpy as np

xai_dir = 'results_xai'
if '--xai-dir' in sys.argv:
    xai_dir = sys.argv[sys.argv.index('--xai-dir') + 1]

path = os.path.join(xai_dir, 'attributions.npz')
if not os.path.exists(path):
    raise SystemExit(f'not found: {path}')

npz = np.load(path, allow_pickle=True)
keys = [k for k in npz.files if k != 'smiles']
print('models in file:')
for k in keys:
    print('  ', k)
print()

def stats(name):
    a = [np.asarray(x, dtype=float) for x in list(npz[name])]
    allv = np.concatenate(a)
    totals = np.array([x.sum() for x in a])
    maxima = np.array([x.max() if len(x) else 0 for x in a])
    # how much of a molecule's attribution sits on its single biggest atom
    conc = np.array([x.max()/x.sum() if x.sum() > 0 else np.nan for x in a])
    print(f'=== {name} ===')
    print(f'  per-atom   min {allv.min():.6g}   median {np.median(allv):.6g}   '
          f'p99 {np.percentile(allv,99):.6g}   max {allv.max():.6g}')
    print(f'  per-molecule total   median {np.median(totals):.4g}   '
          f'max {totals.max():.4g}')
    print(f'  biggest single atom  median {np.median(maxima):.4g}   '
          f'max {maxima.max():.4g}')
    print(f'  share of mass on the top atom: median {np.nanmedian(conc):.1%}   '
          f'90th pct {np.nanpercentile(conc,90):.1%}')
    n_extreme = int((allv > 10).sum())
    print(f'  atoms with value > 10 : {n_extreme} of {len(allv)} '
          f'({100*n_extreme/len(allv):.2f}%)')
    print()
    return conc

rf_key = [k for k in keys if 'Forest' in k]
conc_rf = stats(rf_key[0]) if rf_key else None

for k in keys:
    if rf_key and k == rf_key[0]:
        continue
    stats(k)

print('--- how to read this ---')
print('If Random Forest shows a per-molecule total orders of magnitude larger')
print('than the other models, or a median "share of mass on the top atom"')
print('above ~40%, the bit->atom accumulation is concentrating everything onto')
print('a few atoms. The 0.0002 narrow enrichment would then be an artefact of')
print('that concentration, not a chemical finding, and the RF row should be')
print('reported with that caveat or recomputed with a normalised distribution.')
print()
print('If its numbers look like the other models, the RF result stands.')
