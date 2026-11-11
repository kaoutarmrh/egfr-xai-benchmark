#!/usr/bin/env python
"""
run_08b_rf_fix.py — recompute the Random Forest attribution without SHAP.

WHY THIS EXISTS
check_rf.py showed TreeSHAP returning per-molecule attribution totals around
8e11, where every other model sits near 1-25. SHAP values must sum to roughly
(prediction - base value), i.e. a couple of pIC50 units. They did not. This is a
shap / scikit-learn version incompatibility - TreeExplainer reads sklearn's
internal tree layout, and both packages are very new here. run_08 disabled
check_additivity, which is exactly the guard that would have caught it.

THE REPLACEMENT
Occlusion on fingerprint bits: switch off one bit the molecule actually has,
re-predict, and measure how far the prediction moved. That is the same procedure
already used for the GNN, so the two models become directly comparable, and it
needs no shap at all.

    python run_08b_rf_fix.py --data egfr_final.csv --outdir results_xai_rf

Outputs
    rf_attributions.npz    per-atom attribution, occlusion-based
    rf_faithfulness.csv    broad and narrow enrichment, with controls
    run_log.txt
"""

import os, json
import numpy as np
import pandas as pd
from scipy import stats
from common import Ctx, parse_args

args = parse_args(lambda p: (
    p.add_argument('--trees', type=int, default=500),
    p.add_argument('--verify-shap', action='store_true',
                   help='confirm the shap bug with check_additivity=True'),
))
ctx = Ctx(args)
train, val, test = ctx.splits()

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
RDLogger.DisableLog('rdApp.*')

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
MOLS = [Chem.MolFromSmiles(s) for s in test.smiles.values]
NAT = [m.GetNumAtoms() for m in MOLS]
BROAD = [set(json.loads(c)) for c in test.core_atoms.values]


def fp_matrix(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            X[i] = np.array(gen.GetFingerprint(m))
    return X


ctx.log('training Random Forest (seed 42)')
Xtr = fp_matrix(train.smiles.values)
Xte = fp_matrix(test.smiles.values)
rf = RandomForestRegressor(n_estimators=args.trees, max_features='sqrt',
                           n_jobs=-1, random_state=42)
rf.fit(Xtr, train.pIC50.values)
ctx.log(f'test R2: {rf.score(Xte, test.pIC50.values):.4f}')


# ----------------------------------------------------------------------
# optional: document the shap failure
# ----------------------------------------------------------------------
if args.verify_shap:
    ctx.log('\n=== verifying the shap bug ===')
    try:
        import shap
        ctx.log(f'shap {shap.__version__}')
        sv = shap.TreeExplainer(rf).shap_values(Xte[:20], check_additivity=True)
        ctx.log(f'  additivity check PASSED; |shap| sum per molecule: '
                f'{np.abs(sv).sum(axis=1)[:5].round(3)}')
        ctx.log('  (if these are ~1-3, shap is fine on your versions after all)')
    except Exception as e:
        ctx.log(f'  additivity check FAILED: {type(e).__name__}')
        ctx.log(f'  {str(e)[:300]}')
        ctx.log('  -> confirms the shap/sklearn incompatibility')


# ----------------------------------------------------------------------
# occlusion on bits
# ----------------------------------------------------------------------
def env_atoms(mol, center, radius):
    if radius == 0:
        return {center}
    atoms = {center}
    for b in Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center):
        bd = mol.GetBondWithIdx(b)
        atoms.add(bd.GetBeginAtomIdx()); atoms.add(bd.GetEndAtomIdx())
    return atoms


ctx.log('\ncomputing bit-occlusion attribution')
ATTR = []
for i, mol in enumerate(MOLS):
    ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
    gen.GetFingerprint(mol, additionalOutput=ao)
    binfo = ao.GetBitInfoMap()
    bits = sorted(binfo)

    # one row for the molecule as-is, then one row per bit switched off
    X = np.repeat(Xte[i:i+1], len(bits)+1, axis=0)
    for j, b in enumerate(bits):
        X[j+1, b] = 0.0
    preds = rf.predict(X)
    base, off = preds[0], preds[1:]

    a = np.zeros(mol.GetNumAtoms())
    for b, p in zip(bits, off):
        v = abs(base - p)                       # how much this bit mattered
        if v == 0:
            continue
        occ = binfo[b]
        for center, radius in occ:
            atoms = env_atoms(mol, center, radius)
            for at in atoms:
                a[at] += v / len(atoms) / len(occ)
    ATTR.append(a)
    if (i+1) % 200 == 0:
        ctx.log(f'  {i+1}/{len(MOLS)}')

allv = np.concatenate(ATTR)
tots = np.array([x.sum() for x in ATTR])
conc = np.array([x.max()/x.sum() if x.sum() > 0 else np.nan for x in ATTR])
ctx.log('\nsanity check on the new attribution')
ctx.log(f'  per-molecule total : median {np.median(tots):.4g}   max {tots.max():.4g}')
ctx.log(f'  per-atom max       : {allv.max():.4g}')
ctx.log(f'  mass on top atom   : median {np.nanmedian(conc):.1%}')
ctx.log('  (totals should now be the same order as the other models, ~1-30)')

np.savez_compressed(ctx.path('rf_attributions.npz'),
                    smiles=test.smiles.values,
                    Random_Forest_occlusion=np.array(ATTR, dtype=object))


# ----------------------------------------------------------------------
# enrichment, broad and narrow, with controls
# ----------------------------------------------------------------------
NARROW_SMARTS = {'hinge_nitrogens': ('n1cnc2ccccc2c1', 'N_only'),
                 'warhead_carbons': ('[CX3]=[CX3][CX3](=O)[NX3]', [0, 1])}
_NC = {k: (Chem.MolFromSmarts(v[0]), v[1]) for k, v in NARROW_SMARTS.items()}


def narrow_atoms(mol):
    out = set()
    for patt, keep in _NC.values():
        for m in mol.GetSubstructMatches(patt):
            if keep == 'N_only':
                out |= {a for a in m if mol.GetAtomWithIdx(a).GetSymbol() == 'N'}
            else:
                out |= {m[i] for i in keep if i < len(m)}
    return sorted(out)


NARROW = [narrow_atoms(m) for m in MOLS]


def enrichment(attr, target, n):
    attr = np.asarray(attr, dtype=float)
    tot = attr.sum()
    target = [t for t in target if t < len(attr)]
    if tot <= 0 or not target or n == 0:
        return None
    return (attr[target].sum()/tot) / (len(target)/n)


rng = np.random.RandomState(0)
SETS = {'Random Forest (bit occlusion)': ATTR,
        'Random control': [np.abs(rng.normal(0, 1, n)) for n in NAT]}

rows = []
for label, alist in SETS.items():
    r = {'model': label}
    for tag, targets in [('broad', BROAD), ('narrow', NARROW)]:
        vals = [e for e in (enrichment(a, t, n)
                            for a, t, n in zip(alist, targets, NAT)) if e]
        if vals:
            r[f'{tag}_enrichment'] = round(float(np.mean(vals)), 4)
            r[f'{tag}_median'] = round(float(np.median(vals)), 4)
            r[f'{tag}_n'] = len(vals)
            r[f'{tag}_frac_above_1'] = round(float(np.mean(np.array(vals) > 1)), 4)
            r[f'{tag}_p'] = (f'{stats.wilcoxon(np.array(vals)-1.0).pvalue:.3g}'
                             if len(vals) > 10 else 'na')
    rows.append(r)
    ctx.log(f"\n{label}")
    ctx.log(f"  broad  enrichment {r.get('broad_enrichment')}  p {r.get('broad_p')}")
    ctx.log(f"  narrow enrichment {r.get('narrow_enrichment')}  p {r.get('narrow_p')}")

out = pd.DataFrame(rows)
out.to_csv(ctx.path('rf_faithfulness.csv'), index=False)
ctx.log('\n' + out.to_string(index=False))
ctx.log('\nUse these Random Forest rows in the paper. The SHAP rows from run_08')
ctx.log('are invalid and must be discarded.')
ctx.log('done')
