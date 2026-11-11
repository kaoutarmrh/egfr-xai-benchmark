#!/usr/bin/env python
"""
run_15_null_and_inventory.py

(E) Molecule-matched null comparison. The reviewer asked for statistical
    validation AGAINST the null controls, not merely alongside them. The two
    controls are computed per molecule, so the right test is paired: for each
    compound, the model's narrow enrichment against the SAME compound's
    control enrichment, Wilcoxon signed-rank, with a paired effect size.

    This needs no model and no GPU - it re-reads the saved attributions.

(inventory) Lists what model checkpoints and scripts are present, so the
    integrated-gradients sensitivity experiment can be written correctly
    rather than guessed at.

    python run_15_null_and_inventory.py --data egfr_final.csv

Writes to --outdir (default results_revision):
    matched_null.csv
    inventory.txt
"""

import os, sys, json, glob, argparse
import numpy as np
import pandas as pd
from scipy import stats

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

p = argparse.ArgumentParser()
p.add_argument('--data',      default='egfr_final.csv')
p.add_argument('--outdir',    default='results_revision')
p.add_argument('--xai-dir',   default='results_xai')
p.add_argument('--rf-dir',    default='results_xai_rf')
p.add_argument('--tfocc-dir', default='results_tf_occ')
p.add_argument('--boot', type=int, default=2000)
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()

os.makedirs(args.outdir, exist_ok=True)
LOG = open(os.path.join(args.outdir, 'inventory.txt'), 'w')
def say(m=''):
    print(m, flush=True); LOG.write(str(m) + '\n'); LOG.flush()

rng = np.random.RandomState(args.seed)

# ----------------------------------------------------------------------
# load attributions (identical layout to run_13)
# ----------------------------------------------------------------------
SOURCES = [
    (os.path.join(args.xai_dir, 'attributions.npz'), {
        'AttentiveFP_(occlusion)':      'AttentiveFP / node occlusion',
        'AttentiveFP_(grad_x_input)':   'AttentiveFP / grad x input',
        'Transformer_(IG)':             'MoLFormer / integrated grad.',
        'Random_control':               'Control: random atoms',
        'Random_via_bit-map_(control)': 'Control: random via bit-map',
    }),
    (os.path.join(args.rf_dir, 'rf_attributions.npz'), {
        'Random_Forest_occlusion':      'Random forest / bit occlusion'}),
    (os.path.join(args.tfocc_dir, 'tf_occlusion_attributions.npz'), {
        'Transformer_occlusion':        'MoLFormer / token occlusion'}),
]

ATTR, REF = {}, None
for path, mapping in SOURCES:
    if not os.path.exists(path):
        say(f'[skip] {path}'); continue
    z = np.load(path, allow_pickle=True)
    smi = list(z['smiles'])
    if REF is None: REF = smi
    idx = {s: i for i, s in enumerate(smi)}
    order = [idx[s] for s in REF if s in idx]
    if len(order) != len(REF):
        say(f'[warn] {path} incomplete; skipped'); continue
    for k, label in mapping.items():
        if k in z.files:
            arr = list(z[k])
            ATTR[label] = [np.abs(np.asarray(arr[i], dtype=float)) for i in order]

if not ATTR:
    sys.exit('no attribution files found')

df = pd.read_csv(args.data)
df = df[df.smiles.isin(REF)].set_index('smiles').loc[REF].reset_index()
MOLS = [Chem.MolFromSmiles(s) for s in REF]
NAT  = np.array([m.GetNumAtoms() for m in MOLS])

QUINAZOLINE = Chem.MolFromSmarts('n1cnc2ccccc2c1')
WARHEAD     = Chem.MolFromSmarts('[CX3]=[CX3][CX3](=O)[NX3]')

def quinazoline_nitrogens(mol):
    ri = mol.GetRingInfo(); n1, n3 = set(), set()
    for match in mol.GetSubstructMatches(QUINAZOLINE):
        for a in [x for x in match if mol.GetAtomWithIdx(x).GetSymbol() == 'N']:
            fused = any(ri.NumAtomRings(nb.GetIdx()) >= 2
                        for nb in mol.GetAtomWithIdx(a).GetNeighbors())
            (n1 if fused else n3).add(a)
    return n1, n3

def warhead_carbons(mol):
    return {m[i] for m in mol.GetSubstructMatches(WARHEAD) for i in (0, 1)}

NARROW = []
for m in MOLS:
    n1, n3 = quinazoline_nitrogens(m)
    NARROW.append(sorted(n1 | n3 | warhead_carbons(m)))
BROAD = [set(json.loads(c)) for c in df.core_atoms.values]

def enrichment(a, target, n):
    a = np.asarray(a, dtype=float); tot = a.sum()
    t = [i for i in target if i < len(a)]
    if tot <= 0 or not t or n == 0: return np.nan
    return (a[t].sum() / tot) / (len(t) / n)

# ----------------------------------------------------------------------
# (E) paired comparison against each molecule-matched null
# ----------------------------------------------------------------------
say('=' * 74)
say('E  Model enrichment vs molecule-matched null controls (paired)')
say('=' * 74)
say('For every compound the model and the control are scored on the SAME')
say('molecule and the SAME target, so the comparison is paired within')
say('compound rather than against a nominal 1.0.')
say('')

CONTROLS = ['Control: random atoms', 'Control: random via bit-map']
MODELS   = [k for k in ATTR if not k.startswith('Control')]
matched  = [i for i in range(len(MOLS)) if NARROW[i] and BROAD[i]]
say(f'matched compounds: n = {len(matched)}\n')

def boot_ci(x, f, B, rng):
    n = len(x); idx = rng.randint(0, n, (B, n))
    v = np.array([f(x[i]) for i in idx])
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)

rows = []
for model in MODELS:
    em = np.array([enrichment(ATTR[model][i], NARROW[i], NAT[i]) for i in matched])
    for ctrl in CONTROLS:
        if ctrl not in ATTR: continue
        ec = np.array([enrichment(ATTR[ctrl][i], NARROW[i], NAT[i]) for i in matched])
        keep = ~(np.isnan(em) | np.isnan(ec))
        a, b = em[keep], ec[keep]
        if len(a) < 20: continue
        try:
            W, pw = stats.wilcoxon(a, b)
        except ValueError:
            W, pw = np.nan, np.nan
        d = a - b
        lo, hi = boot_ci(d, np.mean, args.boot, rng)
        rows.append(dict(model=model, control=ctrl, n=int(len(a)),
                         model_enrichment=round(float(np.mean(a)), 3),
                         control_enrichment=round(float(np.mean(b)), 3),
                         mean_difference=round(float(np.mean(d)), 3),
                         difference_CI=f'[{lo:.3f}, {hi:.3f}]',
                         frac_model_above_control=round(float((d > 0).mean()), 3),
                         paired_p=float(f'{pw:.2g}') if np.isfinite(pw) else np.nan))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(args.outdir, 'matched_null.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
say(out.to_string(index=False))
say('')
say('Read: mean_difference is model minus control on the same molecules.')
say('A difference CI excluding zero means the model beats its own null.')

# ----------------------------------------------------------------------
# inventory: what is available for the IG sensitivity experiment
# ----------------------------------------------------------------------
say('\n' + '=' * 74)
say('INVENTORY  (for the integrated-gradients sensitivity experiment)')
say('=' * 74)

say('\npython scripts present here:')
for f in sorted(glob.glob('*.py')):
    say(f'   {f:40s} {os.path.getsize(f)/1024:6.1f} KB')

say('\nmodel checkpoints and directories (searching ./ and ../):')
pats = ['*.pt', '*.bin', '*.safetensors', '*/config.json', '*/*/config.json',
        '../*.pt', '../*/config.json']
seen = set()
for pat in pats:
    for f in glob.glob(pat):
        r = os.path.realpath(f)
        if r in seen: continue
        seen.add(r)
        say(f'   {f:55s} {os.path.getsize(f)/1024/1024:7.1f} MB')
if not seen:
    say('   (none found)')

say('\nresults directories:')
for d in sorted(glob.glob('results*')):
    if os.path.isdir(d):
        say(f'   {d}/  -> {", ".join(sorted(os.listdir(d))[:8])}')

say('\nIf a MoLFormer checkpoint appears above, the IG sensitivity run')
say('(different baselines and step counts) can be scripted against it.')
LOG.close()
print('\nwrote', os.path.join(args.outdir, 'matched_null.csv'))
print('wrote', os.path.join(args.outdir, 'inventory.txt'))
