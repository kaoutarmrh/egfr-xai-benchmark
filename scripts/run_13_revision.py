#!/usr/bin/env python
"""
run_13_revision.py — the AICSET'26 reviewer revisions, in one pass.

Addresses, in order of how much they can change the paper:

  R5  Paired broad-vs-narrow enrichment on ONE matched compound set.
      The reviewer is right that 527-vs-236 is not a valid comparison.
      THIS CAN CHANGE YOUR HEADLINE SELECTIVITY NUMBERS.
  R3  N1-only vs N1+N3 narrow target. The literature gives N1 the direct
      Met793 hinge H-bond; N3 is water-mediated, so the strict "direct
      interaction" target is N1 alone.
  R6  Bootstrap confidence intervals and sample sizes for every row.
  R4  ECFP bit-collision rate, and atom-mapping validation on ALL curated
      molecules rather than a 1,500 subsample.
  R7  Enrichment stratified by chemotype and by scaffold frequency, to test
      whether the result is driven by common chemotypes.

Nothing is retrained. Everything is recomputed from the saved attributions.

    python run_13_revision.py --data egfr_final.csv --outdir results_revision

Expects (skips gracefully whichever are missing):
    results_xai/attributions.npz
    results_xai_rf/rf_attributions.npz
    results_tf_occ/tf_occlusion_attributions.npz
"""

import os, sys, json, argparse, itertools
import numpy as np
import pandas as pd
from scipy import stats

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

p = argparse.ArgumentParser()
p.add_argument('--data', default='egfr_final.csv')
p.add_argument('--outdir', default='results_revision')
p.add_argument('--xai-dir', default='results_xai')
p.add_argument('--rf-dir', default='results_xai_rf')
p.add_argument('--tfocc-dir', default='results_tf_occ')
p.add_argument('--boot', type=int, default=2000, help='bootstrap resamples')
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()

os.makedirs(args.outdir, exist_ok=True)
LOG = open(os.path.join(args.outdir, 'revision_log.txt'), 'w')
def say(m=''):
    print(m, flush=True); LOG.write(str(m) + '\n'); LOG.flush()

rng = np.random.RandomState(args.seed)

say('=' * 72)
say('AICSET revision analysis')
say('=' * 72)


# ======================================================================
# load attributions
# ======================================================================
SOURCES = [
    (os.path.join(args.xai_dir, 'attributions.npz'), {
        'AttentiveFP_(occlusion)':      'AttentiveFP / node occlusion',
        'AttentiveFP_(grad_x_input)':   'AttentiveFP / grad x input',
        'Transformer_(IG)':             'MoLFormer / integrated grad.',
        'Random_control':               'Control: random atoms',
        'Random_via_bit-map_(control)': 'Control: random via bit-map',
    }),
    (os.path.join(args.rf_dir, 'rf_attributions.npz'), {
        'Random_Forest_occlusion':      'Random forest / bit occlusion',
    }),
    (os.path.join(args.tfocc_dir, 'tf_occlusion_attributions.npz'), {
        'Transformer_occlusion':        'MoLFormer / token occlusion',
    }),
]

ATTR, REF_SMILES = {}, None
for path, mapping in SOURCES:
    if not os.path.exists(path):
        say(f'[skip] not found: {path}')
        continue
    z = np.load(path, allow_pickle=True)
    smi = list(z['smiles'])
    if REF_SMILES is None:
        REF_SMILES = smi
    elif smi != REF_SMILES:
        say(f'[warn] {path} covers different molecules; realigning by SMILES')
    idx = {s: i for i, s in enumerate(smi)}
    order = [idx[s] for s in REF_SMILES if s in idx]
    if len(order) != len(REF_SMILES):
        say(f'[warn] {path} is missing {len(REF_SMILES)-len(order)} molecules; skipped')
        continue
    for key, label in mapping.items():
        if key in z.files:
            arr = list(z[key])
            ATTR[label] = [np.abs(np.asarray(arr[i], dtype=float)) for i in order]

if not ATTR:
    sys.exit('No attribution files found. Check --xai-dir / --rf-dir / --tfocc-dir.')

say(f'\nloaded {len(ATTR)} attribution sets over {len(REF_SMILES)} molecules:')
for k in ATTR:
    say(f'   {k}')

df = pd.read_csv(args.data)
df = df[df.smiles.isin(REF_SMILES)].set_index('smiles').loc[REF_SMILES].reset_index()
MOLS = [Chem.MolFromSmiles(s) for s in REF_SMILES]
NAT = np.array([m.GetNumAtoms() for m in MOLS])
BROAD = [set(json.loads(c)) for c in df.core_atoms.values]


# ======================================================================
# R3 — narrow targets: N1 only (direct) vs N1+N3 (current paper)
# ======================================================================
say('\n' + '=' * 72)
say('R3  narrow pharmacophore: N1 (direct Met793 H-bond) vs N1+N3')
say('=' * 72)
say('Literature: the quinazoline N1 accepts the direct hinge hydrogen bond')
say('from the Met793 backbone amide; N3 makes a WATER-MEDIATED contact with')
say('threonine. A strict "direct interaction" target is therefore N1 alone.')

QUINAZOLINE = Chem.MolFromSmarts('n1cnc2ccccc2c1')
WARHEAD     = Chem.MolFromSmarts('[CX3]=[CX3][CX3](=O)[NX3]')


def quinazoline_nitrogens(mol):
    """Return (n1_atoms, n3_atoms) for every quinazoline in the molecule.

    N1 is the ring nitrogen bonded to the ring-fusion carbon (C8a); N3 is the
    other ring nitrogen, adjacent to C4. Determined from ring membership, so it
    works for any quinazoline, not only 4-anilino ones.
    """
    ri = mol.GetRingInfo()
    n1, n3 = set(), set()
    for match in mol.GetSubstructMatches(QUINAZOLINE):
        ns = [a for a in match if mol.GetAtomWithIdx(a).GetSymbol() == 'N']
        for a in ns:
            fused = any(ri.NumAtomRings(nb.GetIdx()) >= 2
                        for nb in mol.GetAtomWithIdx(a).GetNeighbors())
            (n1 if fused else n3).add(a)
    return n1, n3


def warhead_carbons(mol):
    return {m[i] for m in mol.GetSubstructMatches(WARHEAD) for i in (0, 1)}


# sanity check on reference drugs before applying to the dataset
REF = {'gefitinib':'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
       'erlotinib':'C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1',
       'lapatinib':'CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1',
       'afatinib':'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1',
       'osimertinib':'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-n2cnc3ccccc32)n1'}
say('\nreference-drug check (expect exactly one N1 and one N3 per quinazoline):')
ok = True
for nm, smi in REF.items():
    m = Chem.MolFromSmiles(smi)
    a, b = quinazoline_nitrogens(m)
    w = warhead_carbons(m)
    say(f'   {nm:12s} N1={len(a)} N3={len(b)} warheadC={len(w)}')
    if nm in ('gefitinib', 'erlotinib', 'lapatinib', 'afatinib') and (len(a) != 1 or len(b) != 1):
        ok = False
if not ok:
    say('   [WARNING] unexpected counts on a reference drug — inspect before using')
else:
    say('   OK')

NARROW_N1   = []   # strict: N1 + warhead carbons
NARROW_BOTH = []   # as in the submitted paper: N1 + N3 + warhead carbons
for m in MOLS:
    n1, n3 = quinazoline_nitrogens(m)
    w = warhead_carbons(m)
    NARROW_N1.append(sorted(n1 | w))
    NARROW_BOTH.append(sorted(n1 | n3 | w))

say(f'\nmolecules with a strict (N1) target : {sum(1 for s in NARROW_N1 if s)}')
say(f'molecules with the N1+N3 target     : {sum(1 for s in NARROW_BOTH if s)}')
say(f'mean target size, N1   : {np.mean([len(s)/n for s,n in zip(NARROW_N1,NAT) if s]):.1%} of atoms')
say(f'mean target size, N1+N3: {np.mean([len(s)/n for s,n in zip(NARROW_BOTH,NAT) if s]):.1%} of atoms')
say(f'mean target size, broad: {np.mean([len(s)/n for s,n in zip(BROAD,NAT) if s]):.1%} of atoms')


# ======================================================================
# metric
# ======================================================================
def enrichment(a, target, n):
    a = np.asarray(a, dtype=float)
    tot = a.sum()
    t = [i for i in target if i < len(a)]
    if tot <= 0 or not t or n == 0:
        return np.nan
    return (a[t].sum() / tot) / (len(t) / n)


def boot_ci(vals, nboot, rs):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 5:
        return (np.nan, np.nan)
    idx = rs.randint(0, len(vals), size=(nboot, len(vals)))
    means = vals[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


# ======================================================================
# R5 + R6 — paired comparison on ONE matched set, with CIs
# ======================================================================
say('\n' + '=' * 72)
say('R5  paired broad-vs-narrow enrichment on a MATCHED compound set')
say('R6  bootstrap 95% CIs and sample sizes')
say('=' * 72)

for NARROW, tag in [(NARROW_BOTH, 'N1+N3 (as submitted)'), (NARROW_N1, 'N1 only (strict)')]:
    matched = [i for i in range(len(MOLS)) if NARROW[i] and BROAD[i]]
    say(f'\n--- narrow target: {tag} ---')
    say(f'matched compounds (carry BOTH a broad and a narrow motif): n = {len(matched)}')
    if len(matched) < 20:
        say('   too few compounds; skipped'); continue

    rows = []
    for label, alist in ATTR.items():
        eb, en = [], []
        for i in matched:
            eb.append(enrichment(alist[i], BROAD[i],  NAT[i]))
            en.append(enrichment(alist[i], NARROW[i], NAT[i]))
        eb, en = np.array(eb), np.array(en)
        keep = ~(np.isnan(eb) | np.isnan(en))
        eb, en = eb[keep], en[keep]
        if len(eb) < 20:
            continue

        # paired test: is narrow enrichment higher than broad, same molecules?
        w_stat, w_p = stats.wilcoxon(en, eb)
        # vs chance, on the same matched set
        _, p_b = stats.wilcoxon(eb - 1.0)
        _, p_n = stats.wilcoxon(en - 1.0)
        lo_b, hi_b = boot_ci(eb, args.boot, rng)
        lo_n, hi_n = boot_ci(en, args.boot, rng)
        ratio = en / np.where(eb == 0, np.nan, eb)
        lo_r, hi_r = boot_ci(ratio, args.boot, rng)

        rows.append({
            'model_method': label, 'n': int(len(eb)),
            'broad': round(float(eb.mean()), 3),
            'broad_CI': f'[{lo_b:.3f}, {hi_b:.3f}]',
            'broad_p_vs_chance': f'{p_b:.2g}',
            'narrow': round(float(en.mean()), 3),
            'narrow_CI': f'[{lo_n:.3f}, {hi_n:.3f}]',
            'narrow_p_vs_chance': f'{p_n:.2g}',
            'selectivity_paired': round(float(np.nanmean(ratio)), 3),
            'selectivity_CI': f'[{lo_r:.3f}, {hi_r:.3f}]',
            'paired_p_narrow_vs_broad': f'{w_p:.2g}',
        })

    out = pd.DataFrame(rows)
    fn = 'paired_' + ('n1n3' if 'N1+N3' in tag else 'n1only') + '.csv'
    out.to_csv(os.path.join(args.outdir, fn), index=False)
    say(out[['model_method', 'n', 'broad', 'narrow', 'selectivity_paired',
             'paired_p_narrow_vs_broad']].to_string(index=False))
    say(f'   -> {fn} (full table with CIs)')

say('\nCOMPARE these selectivity values against the submitted paper')
say('(2.99 / 3.12 AttentiveFP, 2.10 MoLFormer, 1.39 random forest).')
say('If they differ materially, Table 1 and Section 3.2 must be rewritten.')


# ======================================================================
# R4 — ECFP bit collisions and atom-mapping validation on ALL molecules
# ======================================================================
say('\n' + '=' * 72)
say('R4  bit collisions and atom-mapping validation (full dataset)')
say('=' * 72)

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
full = pd.read_csv(args.data)
bit_envs = {}
n_env_per_bit = []

for smi in full.smiles:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
    gen.GetFingerprint(m, additionalOutput=ao)
    for bit, occ in ao.GetBitInfoMap().items():
        n_env_per_bit.append(len(occ))
        for c, r in occ:
            if r == 0:
                frag = m.GetAtomWithIdx(c).GetSymbol()
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(m, r, c)
                amap = {}
                sub = Chem.PathToSubmol(m, env, atomMap=amap)
                frag = Chem.MolToSmiles(sub, canonical=True) if sub.GetNumAtoms() else ''
            bit_envs.setdefault(bit, set()).add(frag)

collided = {b: v for b, v in bit_envs.items() if len(v) > 1}
say(f'distinct bits used across the dataset : {len(bit_envs)} of 2048')
say(f'bits mapping to >1 distinct substructure (collisions): {len(collided)} '
    f'({100*len(collided)/max(1,len(bit_envs)):.1f}% of used bits)')
say(f'mean distinct substructures per collided bit: '
    f'{np.mean([len(v) for v in collided.values()]):.2f}' if collided else '')
say(f'mean environments per set bit within a molecule: {np.mean(n_env_per_bit):.2f}')
say('')
say('Two points for the rebuttal:')
say(' (1) Within a molecule, a bit\'s attribution is divided equally among ALL')
say('     atom environments that set it, and among the atoms of each')
say('     environment, so colliding environments share the mass rather than')
say('     either receiving it twice. Total attribution is conserved.')
say(' (2) Attribution is computed by PER-MOLECULE bit occlusion, not from a')
say('     global per-bit importance. Collisions across the dataset therefore')
say('     do not transfer importance between unrelated substructures: only the')
say('     environments actually present in that molecule receive mass.')

pd.DataFrame({'bit': list(bit_envs), 'n_distinct_substructures':
              [len(v) for v in bit_envs.values()]}).to_csv(
    os.path.join(args.outdir, 'bit_collisions.csv'), index=False)

# token -> atom mapping validation on every molecule
import re
ATOM_RE = re.compile(r"\[[^\]]+\]|Br|Cl|Si|Se|se|as|te|[BCNOPSFIbcnops]")
fails, checked = [], 0
for smi in full.smiles:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    checked += 1
    try:
        canon = Chem.MolToSmiles(m)
        order = list(map(int, m.GetProp('_smilesAtomOutputOrder')[1:-1]
                         .split(',')[:m.GetNumAtoms()]))
        spans = [(x.start(), x.end()) for x in ATOM_RE.finditer(canon)]
        if len(spans) != m.GetNumAtoms() or sorted(order) != list(range(m.GetNumAtoms())):
            fails.append(smi)
    except Exception:
        fails.append(smi)
say(f'\natom-token alignment checked on {checked} molecules (not 1,500): '
    f'{len(fails)} failures')
if fails:
    pd.Series(fails).to_csv(os.path.join(args.outdir, 'mapping_failures.csv'), index=False)


# ======================================================================
# R7 — is enrichment driven by common chemotypes?
# ======================================================================
say('\n' + '=' * 72)
say('R7  enrichment stratified by chemotype and by training-set familiarity')
say('=' * 72)
say('NOTE: a scaffold split sends the largest scaffold families to training, so')
say('every test-fold scaffold is a singleton and "common vs rare scaffold"')
say('cannot be computed within the test fold. We therefore measure how')
say('FAMILIAR each test molecule is to the model, as nearest-neighbour Tanimoto')
say('similarity to the training set, which is what "frequently represented')
say('chemotype" means operationally here.')

from rdkit import DataStructs
_g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
train_fps = []
for smi in full[full.split == 'train'].smiles:
    mm = Chem.MolFromSmiles(smi)
    if mm is not None:
        train_fps.append(_g.GetFingerprint(mm))

maxsim = []
for m in MOLS:
    if m is None or not train_fps:
        maxsim.append(np.nan); continue
    maxsim.append(max(DataStructs.BulkTanimotoSimilarity(_g.GetFingerprint(m), train_fps)))
maxsim = np.array(maxsim)
say(f'\nnearest-neighbour Tanimoto to training set: median {np.nanmedian(maxsim):.3f}  '
    f'range {np.nanmin(maxsim):.3f}-{np.nanmax(maxsim):.3f}')

AMINOPYR = Chem.MolFromSmarts('[NX3;H1]-c1ncccn1')
has_quin = np.array([len(quinazoline_nitrogens(m)[0]) > 0 for m in MOLS])
has_war  = np.array([len(warhead_carbons(m)) > 0 for m in MOLS])
has_apy  = np.array([m.HasSubstructMatch(AMINOPYR) for m in MOLS])

t1, t2 = np.nanpercentile(maxsim, [33.3, 66.7])
strata = {
    'quinazoline':              has_quin,
    'acrylamide warhead':       has_war,
    'aminopyrimidine':          has_apy,
    f'familiarity low (<{t1:.2f})':        maxsim < t1,
    f'familiarity mid ({t1:.2f}-{t2:.2f})': (maxsim >= t1) & (maxsim < t2),
    f'familiarity high (>{t2:.2f})':       maxsim >= t2,
}

srows = []
for label, alist in ATTR.items():
    r = {'model_method': label}
    for sname, mask in strata.items():
        idx = [i for i in range(len(MOLS)) if mask[i] and NARROW_BOTH[i]]
        if len(idx) < 20:
            r[sname] = f'n<20'; continue
        v = np.array([enrichment(alist[i], NARROW_BOTH[i], NAT[i]) for i in idx])
        v = v[~np.isnan(v)]
        r[sname] = f'{v.mean():.2f} (n={len(v)})' if len(v) else 'na'
    srows.append(r)

st = pd.DataFrame(srows)
st.to_csv(os.path.join(args.outdir, 'stratified_enrichment.csv'), index=False)
say('')
say(st.to_string(index=False))
say('')
say('Read across each row. If enrichment holds in the LOW-familiarity stratum')
say('and across chemotypes, the result is not an artefact of frequently')
say('represented chemistry -- which is what the reviewer asks you to show.')

say('\n' + '=' * 72)
say(f'outputs written to {os.path.abspath(args.outdir)}')
for f in sorted(os.listdir(args.outdir)):
    say(f'   {f}')
say('=' * 72)
LOG.close()
