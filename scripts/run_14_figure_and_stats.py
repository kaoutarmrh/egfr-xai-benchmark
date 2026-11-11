#!/usr/bin/env python
"""
run_14_figure_and_stats.py — the two remaining reviewer items.

R7  One figure: a reference EGFR inhibitor showing (a) the annotated
    pharmacophore atoms and (b) the atom-level attribution map produced by
    each of the three representations, so the reader can see where the
    explanations agree and where they diverge on the SAME compound.

R2  Assay provenance: how many assays, which variants, how far replicate
    measurements disagreed, and how many compounds each filter removed.

Nothing is trained and nothing is recomputed from scratch.

    python run_14_figure_and_stats.py --data egfr_final.csv --raw egfr_raw.csv

Writes to --outdir (default results_revision):
    fig1_attribution_maps.png
    dataset_stats.txt
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog('rdApp.*')

p = argparse.ArgumentParser()
p.add_argument('--data',      default='egfr_final.csv')
p.add_argument('--raw',       default='egfr_raw.csv')
p.add_argument('--outdir',    default='results_revision')
p.add_argument('--xai-dir',   default='results_xai')
p.add_argument('--rf-dir',    default='results_xai_rf')
p.add_argument('--tfocc-dir', default='results_tf_occ')
args = p.parse_args()

os.makedirs(args.outdir, exist_ok=True)
LOG = open(os.path.join(args.outdir, 'dataset_stats.txt'), 'w')
def say(m=''):
    print(m, flush=True); LOG.write(str(m) + '\n'); LOG.flush()


# ======================================================================
# load attributions (same layout as run_13)
# ======================================================================
SOURCES = [
    (os.path.join(args.xai_dir, 'attributions.npz'), {
        'AttentiveFP_(occlusion)':    'AttentiveFP, node occlusion',
        'AttentiveFP_(grad_x_input)': 'AttentiveFP, grad x input',
        'Transformer_(IG)':           'MoLFormer, integrated grad.',
    }),
    (os.path.join(args.rf_dir, 'rf_attributions.npz'), {
        'Random_Forest_occlusion':    'Random forest, bit occlusion',
    }),
    (os.path.join(args.tfocc_dir, 'tf_occlusion_attributions.npz'), {
        'Transformer_occlusion':      'MoLFormer, token occlusion',
    }),
]

ATTR, REF_SMILES = {}, None
for path, mapping in SOURCES:
    if not os.path.exists(path):
        print(f'[skip] not found: {path}'); continue
    z = np.load(path, allow_pickle=True)
    smi = list(z['smiles'])
    if REF_SMILES is None:
        REF_SMILES = smi
    idx = {s: i for i, s in enumerate(smi)}
    order = [idx[s] for s in REF_SMILES if s in idx]
    if len(order) != len(REF_SMILES):
        print(f'[warn] {path} incomplete; skipped'); continue
    for key, label in mapping.items():
        if key in z.files:
            arr = list(z[key])
            ATTR[label] = [np.abs(np.asarray(arr[i], dtype=float)) for i in order]

if not ATTR:
    sys.exit('No attribution files found.')

MOLS = [Chem.MolFromSmiles(s) for s in REF_SMILES]


# ======================================================================
# narrow pharmacophore (identical definition to run_13)
# ======================================================================
QUINAZOLINE = Chem.MolFromSmarts('n1cnc2ccccc2c1')
WARHEAD     = Chem.MolFromSmarts('[CX3]=[CX3][CX3](=O)[NX3]')

def quinazoline_nitrogens(mol):
    ri = mol.GetRingInfo()
    n1, n3 = set(), set()
    for match in mol.GetSubstructMatches(QUINAZOLINE):
        for a in [x for x in match if mol.GetAtomWithIdx(x).GetSymbol() == 'N']:
            fused = any(ri.NumAtomRings(nb.GetIdx()) >= 2
                        for nb in mol.GetAtomWithIdx(a).GetNeighbors())
            (n1 if fused else n3).add(a)
    return n1, n3

def warhead_carbons(mol):
    return {m[i] for m in mol.GetSubstructMatches(WARHEAD) for i in (0, 1)}


# ======================================================================
# choose the molecule to display
# ======================================================================
REF_DRUGS = {
    'Gefitinib':  'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
    'Erlotinib':  'COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC',
    'Lapatinib':  'CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1',
    'Afatinib':   'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1',
}
canon = {}
for name, smi in REF_DRUGS.items():
    m = Chem.MolFromSmiles(smi)
    if m:
        canon[Chem.MolToSmiles(m)] = name

pick, pick_name = None, None
for i, s in enumerate(REF_SMILES):
    if s in canon:
        pick, pick_name = i, canon[s]
        break

if pick is None:
    # no reference drug in the evaluated set: take the most potent compound
    # that carries both a quinazoline N1 and a broad motif
    df0 = pd.read_csv(args.data).set_index('smiles')
    best, best_p = None, -np.inf
    for i, s in enumerate(REF_SMILES):
        n1, _ = quinazoline_nitrogens(MOLS[i])
        if not n1 or s not in df0.index:
            continue
        pot = float(df0.loc[s, 'pIC50'])
        if pot > best_p:
            best, best_p = i, pot
    pick = best
    pick_name = f'Test compound (pIC50 {best_p:.2f})'

if pick is None:
    sys.exit('could not choose a molecule to display')

mol = MOLS[pick]
n1, n3 = quinazoline_nitrogens(mol)
war    = warhead_carbons(mol)
narrow = sorted(n1 | n3 | war)

say('=' * 70)
say('R7  figure')
say('=' * 70)
say(f'molecule displayed : {pick_name}')
say(f'SMILES             : {REF_SMILES[pick]}')
say(f'heavy atoms        : {mol.GetNumAtoms()}')
say(f'N1 atom indices    : {sorted(n1)}')
say(f'N3 atom indices    : {sorted(n3)}')
say(f'warhead carbons    : {sorted(war)}')
say(f'narrow target      : {narrow}  ({len(narrow)/mol.GetNumAtoms():.1%} of atoms)')


# ======================================================================
# draw
# ======================================================================
def render(mol, colors, legend, size=(420, 360)):
    """Return PNG bytes with the given per-atom highlight colours."""
    try:
        d = rdMolDraw2D.MolDraw2DCairo(*size)
    except Exception:
        return None
    o = d.drawOptions()
    o.useBWAtomPalette()
    o.legendFontSize = 20
    o.highlightRadius = 0.42
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol,
        highlightAtoms=list(colors.keys()),
        highlightAtomColors=colors,
        highlightBonds=[],
        legend=legend)
    d.FinishDrawing()
    return d.GetDrawingText()

heat = plt.get_cmap('OrRd')
panels = []

# panel 1: the structural reference
ref_colors = {}
for a in n1:
    ref_colors[a] = (0.09, 0.45, 0.80)          # direct hinge bond, blue
for a in n3:
    ref_colors[a] = (0.60, 0.78, 0.93)          # water-mediated, pale blue
for a in war:
    ref_colors[a] = (0.85, 0.33, 0.10)          # covalent warhead, orange
panels.append(('Structural reference', render(mol, ref_colors, '')))

# panels 2-4: one attribution map per representation family
WANT = ['Random forest, bit occlusion',
        'AttentiveFP, node occlusion',
        'MoLFormer, token occlusion',
        'MoLFormer, integrated grad.']
for label in WANT:
    if label not in ATTR:
        continue
    w = np.asarray(ATTR[label][pick], dtype=float)
    if w.shape[0] < mol.GetNumAtoms():
        w = np.pad(w, (0, mol.GetNumAtoms() - w.shape[0]))
    w = w[:mol.GetNumAtoms()]
    wn = w / (w.max() + 1e-12)
    colors = {i: heat(0.12 + 0.88 * float(wn[i]))[:3]
              for i in range(mol.GetNumAtoms())}
    frac = w[narrow].sum() / (w.sum() + 1e-12)
    panels.append((f'{label}\n{frac:.0%} of mass on reference atoms',
                   render(mol, colors, '')))

panels = [(t, b) for t, b in panels if b is not None]
if not panels:
    sys.exit('RDKit could not render (Cairo missing?)')

import io
n = len(panels)
fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), dpi=300)
if n == 1:
    axes = [axes]
for ax, (title, png) in zip(axes, panels):
    ax.imshow(mpimg.imread(io.BytesIO(png), format='png'))
    ax.set_title(title, fontsize=8)
    ax.axis('off')
fig.suptitle(pick_name, fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_fig = os.path.join(args.outdir, 'fig1_attribution_maps.png')
fig.savefig(out_fig, bbox_inches='tight')
say(f'\nfigure -> {out_fig}')
say('panels: ' + ' | '.join(t.split(chr(10))[0] for t, _ in panels))


# ======================================================================
# R2  assay provenance
# ======================================================================
say('\n' + '=' * 70)
say('R2  assay provenance and replicate reliability')
say('=' * 70)

if not os.path.exists(args.raw):
    say(f'[skip] {args.raw} not found - run with --raw pointing at egfr_raw.csv')
else:
    raw = pd.read_csv(args.raw, low_memory=False)
    say(f'raw ChEMBL rows                     : {len(raw):,}')
    for col, name in [('assay_chembl_id', 'distinct assays'),
                      ('document_chembl_id', 'distinct source documents'),
                      ('molecule_chembl_id', 'distinct molecules')]:
        if col in raw.columns:
            say(f'{name:36s}: {raw[col].nunique():,}')

    if 'standard_relation' in raw.columns:
        rel = raw.standard_relation.value_counts(dropna=False).to_dict()
        say(f'{"relation operators":36s}: {rel}')
        cens = len(raw) - int((raw.standard_relation == '=').sum())
        say(f'{"censored / non-exact rows removed":36s}: {cens:,}')

    # wild-type vs mutant, using the same rule as the build script
    import re
    PATTERNS = [r'L858R', r'T790M', r'T\s*790', r'L\s*858', r'C797S',
                r'G719[SAC]?', r'exon\s*19', r'del\s*19', r'E746', r'A750',
                r'L861Q', r'S768I', r'insertion', r'exon\s*20',
                r'double\s*mutant', r'triple\s*mutant', r'\bmutant\b',
                r'\bmutation\b']
    MUT = re.compile('|'.join(PATTERNS), flags=re.I)
    if 'assay_description' in raw.columns:
        desc = raw.assay_description.fillna('')
        fld = raw['assay_variant_mutation'] if 'assay_variant_mutation' in raw.columns \
              else pd.Series([None] * len(raw))
        is_mut = fld.notna() | desc.str.contains(MUT)
        say(f'{"rows flagged mutant (excluded)":36s}: {int(is_mut.sum()):,}')
        say(f'{"rows retained as wild type":36s}: {int((~is_mut).sum()):,}')

        # assay format heterogeneity
        fmt = {
            'enzymatic / kinase activity': r'kinase|enzym|phosphoryl|ATPase',
            'binding / displacement':      r'binding|displace|affinit',
            'cell-based':                  r'cell|prolifer|viabilit|growth',
        }
        say('\nassay format (keyword match on the description, wild-type rows):')
        wt = desc[~is_mut]
        for k, pat in fmt.items():
            say(f'   {k:30s}: {int(wt.str.contains(pat, case=False, regex=True).sum()):,}')

    # replicate agreement
    if {'canonical_smiles', 'pchembl_value'} <= set(raw.columns):
        r = raw[raw.standard_relation.eq('=')] if 'standard_relation' in raw.columns else raw
        r = r.dropna(subset=['canonical_smiles', 'pchembl_value'])
        g = r.groupby('canonical_smiles')['pchembl_value']
        agg = g.agg(['count', 'min', 'max', 'median'])
        agg['spread'] = agg['max'] - agg['min']
        rep = agg[agg['count'] > 1]
        say('\nreplicate agreement (pChEMBL units):')
        say(f'{"   structures with >1 measurement":36s}: {len(rep):,} '
            f'({len(rep)/max(len(agg),1):.1%} of structures)')
        if len(rep):
            say(f'{"   median spread among replicates":36s}: {rep.spread.median():.2f}')
            say(f'{"   mean spread among replicates":36s}: {rep.spread.mean():.2f}')
            say(f'{"   90th percentile of spread":36s}: {rep.spread.quantile(0.9):.2f}')
            over = int((rep.spread > 1.0).sum())
            say(f'{"   structures dropped (spread > 1)":36s}: {over:,} '
                f'({over/max(len(agg),1):.1%} of structures)')

final = pd.read_csv(args.data)
say(f'\nfinal curated dataset                : {len(final):,} structures')
if 'split' in final.columns:
    say(f'{"split sizes":36s}: {final.split.value_counts().to_dict()}')

say('\n' + '=' * 70)
say('done')
say('=' * 70)
LOG.close()
