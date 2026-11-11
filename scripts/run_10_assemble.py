#!/usr/bin/env python
"""
run_10_assemble.py — build the paper's results section.

Combines the corrected Random Forest attribution (bit occlusion) with the GNN
and transformer attributions from run_08, discards the invalid SHAP rows, and
produces every table and figure the paper needs.

    python run_10_assemble.py --data egfr_final.csv \
        --xai-dir results_xai --rf-dir results_xai_rf --outdir results_final

Outputs
    table2_faithfulness.csv    broad, narrow, selectivity, with controls
    table3_agreement.csv       do the models point at the same atoms?
    figures/fig3_enrichment.png
    figures/fig4_agreement.png
    figures/fig5_examples.png  molecules coloured by attribution
"""

import os, json
import numpy as np
import pandas as pd
from scipy import stats
from common import Ctx, parse_args

args = parse_args(lambda p: (
    p.add_argument('--xai-dir', default='results_xai'),
    p.add_argument('--rf-dir', default='results_xai_rf'),
    p.add_argument('--drop-shap', action='store_true', default=True),
))
ctx = Ctx(args)
train, val, test = ctx.splits()

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


# ----------------------------------------------------------------------
# load both attribution files and align them on molecule
# ----------------------------------------------------------------------
main = np.load(os.path.join(args.xai_dir, 'attributions.npz'), allow_pickle=True)
rf = np.load(os.path.join(args.rf_dir, 'rf_attributions.npz'), allow_pickle=True)

smiles_main = list(main['smiles'])
smiles_rf = list(rf['smiles'])
if smiles_main != smiles_rf:
    raise SystemExit('the two files cover different molecules; rerun both on '
                     'the same --data')
SMILES = smiles_main

ATTR = {}
for k in main.files:
    if k == 'smiles':
        continue
    if args.drop_shap and 'SHAP' in k:
        ctx.log(f'DISCARDED (invalid, see check_rf.py): {k}')
        continue
    ATTR[k.replace('_', ' ')] = list(main[k])
for k in rf.files:
    if k != 'smiles':
        ATTR[k.replace('_', ' ')] = list(rf[k])

ctx.log(f'\nmodels in the final table ({len(SMILES)} molecules):')
for k in ATTR:
    ctx.log(f'  {k}')

test = test[test.smiles.isin(SMILES)].set_index('smiles').loc[SMILES].reset_index()
MOLS = [Chem.MolFromSmiles(s) for s in SMILES]
NAT = [m.GetNumAtoms() for m in MOLS]
BROAD = [set(json.loads(c)) for c in test.core_atoms.values]

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


# ----------------------------------------------------------------------
# TABLE 2 — faithfulness
# ----------------------------------------------------------------------
# selectivity = narrow / broad. It asks how sharply an attribution separates the
# few atoms that truly bind from the wider pharmacophoric neighbourhood. A method
# spread evenly over the whole region scores ~1; one that isolates the anchors
# scores well above 1.
ctx.log('\n=== TABLE 2: faithfulness ===')
rows = []
for label, alist in ATTR.items():
    r = {'model': label}
    for tag, targets in [('broad', BROAD), ('narrow', NARROW)]:
        vals = [e for e in (enrichment(a, t, n)
                            for a, t, n in zip(alist, targets, NAT)) if e]
        if not vals:
            continue
        v = np.array(vals)
        r[f'{tag}_enrichment'] = round(float(v.mean()), 3)
        r[f'{tag}_median'] = round(float(np.median(v)), 3)
        r[f'{tag}_above_chance'] = round(float((v > 1).mean()), 3)
        r[f'{tag}_n'] = len(v)
        r[f'{tag}_p'] = f'{stats.wilcoxon(v-1.0).pvalue:.2g}'
    if 'narrow_enrichment' in r and r.get('broad_enrichment'):
        r['selectivity'] = round(r['narrow_enrichment']/r['broad_enrichment'], 3)
    rows.append(r)

t2 = pd.DataFrame(rows).sort_values('narrow_enrichment', ascending=False)
t2.to_csv(ctx.path('table2_faithfulness.csv'), index=False)
ctx.log(t2[['model', 'broad_enrichment', 'narrow_enrichment', 'selectivity',
            'narrow_above_chance', 'narrow_p']].to_string(index=False))


# ----------------------------------------------------------------------
# TABLE 3 — agreement, recomputed with the corrected Random Forest
# ----------------------------------------------------------------------
ctx.log('\n=== TABLE 3: do the models point at the same atoms? ===')
ctx.log('(mean per-molecule Spearman correlation of atom attributions)')
labels = list(ATTR)
agree = pd.DataFrame(index=labels, columns=labels, dtype=float)
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        if i > j:
            continue
        cors = []
        for x, y in zip(ATTR[a], ATTR[b]):
            x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
            if len(x) == len(y) > 3 and x.std() > 0 and y.std() > 0:
                cors.append(stats.spearmanr(x, y).correlation)
        v = round(float(np.nanmean(cors)), 3)
        agree.loc[a, b] = v; agree.loc[b, a] = v
agree.to_csv(ctx.path('table3_agreement.csv'))
ctx.log(agree.to_string())
ctx.log('\nNOTE: the -0.388 reported by run_08 used the invalid SHAP values and')
ctx.log('must not be cited. This matrix supersedes it.')


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# fig 3 — enrichment, broad vs narrow
try:
    order = list(t2.model)
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x-w/2, t2.broad_enrichment, w, label='broad pharmacophore (~27% of atoms)')
    ax.bar(x+w/2, t2.narrow_enrichment, w, label='binding anchors only (~7% of atoms)')
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' (', '\n(') for m in order], fontsize=7)
    ax.set_ylabel('enrichment over chance')
    ax.set_title('How the definition of the target changes the conclusion')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ctx.path('figures', 'fig3_enrichment.png'), dpi=300)
    plt.close(fig)
    ctx.log(f"\nfigure -> {ctx.path('figures','fig3_enrichment.png')}")
except Exception as e:
    ctx.log(f'fig3 failed ({type(e).__name__})')

# fig 4 — agreement heatmap
try:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    m = agree.values.astype(float)
    im = ax.imshow(m, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l[:18] for l in labels], rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels([l[:18] for l in labels], fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{m[i,j]:.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label='Spearman')
    ax.set_title('Attribution agreement between models', fontsize=10)
    fig.tight_layout()
    fig.savefig(ctx.path('figures', 'fig4_agreement.png'), dpi=300)
    plt.close(fig)
    ctx.log(f"figure -> {ctx.path('figures','fig4_agreement.png')}")
except Exception as e:
    ctx.log(f'fig4 failed ({type(e).__name__})')

# fig 5 — example molecules coloured by attribution
try:
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    # pick molecules that carry a sharp anchor and are well predicted
    cand = [i for i in range(len(MOLS)) if NARROW[i] and 20 <= NAT[i] <= 40]
    picks = cand[:3]
    models = [m for m in ATTR if 'control' not in m.lower()][:3]

    fig, axes = plt.subplots(len(picks), len(models),
                             figsize=(3.4*len(models), 3.0*len(picks)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(picks):
        mol = MOLS[i]
        for c, label in enumerate(models):
            a = np.asarray(ATTR[label][i], dtype=float)
            if a.sum() > 0:
                a = a/a.max()
            colors = {k: (1.0, 1.0-float(a[k]), 1.0-float(a[k]))
                      for k in range(min(len(a), mol.GetNumAtoms()))}
            d = rdMolDraw2D.MolDraw2DCairo(340, 300)
            rdMolDraw2D.PrepareAndDrawMolecule(
                d, mol, highlightAtoms=list(colors),
                highlightAtomColors=colors)
            d.FinishDrawing()
            import io
            from PIL import Image
            axes[r, c].imshow(Image.open(io.BytesIO(d.GetDrawingText())))
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(label.replace(' (', '\n('), fontsize=8)
    fig.suptitle('Attribution on the same molecules (darker red = more important)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(ctx.path('figures', 'fig5_examples.png'), dpi=250)
    plt.close(fig)
    ctx.log(f"figure -> {ctx.path('figures','fig5_examples.png')}")
except Exception as e:
    ctx.log(f'fig5 failed ({type(e).__name__}: {str(e)[:120]})')


# ----------------------------------------------------------------------
ctx.log('\n=== what to put in the paper ===')
best = t2.iloc[0]
ctx.log(f"strongest alignment: {best.model}  "
        f"narrow {best.narrow_enrichment}  selectivity {best.get('selectivity')}")
ctx.log('Table 2 -> faithfulness   Table 3 -> agreement')
ctx.log('Fig 3 -> broad vs narrow  Fig 4 -> agreement  Fig 5 -> examples')
ctx.log('done')
