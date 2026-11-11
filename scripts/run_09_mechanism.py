#!/usr/bin/env python
"""
run_09_mechanism.py — why do the explanations miss the chemistry?

Step 8 established WHAT: no model's attribution aligns with known EGFR
pharmacophores, and the models contradict each other. This script asks WHY, and
separates two things that are easy to confuse:

  alignment with chemistry  does the model look where a chemist would?   (step 8)
  faithfulness to the model does the attribution reflect what the model
                            actually uses?                               (here)

A model can have perfectly faithful explanations that disagree with chemistry.
That is the interesting case, and it is the one your data points to.

    python run_09_mechanism.py --data egfr_final.csv --xai-dir results_xai \
        --outdir results_mech --gnn-ckpt results_gnn/models/attentivefp.pt

Outputs:
    narrow_vs_broad.csv        sharp pharmacophore atoms only
    scaffold_effect.csv        enrichment vs how common the scaffold is
    ablation.csv               comprehensiveness curves
    figures/*.png
"""

import os, re, json
import numpy as np
import pandas as pd
from common import Ctx, parse_args

args = parse_args(lambda p: (
    p.add_argument('--xai-dir', default='results_xai'),
    p.add_argument('--gnn-ckpt', default='results_gnn/models/attentivefp.pt'),
    p.add_argument('--skip-ablation', action='store_true'),
))
ctx = Ctx(args)
train, val, test = ctx.splits()

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# ----------------------------------------------------------------------
# load what step 8 produced
# ----------------------------------------------------------------------
npz = np.load(os.path.join(args.xai_dir, 'attributions.npz'), allow_pickle=True)
SMILES = list(npz['smiles'])
ATTR = {k: list(npz[k]) for k in npz.files if k != 'smiles'}
ctx.log(f'loaded attributions for {len(SMILES)} molecules')
for k in ATTR:
    ctx.log(f'  {k}')

test = test[test.smiles.isin(SMILES)].set_index('smiles').loc[SMILES].reset_index()
MOLS = [Chem.MolFromSmiles(s) for s in SMILES]
NAT = [m.GetNumAtoms() for m in MOLS]


def enrichment(attr, target, n):
    attr = np.asarray(attr, dtype=float)
    tot = attr.sum()
    target = [t for t in target if t < len(attr)]
    if tot <= 0 or not target or n == 0:
        return None
    pcs = attr[target].sum() / tot
    return pcs / (len(target) / n)


# ======================================================================
# 1. Is the pharmacophore target too generous?
# ======================================================================
# In step 8, "core atoms" averaged ~26% of each molecule, partly because the
# anilino NH linker is a loose pattern matching 60% of the dataset. Here we test
# only the sharp, mechanistically specific atoms:
#   - the quinazoline nitrogens that hydrogen-bond Met793
#   - the two alkene carbons of the Michael acceptor that alkylates Cys797
# Typically 4 atoms out of 35, so ~11% of the molecule instead of 26%.
NARROW = {
    'hinge_nitrogens': ('n1cnc2ccccc2c1', 'N_only'),
    'warhead_carbons': ('[CX3]=[CX3][CX3](=O)[NX3]', [0, 1]),
}
_NC = {k: (Chem.MolFromSmarts(v[0]), v[1]) for k, v in NARROW.items()}


def narrow_atoms(mol):
    out = set()
    for patt, keep in _NC.values():
        for m in mol.GetSubstructMatches(patt):
            if keep == 'N_only':
                out |= {a for a in m if mol.GetAtomWithIdx(a).GetSymbol() == 'N'}
            else:
                out |= {m[i] for i in keep if i < len(m)}
    return sorted(out)


NARROW_SETS = [narrow_atoms(m) for m in MOLS]
BROAD_SETS = [set(json.loads(c)) for c in test.core_atoms.values]
have_narrow = sum(1 for s in NARROW_SETS if s)
ctx.log(f'\n=== 1. narrow vs broad pharmacophore ===')
ctx.log(f'molecules with a sharp feature: {have_narrow}/{len(MOLS)}')
ctx.log(f'mean narrow target: {np.mean([len(s)/n for s, n in zip(NARROW_SETS, NAT) if s]):.1%} of atoms')
ctx.log(f'mean broad  target: {np.mean([len(s)/n for s, n in zip(BROAD_SETS, NAT) if s]):.1%} of atoms')

rows = []
from scipy import stats
for label, alist in ATTR.items():
    r = {'model': label}
    for tag, sets in [('broad', BROAD_SETS), ('narrow', NARROW_SETS)]:
        vals = [e for e in (enrichment(a, s, n) for a, s, n in zip(alist, sets, NAT)) if e]
        if vals:
            r[f'{tag}_enrichment'] = round(float(np.mean(vals)), 4)
            r[f'{tag}_n'] = len(vals)
            r[f'{tag}_p'] = f'{stats.wilcoxon(np.array(vals)-1.0).pvalue:.3g}' if len(vals) > 10 else 'na'
    rows.append(r)

nb = pd.DataFrame(rows)
nb.to_csv(ctx.path('narrow_vs_broad.csv'), index=False)
ctx.log(nb.to_string(index=False))
ctx.log('\nIf models score near chance on the BROAD target but well above it on')
ctx.log('the NARROW one, they do find the real anchor atoms and step 8 simply')
ctx.log('measured against too diffuse a target. If narrow is also at or below')
ctx.log('chance, the step 8 conclusion stands and is now much harder to argue with.')


# ======================================================================
# 2. Discriminative, not necessary?
# ======================================================================
# The quinazoline core appears in most actives. It barely varies, so it cannot
# explain why one active beats another - the substituents do. Any method that
# highlights "what would change the prediction" drifts toward what varies.
#
# NOTE: we cannot use scaffold family size here. The split is scaffold-based, so
# by construction no test scaffold appears in training and the count is always
# zero. Familiarity has to be measured by similarity instead.
#
# Prediction: the more a test molecule resembles the training set, the more its
# core is "already known" and uninformative, so enrichment should FALL.
ctx.log('\n=== 2. does enrichment fall for familiar chemistry? ===')
from rdkit.Chem import rdFingerprintGenerator, DataStructs

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ctx.log('computing nearest-neighbour similarity to the training set ...')
train_fps = []
for s in train.smiles.values:
    m = Chem.MolFromSmiles(s)
    if m is not None:
        train_fps.append(_gen.GetFingerprint(m))

max_sim = []
for m in MOLS:
    if m is None:
        max_sim.append(np.nan); continue
    sims = DataStructs.BulkTanimotoSimilarity(_gen.GetFingerprint(m), train_fps)
    max_sim.append(max(sims))
max_sim = np.array(max_sim)
ctx.log(f'nearest-neighbour Tanimoto: median {np.nanmedian(max_sim):.3f}  '
        f'range {np.nanmin(max_sim):.3f}-{np.nanmax(max_sim):.3f}')

# how many training molecules share this molecule's pharmacophore features
feat_cols = [c for c in test.columns if c.startswith('feat_')]
if feat_cols:
    train_feat = train[feat_cols].values if all(c in train.columns for c in feat_cols) else None
else:
    train_feat = None

srows = []
for label, alist in ATTR.items():
    e, s_ = [], []
    for a, core, n, ms in zip(alist, BROAD_SETS, NAT, max_sim):
        v = enrichment(a, core, n)
        if v is not None and not np.isnan(ms):
            e.append(v); s_.append(ms)
    if len(e) < 30:
        continue
    e, s_ = np.array(e), np.array(s_)
    rho, p = stats.spearmanr(s_, e)
    cut = np.median(s_)
    familiar, novel = e[s_ >= cut], e[s_ < cut]
    srows.append({'model': label,
                  'spearman_rho_vs_similarity': round(float(rho), 4),
                  'p': f'{p:.3g}',
                  'enrich_novel_chemistry': round(float(novel.mean()), 4),
                  'enrich_familiar_chemistry': round(float(familiar.mean()), 4),
                  'difference': round(float(novel.mean()-familiar.mean()), 4),
                  'n': len(e)})
    ctx.log(f'{label:32s} rho {rho:+.3f} (p={p:.2g})   '
            f'novel {novel.mean():.3f}   familiar {familiar.mean():.3f}')

se = pd.DataFrame(srows)
se.to_csv(ctx.path('scaffold_effect.csv'), index=False)
ctx.log('\nA NEGATIVE rho supports discriminative-not-necessary: the more a')
ctx.log('molecule resembles what the model was trained on, the less it')
ctx.log('attributes to the shared pharmacophore, because that core no longer')
ctx.log('distinguishes anything. A rho near zero means the explanation')
ctx.log('misses the chemistry regardless of familiarity - a simpler and')
ctx.log('bleaker finding, but equally reportable.')

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 4))
    for label, alist in ATTR.items():
        e, s_ = [], []
        for a, core, n, ms in zip(alist, BROAD_SETS, NAT, max_sim):
            v = enrichment(a, core, n)
            if v is not None and not np.isnan(ms):
                e.append(v); s_.append(ms)
        if len(e) < 30:
            continue
        e, s_ = np.array(e), np.array(s_)
        bins = np.quantile(s_, [0, .2, .4, .6, .8, 1.0])
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m_ = (s_ >= lo) & (s_ <= hi)
            if m_.sum() > 5:
                xs.append((lo+hi)/2); ys.append(e[m_].mean())
        ax.plot(xs, ys, 'o-', label=label.replace('_', ' ')[:28])
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xlabel('similarity to the training set (nearest-neighbour Tanimoto)')
    ax.set_ylabel('enrichment on known binding atoms')
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(ctx.path('figures', 'similarity_effect.png'), dpi=300); plt.close(fig)
    ctx.log(f"figure -> {ctx.path('figures','similarity_effect.png')}")
except Exception as e:
    ctx.log(f'figure failed ({type(e).__name__}) - csv is saved')


# ======================================================================
# 3. The Random Forest zero-tails anomaly
# ======================================================================
ctx.log('\n=== 3. why did Random Forest give 0.0 on solubilising tails? ===')
CTRL_SETS = [set(json.loads(c)) for c in test.control_atoms.values]
n_with = sum(1 for s in CTRL_SETS if s)
ctx.log(f'test molecules carrying a tail: {n_with}')
rf_key = [k for k in ATTR if 'Forest' in k]
if rf_key and n_with:
    a_list = ATTR[rf_key[0]]
    zeros = tot = 0
    examples = []
    for a, s in zip(a_list, CTRL_SETS):
        if not s:
            continue
        a = np.asarray(a, dtype=float)
        idx = [i for i in s if i < len(a)]
        if not idx:
            continue
        tot += 1
        m = a[idx].sum()
        if m == 0:
            zeros += 1
        elif len(examples) < 3:
            examples.append((idx[:5], np.round(a[idx][:5], 6)))
    ctx.log(f'molecules with EXACTLY zero attribution on tail atoms: {zeros}/{tot}')
    for idx, v in examples:
        ctx.log(f'  non-zero example: atoms {idx} -> {v}')
    if zeros == tot:
        ctx.log('  => every tail atom gets exactly zero. Not rounding. Worth a')
        ctx.log('     line in the paper: SHAP on ECFP4 assigns no importance at')
        ctx.log('     all to solubilising groups, which is chemically sensible.')


# ======================================================================
# 4. Ablation: is the attribution faithful to the MODEL?
# ======================================================================
# Remove the atoms the model called important, and see whether the prediction
# actually moves. Compare against removing the same number at random.
#   ratio >> 1  the explanation reflects what the model uses (faithful)
#   ratio ~ 1   the explanation is decorative even for the model itself
if not args.skip_ablation and os.path.exists(args.gnn_ckpt):
    ctx.log('\n=== 4. ablation on the GNN ===')
    import torch
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn.models import AttentiveFP

    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    ATOMS = ['C','N','O','S','F','Cl','Br','I','P','B','Si','Se','other']
    HYB = [Chem.rdchem.HybridizationType.SP,  Chem.rdchem.HybridizationType.SP2,
           Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
           Chem.rdchem.HybridizationType.SP3D2]
    BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    ATOM_DIM, BOND_DIM = len(ATOMS)+6+5+5+len(HYB)+2, len(BONDS)+2

    def oh(x, al):
        v=[0]*len(al); v[al.index(x) if x in al else len(al)-1]=1; return v
    def af(a):
        return (oh(a.GetSymbol(),ATOMS)+oh(a.GetDegree(),[0,1,2,3,4,5])
              + oh(a.GetFormalCharge(),[-2,-1,0,1,2])+oh(a.GetTotalNumHs(),[0,1,2,3,4])
              + oh(a.GetHybridization(),HYB)+[int(a.GetIsAromatic()),int(a.IsInRing())])
    def bf(b):
        return oh(b.GetBondType(),BONDS)+[int(b.GetIsConjugated()),int(b.IsInRing())]

    model = AttentiveFP(ATOM_DIM, 256, 1, BOND_DIM, 3, 2, 0.2).to(DEV)
    model.load_state_dict(torch.load(args.gnn_ckpt, map_location=DEV))
    model.eval()

    gnn_key = [k for k in ATTR if 'AttentiveFP' in k and 'occlusion' not in k.lower()]
    gnn_key = gnn_key[0] if gnn_key else [k for k in ATTR if 'AttentiveFP' in k][0]
    attrs = ATTR[gnn_key]
    ctx.log(f'ablating according to: {gnn_key}')

    FRACS = [0.1, 0.2, 0.3, 0.5]
    rng = np.random.RandomState(0)
    drops = {f: {'attr': [], 'rand': [], 'pharm': []} for f in FRACS}

    with torch.no_grad():
        for i, (mol, a) in enumerate(zip(MOLS, attrs)):
            n = mol.GetNumAtoms()
            a = np.asarray(a, dtype=float)
            if len(a) != n or n < 6:
                continue
            x0 = torch.tensor([af(at) for at in mol.GetAtoms()], dtype=torch.float)
            src,dst,ea = [],[],[]
            for b in mol.GetBonds():
                p,q = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                f = bf(b); src += [p,q]; dst += [q,p]; ea += [f,f]
            ei = (torch.tensor([src,dst],dtype=torch.long) if src
                  else torch.zeros((2,0),dtype=torch.long))
            eat = (torch.tensor(ea,dtype=torch.float) if ea
                   else torch.zeros((0,BOND_DIM),dtype=torch.float))

            order = np.argsort(-a)                      # most important first
            core = sorted(BROAD_SETS[i])
            graphs, tags = [Data(x=x0.clone(), edge_index=ei, edge_attr=eat)], []
            for fr in FRACS:
                k = max(1, int(round(fr*n)))
                sets = {'attr': order[:k],
                        'rand': rng.choice(n, size=k, replace=False),
                        'pharm': (core[:k] if len(core) >= k else None)}
                for tag, idx in sets.items():
                    if idx is None:
                        continue
                    xk = x0.clone(); xk[list(idx)] = 0.0
                    graphs.append(Data(x=xk, edge_index=ei, edge_attr=eat))
                    tags.append((fr, tag))

            batch = Batch.from_data_list(graphs).to(DEV)
            preds = model(batch.x, batch.edge_index, batch.edge_attr,
                          batch.batch).squeeze(-1).cpu().numpy()
            base = preds[0]
            for (fr, tag), p in zip(tags, preds[1:]):
                drops[fr][tag].append(abs(base - p))
            if (i+1) % 200 == 0:
                ctx.log(f'  {i+1}/{len(MOLS)}')

    arows = []
    for fr in FRACS:
        d = drops[fr]
        row = {'fraction_removed': fr,
               'drop_attributed': round(float(np.mean(d['attr'])), 4),
               'drop_random': round(float(np.mean(d['rand'])), 4),
               'drop_pharmacophore': round(float(np.mean(d['pharm'])), 4) if d['pharm'] else None,
               'n': len(d['attr'])}
        row['ratio_attr_vs_random'] = round(row['drop_attributed']/row['drop_random'], 3) \
            if row['drop_random'] else None
        if d['pharm']:
            row['ratio_pharm_vs_random'] = round(row['drop_pharmacophore']/row['drop_random'], 3) \
                if row['drop_random'] else None
        arows.append(row)
        ctx.log(f"  remove {fr:.0%}: attributed {row['drop_attributed']:.4f}  "
                f"random {row['drop_random']:.4f}  "
                f"pharmacophore {row['drop_pharmacophore']}  "
                f"ratio {row['ratio_attr_vs_random']}")

    ab = pd.DataFrame(arows)
    ab.to_csv(ctx.path('ablation.csv'), index=False)
    ctx.log('\nratio_attr_vs_random >> 1 means the attribution really does point')
    ctx.log('at what the model uses - faithful to the model, even though step 8')
    ctx.log('showed it disagrees with chemistry. Those are the two axes of your')
    ctx.log('paper, and a model can score high on one and low on the other.')

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4))
        ax.plot(ab.fraction_removed, ab.drop_attributed, 'o-', label='attributed atoms')
        ax.plot(ab.fraction_removed, ab.drop_random, 's--', label='random atoms')
        if ab.drop_pharmacophore.notna().all():
            ax.plot(ab.fraction_removed, ab.drop_pharmacophore, '^:',
                    label='known pharmacophore')
        ax.set_xlabel('fraction of atoms removed')
        ax.set_ylabel('|change in predicted pIC50|')
        ax.set_title('Ablation: does removing them matter?')
        ax.legend(); fig.tight_layout()
        fig.savefig(ctx.path('figures', 'ablation.png'), dpi=300); plt.close(fig)
        ctx.log(f"figure -> {ctx.path('figures','ablation.png')}")
    except Exception as e:
        ctx.log(f'figure failed ({type(e).__name__}) - csv is saved')
else:
    ctx.log('\n(ablation skipped)')

ctx.log('\ndone')
