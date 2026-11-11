#!/usr/bin/env python
"""
run_08_xai.py — the core experiment of the paper.

Every model answers "which part mattered?" in a different language:
  Random Forest -> fingerprint bits
  GNN           -> atoms
  Transformer   -> SMILES tokens

None of those are comparable. This script translates all of them into ONE
language - importance per atom - and then asks the question the paper is built
on: does that importance land on the parts of the molecule that medicinal
chemistry already knows matter for EGFR?

    pip install shap
    python run_08_xai.py --data egfr_final.csv --outdir results_xai \
        --gnn-ckpt results_gnn/models/attentivefp.pt \
        --tf-ckpt  results_tf/models/molformer_xl__cls_pool_.pt \
        --tf-name  ibm/MoLFormer-XL-both-10pct --tf-pooling cls

Outputs:
    attributions.npz          per-atom attribution, every model, every molecule
    faithfulness.csv          PCS and enrichment per model
    figures/enrichment.png    the headline figure
    results.csv               enrichment appended per model
"""

import os, re, json, warnings
import numpy as np
import pandas as pd
from common import Ctx, parse_args

warnings.filterwarnings('ignore')

args = parse_args(lambda p: (
    p.add_argument('--models', nargs='*', default=['random','rf_random','rf','gnn','gnn_occ','transformer']),
    p.add_argument('--limit', type=int, default=None, help='test molecules to explain'),
    p.add_argument('--gnn-ckpt', default='results_gnn/models/attentivefp.pt'),
    p.add_argument('--tf-ckpt',  default=None),
    p.add_argument('--tf-name',  default='ibm/MoLFormer-XL-both-10pct'),
    p.add_argument('--tf-pooling', default='cls', choices=['cls', 'mean']),
    p.add_argument('--ig-steps', type=int, default=32),
))
ctx = Ctx(args)
train, val, test = ctx.splits()
if args.limit:
    test = test.head(args.limit)
ctx.log(f'explaining {len(test)} test molecules')

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

MOLS = [Chem.MolFromSmiles(s) for s in test.smiles.values]
NATOMS = [m.GetNumAtoms() for m in MOLS]
CORE = [set(json.loads(c)) for c in test.core_atoms.values]
CTRL = [set(json.loads(c)) for c in test.control_atoms.values]

ATTR = {}          # model -> list of per-atom attribution arrays


# ======================================================================
# shared translation helpers
# ======================================================================
def env_atoms(mol, center, radius):
    """Atoms inside the circular environment that switched on a fingerprint bit."""
    if radius == 0:
        return {center}
    atoms = {center}
    for b in Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center):
        bd = mol.GetBondWithIdx(b)
        atoms.add(bd.GetBeginAtomIdx()); atoms.add(bd.GetEndAtomIdx())
    return atoms


ATOM_RE = re.compile(r"\[[^\]]+\]|Br|Cl|Si|Se|se|as|te|[BCNOPSFIbcnops]")

def atom_char_spans(mol):
    """Canonical SMILES plus, for each atom as it appears in the string,
    the character span it occupies and which RDKit atom it is."""
    smi = Chem.MolToSmiles(mol)
    order = list(map(int, mol.GetProp('_smilesAtomOutputOrder')[1:-1]
                     .split(',')[:mol.GetNumAtoms()]))
    spans = []
    for k, m in enumerate(ATOM_RE.finditer(smi)):
        if k < len(order):
            spans.append((m.start(), m.end(), order[k]))
    return smi, spans


# ======================================================================
# 1. RANDOM FOREST  ->  SHAP on bits  ->  atoms
# ======================================================================
def explain_rf():
    import shap
    from sklearn.ensemble import RandomForestRegressor

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp_matrix(smiles):
        X = np.zeros((len(smiles), 2048), dtype=np.uint8)
        for i, s in enumerate(smiles):
            m = Chem.MolFromSmiles(s)
            if m is not None:
                X[i] = np.array(gen.GetFingerprint(m))
        return X

    ctx.log('training Random Forest (deterministic, seed 42)')
    Xtr = fp_matrix(train.smiles.values)
    rf = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                               n_jobs=-1, random_state=42)
    rf.fit(Xtr, train.pIC50.values)

    Xte = fp_matrix(test.smiles.values)
    ctx.log(f'RF test R2 check: {rf.score(Xte, test.pIC50.values):.4f}')

    ctx.log('computing TreeSHAP (this is the slow part)')
    sv = shap.TreeExplainer(rf).shap_values(Xte, check_additivity=False)

    out = []
    for i, mol in enumerate(MOLS):
        ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=ao)
        binfo = ao.GetBitInfoMap()

        a = np.zeros(mol.GetNumAtoms())
        for bit, occ in binfo.items():
            v = sv[i, bit]
            if v == 0:
                continue
            # a bit's importance is shared evenly among the atoms of its
            # environment, and among its occurrences in the molecule
            for center, radius in occ:
                atoms = env_atoms(mol, center, radius)
                for at in atoms:
                    a[at] += v / len(atoms) / len(occ)
        out.append(np.abs(a))
        if (i+1) % 200 == 0:
            ctx.log(f'  {i+1}/{len(MOLS)}')
    return out


# ======================================================================
# 2. GNN  ->  gradient x input on node features  ->  atoms (native)
# ======================================================================
def explain_gnn():
    import torch, torch.nn as nn
    from torch_geometric.data import Data
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
    ctx.log(f'loaded GNN from {args.gnn_ckpt}')

    out = []
    for i, mol in enumerate(MOLS):
        x = torch.tensor([af(a) for a in mol.GetAtoms()], dtype=torch.float, device=DEV)
        src, dst, ea = [], [], []
        for b in mol.GetBonds():
            p, q = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            f = bf(b); src += [p,q]; dst += [q,p]; ea += [f,f]
        ei = (torch.tensor([src,dst], dtype=torch.long, device=DEV) if src
              else torch.zeros((2,0), dtype=torch.long, device=DEV))
        eattr = (torch.tensor(ea, dtype=torch.float, device=DEV) if ea
                 else torch.zeros((0,BOND_DIM), dtype=torch.float, device=DEV))
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=DEV)

        x = x.clone().requires_grad_(True)
        pred = model(x, ei, eattr, batch).squeeze()
        grad = torch.autograd.grad(pred, x)[0]
        a = (grad * x).sum(-1).abs().detach().cpu().numpy()   # gradient x input
        out.append(a)
        if (i+1) % 200 == 0:
            ctx.log(f'  {i+1}/{len(MOLS)}')
    return out


# ======================================================================
# 3. TRANSFORMER  ->  Integrated Gradients on tokens  ->  atoms
# ======================================================================
def explain_transformer():
    import torch, torch.nn as nn
    from transformers import AutoTokenizer, AutoModel

    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

    class Regressor(nn.Module):
        def __init__(self, ckpt, pooling, dropout=0.1):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
            self.pooling = pooling
            h = self.encoder.config.hidden_size
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h,256),
                                      nn.GELU(), nn.Dropout(dropout), nn.Linear(256,1))
        def from_embeds(self, embeds, mask):
            h = self.encoder(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
            if self.pooling == 'mean':
                m = mask.unsqueeze(-1).float()
                pooled = (h*m).sum(1)/m.sum(1).clamp(min=1e-9)
            else:
                pooled = h[:,0]
            return self.head(pooled).squeeze(-1)

    tok = AutoTokenizer.from_pretrained(args.tf_name, trust_remote_code=True)
    model = Regressor(args.tf_name, args.tf_pooling).to(DEV)
    if args.tf_ckpt and os.path.exists(args.tf_ckpt):
        model.load_state_dict(torch.load(args.tf_ckpt, map_location=DEV))
        ctx.log(f'loaded transformer weights from {args.tf_ckpt}')
    else:
        ctx.log('WARNING: no transformer checkpoint given; explaining an '
                'un-finetuned model is not meaningful')
    model.eval()
    emb_layer = model.encoder.get_input_embeddings()

    out, failures = [], 0
    for i, mol in enumerate(MOLS):
        smi, spans = atom_char_spans(mol)
        enc = tok(smi, return_tensors='pt', return_offsets_mapping=True,
                  truncation=True, max_length=256)
        offsets = enc.pop('offset_mapping')[0].tolist()
        ids = enc['input_ids'].to(DEV); mask = enc['attention_mask'].to(DEV)

        try:
            emb = emb_layer(ids).detach()
            base = torch.zeros_like(emb)
            total = torch.zeros_like(emb)
            for alpha in np.linspace(1.0/args.ig_steps, 1.0, args.ig_steps):
                x = (base + float(alpha)*(emb-base)).requires_grad_(True)
                pred = model.from_embeds(x, mask)
                total += torch.autograd.grad(pred, x)[0]
            ig = ((emb-base) * total/args.ig_steps).sum(-1).abs()[0].cpu().numpy()
        except Exception:
            # fall back to gradient x input if inputs_embeds is unsupported
            failures += 1
            emb = emb_layer(ids).detach().requires_grad_(True)
            pred = model.from_embeds(emb, mask)
            g = torch.autograd.grad(pred, emb)[0]
            ig = (g*emb).sum(-1).abs()[0].detach().cpu().numpy()

        # token attribution -> atom attribution, by character overlap
        a = np.zeros(mol.GetNumAtoms())
        for t,(s,e) in enumerate(offsets):
            if e <= s or t >= len(ig):
                continue                                  # special token
            hits = [ai for (st,en,ai) in spans if st < e and en > s]
            if hits:
                for ai in hits:
                    a[ai] += ig[t]/len(hits)
        out.append(a)
        if (i+1) % 200 == 0:
            ctx.log(f'  {i+1}/{len(MOLS)}')
    if failures:
        ctx.log(f'  note: fell back to gradient-x-input for {failures} molecules')
    return out


# ======================================================================
# 4b. CONTROL: random values pushed through the bit -> atom mapping
# ======================================================================
def explain_rf_random():
    """Tests the TRANSLATION, not any model.

    Spreading a bit's importance over the atoms of its environment can bias the
    result: an atom inside many overlapping environments accumulates more mass
    than an isolated one. Ring systems - which is where the EGFR pharmacophore
    lives - are exactly where environments overlap most.

    So: give the bits that are actually set a random importance, push it through
    the identical pipeline, and measure enrichment.
      ~1.0  -> the mapping is fair, and the Random Forest result is about SHAP
      <1.0  -> the mapping itself penalises the pharmacophore, and the Random
               Forest enrichment must be reported relative to THIS, not to 1.0
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rng = np.random.RandomState(0)
    out = []
    for mol in MOLS:
        ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=ao)
        binfo = ao.GetBitInfoMap()
        a = np.zeros(mol.GetNumAtoms())
        for bit, occ in binfo.items():
            v = rng.normal(0, 1)                    # random "SHAP" value
            for center, radius in occ:
                atoms = env_atoms(mol, center, radius)
                for at in atoms:
                    a[at] += v/len(atoms)/len(occ)
        out.append(np.abs(a))
    return out


# ======================================================================
# 4. CONTROL: random attribution
# ======================================================================
def explain_random():
    """Sanity check on the METRIC itself, not on any model.

    Random attribution must score an enrichment of ~1.0. If it does not, the
    metric is biased and every other number here is meaningless. Always report
    this row in the paper - it is what lets a reader trust the rest.
    """
    rng = np.random.RandomState(0)
    return [np.abs(rng.normal(0, 1, m.GetNumAtoms())) for m in MOLS]


# ======================================================================
# 5. GNN by occlusion  ->  a second opinion on the same model
# ======================================================================
def explain_gnn_occlusion():
    """Mask one atom at a time and measure how much the prediction moves.

    Gradients are a local, first-order approximation and can be misleading.
    Occlusion is slower but direct: it actually removes the atom and looks.
    If gradient and occlusion disagree about the same model, the problem is the
    attribution METHOD. If they agree, the finding is about the MODEL.
    """
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

    out = []
    with torch.no_grad():
        for i, mol in enumerate(MOLS):
            n = mol.GetNumAtoms()
            x0 = torch.tensor([af(a) for a in mol.GetAtoms()], dtype=torch.float)
            src, dst, ea = [], [], []
            for b in mol.GetBonds():
                p,q = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                f = bf(b); src += [p,q]; dst += [q,p]; ea += [f,f]
            ei = (torch.tensor([src,dst], dtype=torch.long) if src
                  else torch.zeros((2,0), dtype=torch.long))
            eat = (torch.tensor(ea, dtype=torch.float) if ea
                   else torch.zeros((0,BOND_DIM), dtype=torch.float))

            # original plus one variant per masked atom, all in one batch
            graphs = [Data(x=x0.clone(), edge_index=ei, edge_attr=eat)]
            for k in range(n):
                xk = x0.clone(); xk[k] = 0.0
                graphs.append(Data(x=xk, edge_index=ei, edge_attr=eat))
            batch = Batch.from_data_list(graphs).to(DEV)
            preds = model(batch.x, batch.edge_index, batch.edge_attr,
                          batch.batch).squeeze(-1).cpu().numpy()
            out.append(np.abs(preds[0] - preds[1:]))
            if (i+1) % 100 == 0:
                ctx.log(f'  {i+1}/{len(MOLS)}')
    return out


# ======================================================================
# run the requested explainers
# ======================================================================
EXPLAINERS = {'random': ('Random control', explain_random),
              'rf_random': ('Random via bit-map (control)', explain_rf_random),
              'rf': ('Random Forest (SHAP)', explain_rf),
              'gnn': ('AttentiveFP (grad x input)', explain_gnn),
              'gnn_occ': ('AttentiveFP (occlusion)', explain_gnn_occlusion),
              'transformer': ('Transformer (IG)', explain_transformer)}

for key in args.models:
    if key not in EXPLAINERS:
        ctx.log(f'unknown model key: {key}'); continue
    label, fn = EXPLAINERS[key]
    ctx.log(f'\n=== {label} ===')
    try:
        ATTR[label] = fn()
        ctx.log(f'  done, {len(ATTR[label])} molecules explained')
    except Exception as e:
        import traceback
        ctx.log(f'  FAILED: {type(e).__name__}: {str(e)[:200]}')
        ctx.log(traceback.format_exc()[-700:])

if not ATTR:
    raise SystemExit('no explanations produced')


# ======================================================================
# THE METRIC
# ======================================================================
def enrichment(attr, core, n):
    """How concentrated is the attribution on known binding atoms?

    PCS        share of attribution mass landing on core atoms
    enrichment PCS divided by what a random attribution would give.
               1.0 = chance. >1 = the model prefers the real chemistry.
    """
    tot = attr.sum()
    if tot <= 0 or not core or n == 0:
        return None
    core = [c for c in core if c < len(attr)]
    if not core:
        return None
    pcs = attr[core].sum()/tot
    baseline = len(core)/n
    return pcs, pcs/baseline


rows, per_mol = [], {}
for label, alist in ATTR.items():
    pcs_all, enr_all, ctrl_all = [], [], []
    for a, core, ctrl, n in zip(alist, CORE, CTRL, NATOMS):
        r = enrichment(np.asarray(a, dtype=float), core, n)
        if r:
            pcs_all.append(r[0]); enr_all.append(r[1])
        rc = enrichment(np.asarray(a, dtype=float), ctrl, n)
        if rc:
            ctrl_all.append(rc[1])

    from scipy import stats
    t, p = stats.wilcoxon(np.array(enr_all)-1.0) if len(enr_all) > 10 else (np.nan, np.nan)
    row = {'model': label, 'n_molecules': len(enr_all),
           'PCS_mean': round(float(np.mean(pcs_all)), 4),
           'enrichment_mean': round(float(np.mean(enr_all)), 4),
           'enrichment_median': round(float(np.median(enr_all)), 4),
           'enrichment_sd': round(float(np.std(enr_all)), 4),
           'frac_above_chance': round(float(np.mean(np.array(enr_all) > 1.0)), 4),
           'control_enrichment': round(float(np.mean(ctrl_all)), 4) if ctrl_all else None,
           'wilcoxon_p': f'{p:.3g}'}
    rows.append(row); per_mol[label] = enr_all
    ctx.log(f"\n{label}")
    ctx.log(f"  attribution on known binding atoms : {row['PCS_mean']:.3f} of total")
    ctx.log(f"  enrichment over chance             : {row['enrichment_mean']:.3f}")
    ctx.log(f"  molecules above chance             : {row['frac_above_chance']:.1%}")
    ctx.log(f"  enrichment on solubilising tails   : {row['control_enrichment']}")
    ctx.log(f"  Wilcoxon vs chance                 : p = {row['wilcoxon_p']}")

faith = pd.DataFrame(rows)
faith.to_csv(ctx.path('faithfulness.csv'), index=False)
np.savez_compressed(ctx.path('attributions.npz'),
                    smiles=test.smiles.values,
                    **{k.replace(' ','_'): np.array(v, dtype=object)
                       for k, v in ATTR.items()})


# ======================================================================
# cross-model agreement
# ======================================================================
if len(ATTR) > 1:
    from scipy.stats import spearmanr
    labels = list(ATTR)
    ctx.log('\n=== do the models point at the same atoms? ===')
    ctx.log('(mean Spearman correlation of per-atom attribution)')
    agree = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i > j:
                continue
            cors = []
            for x, y in zip(ATTR[a], ATTR[b]):
                x, y = np.asarray(x,dtype=float), np.asarray(y,dtype=float)
                if len(x) == len(y) and len(x) > 3 and x.std() > 0 and y.std() > 0:
                    cors.append(spearmanr(x, y).correlation)
            v = round(float(np.nanmean(cors)), 3)
            agree.loc[a,b] = v; agree.loc[b,a] = v
    ctx.log(agree.to_string())
    agree.to_csv(ctx.path('attribution_agreement.csv'))


# ======================================================================
# headline figure
# ======================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    labels = list(per_mol)
    names = [l.replace(' (', '\n(') for l in labels]
    try:
        ax.boxplot([per_mol[l] for l in labels], tick_labels=names, showfliers=False)
    except TypeError:
        ax.boxplot([per_mol[l] for l in labels], labels=names, showfliers=False)
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.text(0.02, 1.02, 'chance', transform=ax.get_yaxis_transform(), fontsize=8)
    ax.set_ylabel('enrichment on known EGFR binding atoms')
    ax.set_title('Do the models attend to the real pharmacophore?')
    fig.tight_layout()
    fig.savefig(ctx.path('figures', 'enrichment.png'), dpi=300)
    plt.close(fig)
    ctx.log(f"\nfigure -> {ctx.path('figures','enrichment.png')}")
except Exception as e:
    ctx.log(f'figure failed ({type(e).__name__}) - results are still saved')

for r in rows:
    ctx.record(r['model'] + ' [XAI]', 'atom attribution',
               np.array([1.0]), np.array([1.0]),
               extra={'enrichment': r['enrichment_mean'],
                      'PCS': r['PCS_mean'],
                      'frac_above_chance': r['frac_above_chance']})

ctx.log('\n' + faith.to_string(index=False))
ctx.log('done')
