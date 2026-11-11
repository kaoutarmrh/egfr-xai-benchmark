#!/usr/bin/env python
"""
run_11_transformer_occlusion.py — a second opinion on the transformer.

WHY THIS EXISTS
The paper claims chemical language models produce explanations no better than
chance. That claim currently rests on ONE method: Integrated Gradients with a
zero-embedding baseline. A zero embedding is not a meaningful "absence of
molecule", and a referee will say so.

For the GNN we ran two independent methods (gradient and occlusion) and they
agreed, which is why that finding is safe. The transformer needs the same
treatment. This script masks each token and measures how far the prediction
moves - no gradients, no baseline choice, no assumptions.

  occlusion also at chance -> the negative finding is about the MODEL. Safe.
  occlusion well above chance -> the finding was about Integrated Gradients,
                                 and the paper's claim must be rewritten.

    python run_11_transformer_occlusion.py --data egfr_final.csv \
        --outdir results_tf_occ \
        --tf-ckpt results_tf/models/molformer_xl__cls_pool_.pt \
        --tf-name ibm/MoLFormer-XL-both-10pct --tf-pooling cls
"""

import os, re, json
import numpy as np
import pandas as pd
from scipy import stats
from common import Ctx, parse_args

args = parse_args(lambda p: (
    p.add_argument('--tf-ckpt', default='results_tf/models/molformer_xl__cls_pool_.pt'),
    p.add_argument('--tf-name', default='ibm/MoLFormer-XL-both-10pct'),
    p.add_argument('--tf-pooling', default='cls', choices=['cls', 'mean']),
    p.add_argument('--limit', type=int, default=None),
))
ctx = Ctx(args)
train, val, test = ctx.splits()
if args.limit:
    test = test.head(args.limit)

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ctx.log(f'device: {DEV}')

MOLS = [Chem.MolFromSmiles(s) for s in test.smiles.values]
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

ATOM_RE = re.compile(r"\[[^\]]+\]|Br|Cl|Si|Se|se|as|te|[BCNOPSFIbcnops]")


def atom_char_spans(mol):
    smi = Chem.MolToSmiles(mol)
    order = list(map(int, mol.GetProp('_smilesAtomOutputOrder')[1:-1]
                     .split(',')[:mol.GetNumAtoms()]))
    spans = []
    for k, m in enumerate(ATOM_RE.finditer(smi)):
        if k < len(order):
            spans.append((m.start(), m.end(), order[k]))
    return smi, spans


class Regressor(nn.Module):
    def __init__(self, ckpt, pooling, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
        self.pooling = pooling
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, 256),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == 'mean':
            m = mask.unsqueeze(-1).float()
            pooled = (h*m).sum(1)/m.sum(1).clamp(min=1e-9)
        else:
            pooled = h[:, 0]
        return self.head(pooled).squeeze(-1)


tok = AutoTokenizer.from_pretrained(args.tf_name, trust_remote_code=True)
model = Regressor(args.tf_name, args.tf_pooling).to(DEV)
model.load_state_dict(torch.load(args.tf_ckpt, map_location=DEV))
model.eval()
ctx.log(f'loaded {args.tf_name} from {args.tf_ckpt}')

# what to replace a token with: the mask token if there is one, else pad
MASK_ID = tok.mask_token_id if tok.mask_token_id is not None else tok.pad_token_id
ctx.log(f'masking token id: {MASK_ID} '
        f'({"mask" if tok.mask_token_id is not None else "pad"})')

ctx.log('\ncomputing token-occlusion attribution')
ATTR, skipped = [], 0
with torch.no_grad():
    for i, mol in enumerate(MOLS):
        smi, spans = atom_char_spans(mol)
        enc = tok(smi, return_tensors='pt', return_offsets_mapping=True,
                  truncation=True, max_length=256)
        offsets = enc.pop('offset_mapping')[0].tolist()
        ids = enc['input_ids'].to(DEV)
        mask = enc['attention_mask'].to(DEV)
        L = ids.shape[1]

        # only occlude tokens that cover at least one atom
        targets = [t for t, (s, e) in enumerate(offsets)
                   if e > s and any(st < e and en > s for (st, en, _) in spans)]
        if not targets:
            ATTR.append(np.zeros(mol.GetNumAtoms())); skipped += 1; continue

        # original plus one variant per occluded token, in one batch
        batch_ids = ids.repeat(len(targets)+1, 1).clone()
        for j, t in enumerate(targets):
            batch_ids[j+1, t] = MASK_ID
        batch_mask = mask.repeat(len(targets)+1, 1)
        preds = model(batch_ids, batch_mask).cpu().numpy()
        base, off = preds[0], preds[1:]

        a = np.zeros(mol.GetNumAtoms())
        for t, p in zip(targets, off):
            v = abs(base - p)
            s, e = offsets[t]
            hits = [ai for (st, en, ai) in spans if st < e and en > s]
            for ai in hits:
                a[ai] += v/len(hits)
        ATTR.append(a)
        if (i+1) % 200 == 0:
            ctx.log(f'  {i+1}/{len(MOLS)}')

if skipped:
    ctx.log(f'  {skipped} molecules had no atom-covering tokens')

allv = np.concatenate([a for a in ATTR if len(a)])
tots = np.array([a.sum() for a in ATTR])
ctx.log(f'\nsanity: per-molecule total median {np.median(tots):.4g}  '
        f'max {tots.max():.4g}   per-atom max {allv.max():.4g}')
ctx.log('(should be the same order as the other models, roughly 1-30)')

np.savez_compressed(ctx.path('tf_occlusion_attributions.npz'),
                    smiles=test.smiles.values,
                    Transformer_occlusion=np.array(ATTR, dtype=object))


def enrichment(attr, target, n):
    attr = np.asarray(attr, dtype=float)
    tot = attr.sum()
    target = [t for t in target if t < len(attr)]
    if tot <= 0 or not target or n == 0:
        return None
    return (attr[target].sum()/tot) / (len(target)/n)


rng = np.random.RandomState(0)
SETS = {'Transformer (occlusion)': ATTR,
        'Random control': [np.abs(rng.normal(0, 1, n)) for n in NAT]}

rows = []
for label, alist in SETS.items():
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
    if r.get('broad_enrichment'):
        r['selectivity'] = round(r['narrow_enrichment']/r['broad_enrichment'], 3)
    rows.append(r)
    ctx.log(f"\n{label}")
    ctx.log(f"  broad  {r.get('broad_enrichment')}  p {r.get('broad_p')}")
    ctx.log(f"  narrow {r.get('narrow_enrichment')}  p {r.get('narrow_p')}")

out = pd.DataFrame(rows)
out.to_csv(ctx.path('transformer_occlusion.csv'), index=False)

# agreement with Integrated Gradients on the same model
xai = os.path.join('results_xai', 'attributions.npz')
if os.path.exists(xai):
    z = np.load(xai, allow_pickle=True)
    key = [k for k in z.files if 'Transformer' in k]
    if key:
        ig = list(z[key[0]])
        cors = []
        for x, y in zip(ATTR, ig):
            x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
            if len(x) == len(y) > 3 and x.std() > 0 and y.std() > 0:
                cors.append(stats.spearmanr(x, y).correlation)
        ctx.log(f'\nagreement with Integrated Gradients on the SAME model: '
                f'{np.nanmean(cors):.3f}')
        ctx.log('(for reference, the two GNN methods agreed at 0.561)')

ctx.log('\n' + out.to_string(index=False))
ctx.log('\n--- how to read this ---')
ctx.log('At chance (~1.0, p>0.05): the negative finding about chemical language')
ctx.log('  models is confirmed by two independent methods. Safe to publish.')
ctx.log('Well above chance: the finding was about Integrated Gradients, not the')
ctx.log('  model, and the paper must be rewritten around that instead.')
ctx.log('done')
