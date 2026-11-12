#!/usr/bin/env python
"""
run_16_ig_sensitivity.py — the two experiments the reviewer still wants.

(C) Is the integrated-gradients result an artefact of the baseline or of
    under-integration?  run_08_xai.py used a ZERO embedding baseline and 32
    steps.  A zero embedding is not a valid token embedding, so the attribution
    is integrated along a path that leaves the model's input manifold.  We vary
    both factors and also report the COMPLETENESS error, the quantity integrated
    gradients is supposed to satisfy:

        sum_i IG_i  ==  f(x) - f(baseline)

    A large relative error means the attribution has not converged and the
    numbers cannot be interpreted; a small error means the path integral is
    sound and the chance-level enrichment is a property of the model, not of
    the estimator.  This is the difference between speculating about the cause
    and measuring it.

(D) Faithfulness for MoLFormer, so that the ablation is not reported for
    AttentiveFP alone: mask the tokens covering the top-k% atoms by attribution,
    re-predict, and compare with masking k% of atoms at random.  Run for both
    token occlusion and integrated gradients.

Needs a GPU (a Colab T4 is enough).  Nothing is retrained.

    python run_16_ig_sensitivity.py --data egfr_final.csv \
        --tf-ckpt results_tf/models/molformer_xl__cls_pool_.pt \
        --tf-name ibm/MoLFormer-XL-both-10pct --tf-pooling cls \
        --outdir results_revision

Writes: ig_sensitivity.csv, ig_completeness.csv, tf_faithfulness.csv,
NOTE: ig_completeness.csv is not in this repository. See results/PROVENANCE.md:
only its median column was recovered after a runtime reset, and that column is
already in ig_sensitivity.csv, so a half-reconstructed file was not published.
Re-running this script regenerates it.

        ig_sensitivity_log.txt
"""

import os, re, json, argparse, time
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

p = argparse.ArgumentParser()
p.add_argument('--data',       default='egfr_final.csv')
p.add_argument('--outdir',     default='results_revision')
p.add_argument('--tf-ckpt',    default='results_tf/models/molformer_xl__cls_pool_.pt')
p.add_argument('--tf-name',    default='ibm/MoLFormer-XL-both-10pct')
p.add_argument('--tf-pooling', default='cls', choices=['cls', 'mean'])
p.add_argument('--tfocc-dir',  default='results_tf_occ')
p.add_argument('--limit', type=int, default=None, help='debug: fewer molecules')
args = p.parse_args()

os.makedirs(args.outdir, exist_ok=True)
LOG = open(os.path.join(args.outdir, 'ig_sensitivity_log.txt'), 'w')
def say(m=''):
    print(m, flush=True); LOG.write(str(m) + '\n'); LOG.flush()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
say(f'device: {DEV}')
if DEV == 'cpu':
    say('WARNING: no GPU detected. This will be slow. '
        'Runtime > Change runtime type > T4 GPU.')

# ----------------------------------------------------------------------
# model, loaded exactly as run_08_xai.py / run_11 do
# ----------------------------------------------------------------------
class Regressor(nn.Module):
    def __init__(self, ckpt, pooling, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
        self.pooling = pooling
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, 256),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))
    def from_embeds(self, embeds, mask):
        h = self.encoder(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        if self.pooling == 'mean':
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            pooled = h[:, 0]
        return self.head(pooled).squeeze(-1)
    def forward(self, ids, mask):
        return self.from_embeds(self.encoder.get_input_embeddings()(ids), mask)

if not os.path.exists(args.tf_ckpt):
    raise SystemExit(f'checkpoint not found: {args.tf_ckpt}\n'
                     f'look for it under results_tf/models/')

tok   = AutoTokenizer.from_pretrained(args.tf_name, trust_remote_code=True)
model = Regressor(args.tf_name, args.tf_pooling).to(DEV)
model.load_state_dict(torch.load(args.tf_ckpt, map_location=DEV))
model.eval()
emb_layer = model.encoder.get_input_embeddings()
say(f'loaded {args.tf_name} from {args.tf_ckpt} (pooling: {args.tf_pooling})')

# ----------------------------------------------------------------------
# data: the SAME matched set Table 1 uses
# ----------------------------------------------------------------------
df = pd.read_csv(args.data)
test = df[df.split == 'test'].reset_index(drop=True)
MOLS = [Chem.MolFromSmiles(s) for s in test.smiles.values]

QUINAZOLINE = Chem.MolFromSmarts('n1cnc2ccccc2c1')
WARHEAD     = Chem.MolFromSmarts('[CX3]=[CX3][CX3](=O)[NX3]')

def narrow_atoms(mol):
    out = set()
    for m in mol.GetSubstructMatches(QUINAZOLINE):
        out |= {a for a in m if mol.GetAtomWithIdx(a).GetSymbol() == 'N'}
    for m in mol.GetSubstructMatches(WARHEAD):
        out |= {m[i] for i in (0, 1)}
    return sorted(out)

NARROW = [narrow_atoms(m) for m in MOLS]
BROAD  = [set(json.loads(c)) for c in test.core_atoms.values]
NAT    = [m.GetNumAtoms() for m in MOLS]

matched = [i for i in range(len(MOLS)) if NARROW[i] and BROAD[i]]
if args.limit:
    matched = matched[:args.limit]
say(f'matched compounds (broad AND narrow motif): n = {len(matched)}')

def enrichment(a, target, n):
    a = np.asarray(a, dtype=float); tot = a.sum()
    t = [i for i in target if i < len(a)]
    if tot <= 0 or not t or n == 0: return np.nan
    return (a[t].sum() / tot) / (len(t) / n)

# ----------------------------------------------------------------------
# token <-> atom alignment, identical to run_08_xai.py
# ----------------------------------------------------------------------
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

def encode(mol):
    smi, spans = atom_char_spans(mol)
    enc = tok(smi, return_tensors='pt', return_offsets_mapping=True,
              truncation=True, max_length=256)
    offsets = enc.pop('offset_mapping')[0].tolist()
    return (enc['input_ids'].to(DEV), enc['attention_mask'].to(DEV),
            offsets, spans, mol.GetNumAtoms())

def tokens_to_atoms(tokvals, offsets, spans, natoms):
    a = np.zeros(natoms)
    for t, (s, e) in enumerate(offsets):
        if e <= s or t >= len(tokvals):
            continue
        hits = [ai for (st, en, ai) in spans if st < e and en > s]
        for ai in hits:
            a[ai] += tokvals[t] / len(hits)
    return a

# ----------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------
PAD_ID  = tok.pad_token_id
MASK_ID = tok.mask_token_id if tok.mask_token_id is not None else PAD_ID
with torch.no_grad():
    W = emb_layer.weight.detach()
    AVG_EMB = W.mean(0, keepdim=True)        # centroid of the embedding table
say(f'baselines: zero | pad(id {PAD_ID}) | mask(id {MASK_ID}) | average-embedding')

def make_baseline(kind, emb, ids):
    if kind == 'zero':
        return torch.zeros_like(emb)
    if kind == 'pad':
        return emb_layer(torch.full_like(ids, PAD_ID)).detach()
    if kind == 'mask':
        return emb_layer(torch.full_like(ids, MASK_ID)).detach()
    if kind == 'average':
        return AVG_EMB.expand_as(emb).clone()
    raise ValueError(kind)

def integrated_gradients(emb, mask, base, steps):
    """Riemann right-sum, exactly as run_08_xai.py, returning per-token
    attribution and the completeness residual."""
    total = torch.zeros_like(emb)
    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        x = (base + float(alpha) * (emb - base)).requires_grad_(True)
        pred = model.from_embeds(x, mask)
        total += torch.autograd.grad(pred, x)[0]
    attr_full = (emb - base) * total / steps                 # per hidden unit
    tokvals = attr_full.sum(-1).abs()[0].detach().cpu().numpy()
    with torch.no_grad():
        fx = model.from_embeds(emb, mask).item()
        fb = model.from_embeds(base, mask).item()
    completeness = attr_full.sum().item()                    # signed, not abs
    return tokvals, fx, fb, completeness

# ----------------------------------------------------------------------
# (C) sensitivity sweep
# ----------------------------------------------------------------------
say('\n' + '=' * 74)
say('C  Integrated gradients: baseline and step-count sensitivity')
say('=' * 74)

SETTINGS = ([('zero', s) for s in (32, 64, 128, 256)] +
            [(b, 64) for b in ('pad', 'mask', 'average')])

rows, comp_rows = [], []
for kind, steps in SETTINGS:
    t0 = time.time()
    enr_n, enr_b, rel_err = [], [], []
    for c, i in enumerate(matched):
        ids, mask, offsets, spans, natoms = encode(MOLS[i])
        emb  = emb_layer(ids).detach()
        base = make_baseline(kind, emb, ids)
        tokvals, fx, fb, comp = integrated_gradients(emb, mask, base, steps)
        a = tokens_to_atoms(tokvals, offsets, spans, natoms)
        enr_n.append(enrichment(a, NARROW[i], NAT[i]))
        enr_b.append(enrichment(a, BROAD[i],  NAT[i]))
        denom = fx - fb
        rel_err.append(abs(comp - denom) / max(abs(denom), 1e-6))
        if (c + 1) % 100 == 0:
            say(f'   [{kind} {steps}] {c+1}/{len(matched)}')
    enr_n = np.array(enr_n, dtype=float); enr_b = np.array(enr_b, dtype=float)
    keep = ~(np.isnan(enr_n) | np.isnan(enr_b))
    from scipy import stats
    pw = stats.wilcoxon(enr_n[keep] - 1.0).pvalue if keep.sum() > 10 else np.nan
    # Two different quantities, kept apart on purpose.
    #   selectivity_paired      = mean of the within-molecule ratios. This is
    #                             the definition used in run_13_revision.py and
    #                             reported in Table 1 of the paper.
    #   selectivity_ratio_of_means = ratio of the two means. Earlier versions of
    #                             this script reported only this one, under the
    #                             bare name "selectivity", which invited a
    #                             comparison with Table 1 that is not valid.
    #                             It is kept so published numbers stay traceable.
    _ratio = enr_n[keep] / np.where(enr_b[keep] == 0, np.nan, enr_b[keep])
    row = dict(baseline=kind, steps=steps, n=int(keep.sum()),
               narrow=round(float(enr_n[keep].mean()), 3),
               broad=round(float(enr_b[keep].mean()), 3),
               selectivity_paired=round(float(np.nanmean(_ratio)), 3),
               selectivity_ratio_of_means=round(
                   float(enr_n[keep].mean() / enr_b[keep].mean()), 3),
               p_narrow_vs_chance=float(f'{pw:.2g}'),
               median_completeness_error=round(float(np.median(rel_err)), 3),
               seconds=int(time.time() - t0))
    rows.append(row)
    comp_rows.append(dict(baseline=kind, steps=steps,
                          median_rel_error=round(float(np.median(rel_err)), 4),
                          p90_rel_error=round(float(np.percentile(rel_err, 90)), 4)))
    say(f'  {kind:8s} steps={steps:4d}  narrow={row["narrow"]:.3f}  '
        f'broad={row["broad"]:.3f}  '
        f'sel_paired={row["selectivity_paired"]:.3f}  '
        f'sel_ratio_of_means={row["selectivity_ratio_of_means"]:.3f}  '
        f'p={row["p_narrow_vs_chance"]:.3g}  '
        f'completeness err={row["median_completeness_error"]:.3f}  '
        f'({row["seconds"]}s)')

sens = pd.DataFrame(rows)
sens.to_csv(os.path.join(args.outdir, 'ig_sensitivity.csv'), index=False)
pd.DataFrame(comp_rows).to_csv(os.path.join(args.outdir, 'ig_completeness.csv'),
                               index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
say('\n' + sens.to_string(index=False))
say('\nHow to read this:')
say('  narrow stays ~1.0 everywhere      -> the negative finding is robust;')
say('                                       it is the model, not the estimator.')
say('  narrow rises with a better baseline -> the finding was about the zero')
say('                                       baseline, and the paper must say so.')
say('  completeness error large (>0.2)   -> the integral has not converged.')

# ----------------------------------------------------------------------
# (D) faithfulness by ablation, for MoLFormer
# ----------------------------------------------------------------------
say('\n' + '=' * 74)
say('D  MoLFormer faithfulness by token ablation')
say('=' * 74)

occ_path = os.path.join(args.tfocc_dir, 'tf_occlusion_attributions.npz')
ATTRSETS = {}
if os.path.exists(occ_path):
    z = np.load(occ_path, allow_pickle=True)
    smi2i = {s: k for k, s in enumerate(list(z['smiles']))}
    arr = list(z['Transformer_occlusion'])
    ATTRSETS['token occlusion'] = {i: np.abs(np.asarray(
        arr[smi2i[test.smiles.values[i]]], dtype=float))
        for i in matched if test.smiles.values[i] in smi2i}
else:
    say(f'[skip] {occ_path} not found')

# integrated gradients at the original setting, recomputed here
ig_attr = {}
for i in matched:
    ids, mask, offsets, spans, natoms = encode(MOLS[i])
    emb = emb_layer(ids).detach()
    tokvals, *_ = integrated_gradients(emb, mask, torch.zeros_like(emb), 32)
    ig_attr[i] = tokens_to_atoms(tokvals, offsets, spans, natoms)
ATTRSETS['integrated gradients'] = ig_attr

rng = np.random.RandomState(42)
frows = []
for label, attrs in ATTRSETS.items():
    for k in (10, 20, 30):
        d_top, d_rnd = [], []
        for i in attrs:
            ids, mask, offsets, spans, natoms = encode(MOLS[i])
            a = attrs[i]
            if len(a) != natoms or natoms < 4:
                continue
            ntop = max(1, int(round(k / 100 * natoms)))
            top  = set(np.argsort(-a)[:ntop].tolist())
            rnd  = set(rng.choice(natoms, ntop, replace=False).tolist())

            def masked_pred(atomset):
                b = ids.clone()
                for t, (s, e) in enumerate(offsets):
                    if e <= s:
                        continue
                    hits = [ai for (st, en, ai) in spans if st < e and en > s]
                    if hits and any(h in atomset for h in hits):
                        b[0, t] = MASK_ID
                with torch.no_grad():
                    return model(b, mask).item()

            with torch.no_grad():
                base = model(ids, mask).item()
            d_top.append(abs(base - masked_pred(top)))
            d_rnd.append(abs(base - masked_pred(rnd)))
        if not d_top:
            continue
        mt, mr = float(np.mean(d_top)), float(np.mean(d_rnd))
        frows.append(dict(attribution=label, k_percent=k, n=len(d_top),
                          delta_top=round(mt, 3), delta_random=round(mr, 3),
                          ratio=round(mt / mr, 2) if mr > 0 else np.nan))
        say(f'  {label:22s} k={k:2d}%  top {mt:.3f}  random {mr:.3f}  '
            f'ratio {mt/max(mr,1e-9):.2f}')

if frows:
    pd.DataFrame(frows).to_csv(os.path.join(args.outdir, 'tf_faithfulness.csv'),
                               index=False)
    say('\nA ratio above 1 means the attribution identifies inputs the model')
    say('actually uses. Compare with AttentiveFP: 2.56, 2.13, 1.79.')

say('\ndone')
LOG.close()
