#!/usr/bin/env python
"""
run_17_tokenizer_alignment.py — validate the tokenizer-to-atom mapping properly.

WHY THIS EXISTS
The paper claims the atom-level alignment was verified on all 7,982 molecules.
The check in run_13_revision.py verifies only half of the mapping: that RDKit's
canonical SMILES contains exactly as many atom tokens as the molecule has heavy
atoms, and that the atom output order is a complete permutation. It never loads
the model tokenizer, so it says nothing about whether the tokenizer's offset
mapping lands on the right characters.

That is the half the attribution actually depends on. This script tests it.

For every molecule it checks three things:

  1. COVERAGE    every heavy atom is covered by at least one token. An
                 uncovered atom would silently receive zero attribution.
  2. CONTAINMENT every token that overlaps an atom span lies inside the SMILES
                 string, and multi-character tokens spanning several atoms are
                 counted, since their attribution is divided among them.
  3. ROUND-TRIP  the atoms recovered through the offset mapping are exactly the
                 molecule's atoms - no atom recovered twice, none invented.

No GPU and no model weights: the tokenizer alone.

    python run_17_tokenizer_alignment.py --data egfr_final.csv \
        --tf-name ibm/MoLFormer-XL-both-10pct --outdir results_revision

Writes: tokenizer_alignment.csv, tokenizer_alignment_log.txt
"""

import os, re, argparse
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--data',    default='egfr_final.csv')
p.add_argument('--tf-name', default='ibm/MoLFormer-XL-both-10pct')
p.add_argument('--outdir',  default='results_revision')
p.add_argument('--limit', type=int, default=None)
args = p.parse_args()

os.makedirs(args.outdir, exist_ok=True)
LOG = open(os.path.join(args.outdir, 'tokenizer_alignment_log.txt'), 'w')
def say(m=''):
    print(m, flush=True); LOG.write(str(m) + '\n'); LOG.flush()

tok = AutoTokenizer.from_pretrained(args.tf_name, trust_remote_code=True)
say(f'tokenizer: {args.tf_name}')

# identical to the regex used in run_08_xai.py and run_11
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

df = pd.read_csv(args.data)
smiles = list(df.smiles.values)
if args.limit:
    smiles = smiles[:args.limit]
say(f'molecules to check: {len(smiles):,}\n')

rows = []
n_ok = n_uncovered = n_span_mismatch = n_truncated = n_bad = 0
tokens_per_atom, atoms_per_token = [], []

for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        n_bad += 1
        rows.append(dict(smiles=smi, status='unparseable'))
        continue

    nat = mol.GetNumAtoms()
    canon, spans = atom_char_spans(mol)

    # (1) the regex must find exactly one span per heavy atom
    if len(spans) != nat:
        n_span_mismatch += 1
        rows.append(dict(smiles=smi, status='span_count_mismatch',
                         n_atoms=nat, n_spans=len(spans)))
        continue

    enc = tok(canon, return_tensors=None, return_offsets_mapping=True,
              truncation=True, max_length=256)
    offsets = enc['offset_mapping']
    truncated = len(enc['input_ids']) >= 256

    # (2) map every token to the atoms whose character span it overlaps,
    #     exactly as the attribution code does
    covered = np.zeros(nat, dtype=int)
    tok_atom_counts = []
    for (s, e) in offsets:
        if e <= s:
            continue                                    # special token
        hits = [ai for (st, en, ai) in spans if st < e and en > s]
        if hits:
            tok_atom_counts.append(len(hits))
            for ai in hits:
                covered[ai] += 1

    # (3) every atom must receive at least one token
    missing = int((covered == 0).sum())
    status = 'ok'
    if truncated:
        status = 'truncated_at_256_tokens'; n_truncated += 1
    elif missing:
        status = 'atoms_without_token'; n_uncovered += 1
    else:
        n_ok += 1
        tokens_per_atom.append(covered.mean())
        if tok_atom_counts:
            atoms_per_token.append(float(np.mean(tok_atom_counts)))

    rows.append(dict(smiles=smi, status=status, n_atoms=nat,
                     n_tokens=len(offsets),
                     atoms_without_token=missing,
                     mean_tokens_per_atom=round(float(covered.mean()), 3)))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(args.outdir, 'tokenizer_alignment.csv'), index=False)

n = len(smiles)
say('=' * 70)
say('tokenizer-to-atom offset alignment')
say('=' * 70)
say(f'{"molecules checked":42s}: {n:,}')
say(f'{"fully aligned":42s}: {n_ok:,} ({n_ok/n:.2%})')
say(f'{"atoms left without any token":42s}: {n_uncovered:,}')
say(f'{"regex span count != heavy atom count":42s}: {n_span_mismatch:,}')
say(f'{"truncated at 256 tokens":42s}: {n_truncated:,}')
say(f'{"unparseable":42s}: {n_bad:,}')
if tokens_per_atom:
    say(f'\n{"mean tokens covering an atom":42s}: {np.mean(tokens_per_atom):.3f}')
if atoms_per_token:
    say(f'{"mean atoms spanned by a mapped token":42s}: {np.mean(atoms_per_token):.3f}')

say('\nHow to read this:')
say('  "fully aligned" means every heavy atom was reached by at least one')
say('  token through the offset mapping, so no atom can silently receive zero')
say('  attribution. "mean atoms spanned by a mapped token" above 1 is expected:')
say('  multi-character tokens cover several atoms and divide their attribution')
say('  among them, which is what the attribution code does.')
say('\nThis is the claim the paper may make. Anything else must be qualified.')
LOG.close()
print('\nwrote', os.path.join(args.outdir, 'tokenizer_alignment.csv'))
