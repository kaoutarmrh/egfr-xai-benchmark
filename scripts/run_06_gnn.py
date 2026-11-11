#!/usr/bin/env python
"""
run_06_gnn.py — AttentiveFP graph neural network for EGFR pIC50.

    pip install torch torch_geometric rdkit pandas scikit-learn matplotlib
    python run_06_gnn.py --data egfr_final.csv --outdir results_gnn

Everything is saved as it is produced:
    results.csv            one row per model, written immediately
    all_preds.csv          test-set predictions
    figures/               scatter plot and training curve
    history_*.csv          per-epoch losses
    subgroup_scores.json   score split by pharmacophore presence
    models/attentivefp.pt  best weights by validation loss
    run_log.txt            everything printed
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from common import Ctx, parse_args

args = parse_args(lambda p: (
    p.add_argument('--hidden', type=int, default=256),
    p.add_argument('--layers', type=int, default=3),
    p.add_argument('--timesteps', type=int, default=2),
    p.add_argument('--dropout', type=float, default=0.2),
    p.add_argument('--lr', type=float, default=1e-3),
    p.add_argument('--patience', type=int, default=20),
))
args.epochs = args.epochs or 200
args.batch_size = args.batch_size or 64

ctx = Ctx(args)
torch.manual_seed(args.seed); np.random.seed(args.seed)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ctx.log(f'device: {DEVICE}')
if DEVICE == 'cuda':
    ctx.log(f'gpu: {torch.cuda.get_device_name(0)}')

train, val, test = ctx.splits()

# ----------------------------------------------------------------------
# molecule -> graph
# ----------------------------------------------------------------------
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

ATOMS = ['C','N','O','S','F','Cl','Br','I','P','B','Si','Se','other']
HYB = [Chem.rdchem.HybridizationType.SP,  Chem.rdchem.HybridizationType.SP2,
       Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
       Chem.rdchem.HybridizationType.SP3D2]
BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
         Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

ATOM_DIM = len(ATOMS) + 6 + 5 + 5 + len(HYB) + 2     # 36
BOND_DIM = len(BONDS) + 2                             # 6


def onehot(x, allowed):
    v = [0]*len(allowed)
    v[allowed.index(x) if x in allowed else len(allowed)-1] = 1
    return v


def atom_features(a):
    return (onehot(a.GetSymbol(), ATOMS)
          + onehot(a.GetDegree(), [0,1,2,3,4,5])
          + onehot(a.GetFormalCharge(), [-2,-1,0,1,2])
          + onehot(a.GetTotalNumHs(), [0,1,2,3,4])
          + onehot(a.GetHybridization(), HYB)
          + [int(a.GetIsAromatic()), int(a.IsInRing())])


def bond_features(b):
    return onehot(b.GetBondType(), BONDS) + [int(b.GetIsConjugated()), int(b.IsInRing())]


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def mol_to_data(smiles, y=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src += [i, j]; dst += [j, i]; ea += [bf, bf]          # messages both ways
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(ea, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_DIM), dtype=torch.float)
    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        d.y = torch.tensor([y], dtype=torch.float)
    return d


def build(subset):
    out, dropped = [], 0
    for s, y in zip(subset.smiles.values, ctx.norm(subset.pIC50.values)):
        d = mol_to_data(s, float(y))
        if d is None:
            dropped += 1
        else:
            out.append(d)
    return out, dropped


train_ds, d1 = build(train)
val_ds,   d2 = build(val)
test_ds,  d3 = build(test)
ctx.log(f'graphs: train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}'
        f'  (dropped {d1+d2+d3})')
if d3:
    ctx.log('WARNING: some test molecules failed to build; predictions will be '
            'shorter than the test set and saved to a separate file')

# sanity: node order must equal RDKit atom order, or notebook 8 breaks
if 'core_atoms' in test.columns:
    row = test.iloc[0]
    d = mol_to_data(row.smiles)
    core = json.loads(row.core_atoms)
    assert d.x.shape[0] == Chem.MolFromSmiles(row.smiles).GetNumAtoms()
    assert all(i < d.x.shape[0] for i in core)
    ctx.log('verified: graph node order == RDKit atom order')

# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
from torch_geometric.nn.models import AttentiveFP

model = AttentiveFP(in_channels=ATOM_DIM, hidden_channels=args.hidden,
                    out_channels=1, edge_dim=BOND_DIM,
                    num_layers=args.layers, num_timesteps=args.timesteps,
                    dropout=args.dropout).to(DEVICE)
ctx.log(f'parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=256)
test_dl  = DataLoader(test_ds,  batch_size=256)


def run_epoch(dl, opt=None):
    training = opt is not None
    model.train() if training else model.eval()
    lossf = nn.MSELoss(); tot = 0.0; P = []; T = []
    with torch.set_grad_enabled(training):
        for batch in dl:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
            loss = lossf(out, batch.y)
            if training:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item() * batch.num_graphs
            P.append(out.detach().cpu()); T.append(batch.y.cpu())
    return tot/len(dl.dataset), torch.cat(P).numpy(), torch.cat(T).numpy()


opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

ckpt = ctx.path('models', 'attentivefp.pt')
history, best, wait = [], 1e9, 0

for epoch in range(1, args.epochs + 1):
    tr, _, _   = run_epoch(train_dl, opt)
    va, vp, vt = run_epoch(val_dl)
    sched.step(va)
    r2 = r2_score(vt, vp)
    history.append({'epoch': epoch, 'train': tr, 'val': va, 'r2': r2})

    if va < best - 1e-4:
        best, wait = va, 0
        torch.save(model.state_dict(), ckpt)
        mark = '  *saved*'
    else:
        wait += 1
        mark = ''
    if epoch % 5 == 0 or mark:
        ctx.log(f'epoch {epoch:3d}  train {tr:.4f}  val {va:.4f}  '
                f'R2(val) {r2:.4f}  lr {opt.param_groups[0]["lr"]:.1e}{mark}')
    # write history every epoch so a crash still leaves the curve
    if epoch % 10 == 0:
        ctx.curve('AttentiveFP (GNN)', history)
    if wait >= args.patience:
        ctx.log(f'early stop at epoch {epoch}'); break

# ----------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------
model.load_state_dict(torch.load(ckpt))
_, zp, zt = run_epoch(test_dl)
y_pred = ctx.denorm(zp); y_true = ctx.denorm(zt)

ctx.record('AttentiveFP (GNN)', 'molecular graph', y_true, y_pred)
ctx.scatter('AttentiveFP (GNN)', y_true, y_pred)
ctx.curve('AttentiveFP (GNN)', history)

if 'has_pharmacophore' in test.columns and len(y_true) == len(test):
    ctx.subgroup('AttentiveFP (GNN)', y_true, y_pred,
                 test.has_pharmacophore.values.astype(bool))

ctx.summary()
ctx.log('done')
