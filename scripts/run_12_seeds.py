#!/usr/bin/env python
"""
run_12_seeds.py — error bars for the results table.

WHY THIS EXISTS
Every number in the paper comes from seed 42. Random Forest 0.515 against
XGBoost 0.489 - is that a real difference or run-to-run noise? A referee will
ask, and right now there is no answer.

This retrains the headline models under several seeds and reports mean +/- sd,
so the paper can state which differences are real.

    python run_12_seeds.py --data egfr_final.csv --outdir results_seeds --seeds 5
    python run_12_seeds.py --data egfr_final.csv --outdir results_seeds \
        --models rf xgb gnn --seeds 5

Note: the SPLIT never changes. Only model initialisation and training
randomness vary, which is the right thing to measure - the split is a fixed
property of the benchmark, not a source of noise to average away.
"""

import os, json
import numpy as np
import pandas as pd
from scipy import stats
from common import Ctx, parse_args

args = parse_args(lambda p: (
    p.add_argument('--seeds', type=int, default=5),
    p.add_argument('--models', nargs='*', default=['rf', 'xgb', 'gnn']),
    p.add_argument('--gnn-epochs', type=int, default=120),
))
ctx = Ctx(args)
train, val, test = ctx.splits()
SEEDS = list(range(42, 42 + args.seeds))
ctx.log(f'seeds: {SEEDS}')

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_squared_error, r2_score
RDLogger.DisableLog('rdApp.*')

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp_matrix(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            X[i] = np.array(gen.GetFingerprint(m))
    return X


Xtr, ytr = fp_matrix(train.smiles.values), train.pIC50.values
Xva, yva = fp_matrix(val.smiles.values), val.pIC50.values
Xte, yte = fp_matrix(test.smiles.values), test.pIC50.values

RUNS = []


def note(model, seed, y_true, y_pred):
    r = {'model': model, 'seed': seed,
         'RMSE': float(mean_squared_error(y_true, y_pred) ** 0.5),
         'R2': float(r2_score(y_true, y_pred))}
    RUNS.append(r)
    ctx.log(f'  {model:14s} seed {seed}   RMSE {r["RMSE"]:.4f}   R2 {r["R2"]:.4f}')
    pd.DataFrame(RUNS).to_csv(ctx.path('seed_runs.csv'), index=False)
    return r


# ----------------------------------------------------------------------
if 'rf' in args.models:
    from sklearn.ensemble import RandomForestRegressor
    ctx.log('\n=== Random Forest ===')
    for s in SEEDS:
        m = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                  n_jobs=-1, random_state=s)
        m.fit(Xtr, ytr)
        note('Random Forest', s, yte, m.predict(Xte))

if 'xgb' in args.models:
    from xgboost import XGBRegressor
    ctx.log('\n=== XGBoost ===')
    for s in SEEDS:
        m = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=8,
                         subsample=0.8, colsample_bytree=0.8,
                         early_stopping_rounds=50, random_state=s, n_jobs=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        note('XGBoost', s, yte, m.predict(Xte))

if 'gnn' in args.models:
    ctx.log('\n=== AttentiveFP ===')
    import torch, torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
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
        v = [0]*len(al); v[al.index(x) if x in al else len(al)-1] = 1; return v

    def af(a):
        return (oh(a.GetSymbol(),ATOMS)+oh(a.GetDegree(),[0,1,2,3,4,5])
              + oh(a.GetFormalCharge(),[-2,-1,0,1,2])+oh(a.GetTotalNumHs(),[0,1,2,3,4])
              + oh(a.GetHybridization(),HYB)+[int(a.GetIsAromatic()),int(a.IsInRing())])

    def bf(b):
        return oh(b.GetBondType(),BONDS)+[int(b.GetIsConjugated()),int(b.IsInRing())]

    def build(sub):
        out = []
        for s, y in zip(sub.smiles.values, ctx.norm(sub.pIC50.values)):
            mol = Chem.MolFromSmiles(s)
            if mol is None or mol.GetNumAtoms() == 0:
                continue
            x = torch.tensor([af(a) for a in mol.GetAtoms()], dtype=torch.float)
            src, dst, ea = [], [], []
            for b in mol.GetBonds():
                p, q = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                f = bf(b); src += [p,q]; dst += [q,p]; ea += [f,f]
            ei = (torch.tensor([src,dst],dtype=torch.long) if src
                  else torch.zeros((2,0),dtype=torch.long))
            eat = (torch.tensor(ea,dtype=torch.float) if ea
                   else torch.zeros((0,BOND_DIM),dtype=torch.float))
            d = Data(x=x, edge_index=ei, edge_attr=eat,
                     y=torch.tensor([float(y)], dtype=torch.float))
            out.append(d)
        return out

    tr_ds, va_ds, te_ds = build(train), build(val), build(test)

    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        model = AttentiveFP(ATOM_DIM, 256, 1, BOND_DIM, 3, 2, 0.2).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=256)
        te_dl = DataLoader(te_ds, batch_size=256)
        lossf = nn.MSELoss()

        def run(dl, opt=None):
            training = opt is not None
            model.train() if training else model.eval()
            tot = 0.0; P = []; T = []
            with torch.set_grad_enabled(training):
                for b in dl:
                    b = b.to(DEV)
                    o = model(b.x, b.edge_index, b.edge_attr, b.batch).squeeze(-1)
                    l = lossf(o, b.y)
                    if training:
                        opt.zero_grad(); l.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    tot += l.item()*b.num_graphs
                    P.append(o.detach().cpu()); T.append(b.y.cpu())
            return tot/len(dl.dataset), torch.cat(P).numpy(), torch.cat(T).numpy()

        best, wait, state = 1e9, 0, None
        for e in range(1, args.gnn_epochs+1):
            run(tr_dl, opt)
            va, _, _ = run(va_dl)
            sched.step(va)
            if va < best - 1e-4:
                best, wait = va, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 20:
                    break
        model.load_state_dict(state)
        _, zp, zt = run(te_dl)
        note('AttentiveFP', s, ctx.denorm(zt), ctx.denorm(zp))

# ----------------------------------------------------------------------
runs = pd.DataFrame(RUNS)
summary = runs.groupby('model').agg(
    R2_mean=('R2', 'mean'), R2_sd=('R2', 'std'),
    RMSE_mean=('RMSE', 'mean'), RMSE_sd=('RMSE', 'std'),
    n=('R2', 'size')).reset_index().sort_values('R2_mean', ascending=False)
summary['R2'] = summary.apply(
    lambda r: f"{r.R2_mean:.3f} +/- {r.R2_sd:.3f}", axis=1)
summary['RMSE'] = summary.apply(
    lambda r: f"{r.RMSE_mean:.3f} +/- {r.RMSE_sd:.3f}", axis=1)
summary.to_csv(ctx.path('seed_summary.csv'), index=False)

ctx.log('\n=== summary (use these in the paper) ===')
ctx.log(summary[['model', 'R2', 'RMSE', 'n']].to_string(index=False))

# which differences survive the noise?
ctx.log('\n=== are the differences real? ===')
names = list(summary.model)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a = runs[runs.model == names[i]].R2.values
        b = runs[runs.model == names[j]].R2.values
        if len(a) > 1 and len(b) > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            verdict = 'REAL' if p < 0.05 else 'within noise'
            ctx.log(f'  {names[i]:16s} vs {names[j]:16s}  '
                    f'diff {a.mean()-b.mean():+.4f}  p={p:.3g}  {verdict}')

ctx.log('\nReport any "within noise" pair as comparable rather than ranked.')
ctx.log('done')
