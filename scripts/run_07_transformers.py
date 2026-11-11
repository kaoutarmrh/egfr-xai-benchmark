#!/usr/bin/env python
"""
run_07_transformers.py — compare chemical language models on EGFR pIC50.

    pip install torch transformers rdkit pandas scikit-learn matplotlib sentencepiece
    python run_07_transformers.py --data egfr_final.csv --outdir results_tf

Each checkpoint is trained, evaluated and SAVED before the next one starts, so a
crash or an out-of-memory on checkpoint 3 keeps checkpoints 1 and 2. Rerunning
skips checkpoints already present in results.csv unless --force is given.

Useful flags:
    --only ChemBERTa-77M-MTR MoLFormer-XL      train just these
    --batch-size 16                            if the GPU is small
    --force                                    retrain everything
"""

import os, gc, traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

from common import Ctx, parse_args

CHECKPOINTS = {
    'ChemBERTa-77M-MTR':   'DeepChem/ChemBERTa-77M-MTR',
    'ChemBERTa-77M-MLM':   'DeepChem/ChemBERTa-77M-MLM',
    'ChemBERTa-zinc-base': 'seyonec/ChemBERTa-zinc-base-v1',
    'MoLFormer-XL':        'ibm/MoLFormer-XL-both-10pct',
}

args = parse_args(lambda p: (
    p.add_argument('--only', nargs='*', default=None),
    p.add_argument('--force', action='store_true'),
    p.add_argument('--lr-body', type=float, default=3e-5),
    p.add_argument('--lr-head', type=float, default=1e-3),
    p.add_argument('--patience', type=int, default=5),
    p.add_argument('--skip-pooling-test', action='store_true'),
))
args.epochs = args.epochs or 25
args.batch_size = args.batch_size or 32

ctx = Ctx(args)
torch.manual_seed(args.seed); np.random.seed(args.seed)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ctx.log(f'device: {DEVICE}')
if DEVICE == 'cuda':
    ctx.log(f'gpu: {torch.cuda.get_device_name(0)}  '
            f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

train, val, test = ctx.splits()

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


class Regressor(nn.Module):
    """Pre-trained encoder + regression head.

    pooling='mean' averages every real token, so all atoms contribute.
    pooling='cls'  reads only the first token, which is the library default.
    """
    def __init__(self, checkpoint, pooling='mean', dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        self.pooling = pooling
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 256),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        if self.pooling == 'mean':
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            pooled = h[:, 0]
        return self.head(pooled).squeeze(-1)


def loaders(tok, maxlen):
    def one(subset, shuffle):
        enc = tok(list(subset.smiles.values), padding='max_length',
                  truncation=True, max_length=maxlen, return_tensors='pt')
        ds = TensorDataset(enc['input_ids'], enc['attention_mask'],
                           torch.tensor(ctx.norm(subset.pIC50.values), dtype=torch.float))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)
    return one(train, True), one(val, False), one(test, False)


def train_one(label, checkpoint, pooling='mean'):
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    lengths = [len(tok.tokenize(s)) for s in train.smiles.sample(300, random_state=0)]
    maxlen = min(256, int(np.percentile(lengths, 99)) + 4)
    ctx.log(f'  tokenizer maxlen {maxlen}')

    model = Regressor(checkpoint, pooling).to(DEVICE)
    tr_dl, va_dl, te_dl = loaders(tok, maxlen)

    head = [p for n, p in model.named_parameters() if n.startswith('head')]
    body = [p for n, p in model.named_parameters() if not n.startswith('head')]
    opt = torch.optim.AdamW([{'params': body, 'lr': args.lr_body},
                             {'params': head, 'lr': args.lr_head}], weight_decay=0.01)
    steps = len(tr_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = nn.MSELoss()

    def epoch(dl, training):
        model.train() if training else model.eval()
        tot = 0.0; P = []; T = []
        with torch.set_grad_enabled(training):
            for ids, mask, yb in dl:
                ids, mask, yb = ids.to(DEVICE), mask.to(DEVICE), yb.to(DEVICE)
                out = model(ids, mask); loss = lossf(out, yb)
                if training:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step()
                tot += loss.item() * len(yb)
                P.append(out.detach().cpu()); T.append(yb.cpu())
        return tot/len(dl.dataset), torch.cat(P).numpy(), torch.cat(T).numpy()

    safe = ''.join(c if c.isalnum() else '_' for c in label.lower())
    ckpt_path = ctx.path('models', f'{safe}.pt')
    history, best, wait = [], 1e9, 0

    for e in range(1, args.epochs + 1):
        tr, _, _   = epoch(tr_dl, True)
        va, vp, vt = epoch(va_dl, False)
        r2 = r2_score(vt, vp)
        history.append({'epoch': e, 'train': tr, 'val': va, 'r2': r2})
        if va < best - 1e-4:
            best, wait = va, 0
            torch.save(model.state_dict(), ckpt_path)
            mark = '  *saved*'
        else:
            wait += 1; mark = ''
        ctx.log(f'  epoch {e:2d}  train {tr:.4f}  val {va:.4f}  R2(val) {r2:.4f}{mark}')
        if wait >= args.patience:
            ctx.log('  early stop'); break

    model.load_state_dict(torch.load(ckpt_path))
    _, zp, zt = epoch(te_dl, False)
    y_pred, y_true = ctx.denorm(zp), ctx.denorm(zt)

    ctx.record(label, 'SMILES tokens', y_true, y_pred)
    ctx.scatter(label, y_true, y_pred)
    ctx.curve(label, history)
    if 'has_pharmacophore' in test.columns and len(y_true) == len(test):
        ctx.subgroup(label, y_true, y_pred, test.has_pharmacophore.values.astype(bool))

    del model, opt, tr_dl, va_dl, te_dl
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return y_true, y_pred


# ----------------------------------------------------------------------
wanted = args.only or list(CHECKPOINTS)
scores = {}

for name in wanted:
    if name not in CHECKPOINTS:
        ctx.log(f'unknown checkpoint name: {name}'); continue
    label = f'{name} (mean-pool)'
    if ctx.already_done(label) and not args.force:
        ctx.log(f'skipping {label} (already in results.csv; use --force to redo)')
        continue
    ctx.log(f'\n=== {name} -> {CHECKPOINTS[name]} ===')
    try:
        yt, yp = train_one(label, CHECKPOINTS[name], 'mean')
        scores[name] = r2_score(yt, yp)
    except torch.cuda.OutOfMemoryError:
        ctx.log(f'  OUT OF MEMORY - retry with --batch-size 16 or 8')
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        ctx.log(f'  FAILED: {type(e).__name__}: {str(e)[:200]}')
        ctx.log(traceback.format_exc()[-800:])

# ----------------------------------------------------------------------
# pooling comparison on the winner
# ----------------------------------------------------------------------
if scores and not args.skip_pooling_test:
    best_name = max(scores, key=scores.get)
    label = f'{best_name} (CLS-pool)'
    ctx.log(f'\n=== pooling test: {best_name} with CLS pooling ===')
    if ctx.already_done(label) and not args.force:
        ctx.log('  already done')
    else:
        try:
            yt, yp = train_one(label, CHECKPOINTS[best_name], 'cls')
            ctx.log(f'\nmean-pool advantage: {scores[best_name]-r2_score(yt,yp):+.4f} R2')
        except Exception as e:
            ctx.log(f'  FAILED: {type(e).__name__}: {str(e)[:200]}')

ctx.summary()
ctx.log('done')
