"""
common.py — shared utilities for all server training scripts.

Key difference from the Colab notebooks: every result is written to disk the
moment it is produced. A crash at model 3 of 4 keeps models 1 and 2.

Usage inside a script:
    from common import Ctx
    ctx = Ctx()                      # reads --data / --outdir from argv
    train, val, test = ctx.splits()
    ...
    ctx.record('AttentiveFP', 'graph', y_true, y_pred)   # saves immediately
"""

import os, sys, json, argparse, datetime
import numpy as np
import pandas as pd


def parse_args(extra=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='egfr_final.csv', help='dataset csv')
    p.add_argument('--outdir', default='.', help='where results/figures/models go')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    if extra:
        extra(p)
    return p.parse_args()


class Ctx:
    """Holds paths, the dataset, the normalisation constants, and does the saving."""

    def __init__(self, args=None):
        self.args = args or parse_args()
        self.outdir = os.path.abspath(self.args.outdir)
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.path('figures'), exist_ok=True)
        os.makedirs(self.path('models'), exist_ok=True)

        self.results_csv = self.path('results.csv')
        self.preds_csv   = self.path('all_preds.csv')
        self.log_txt     = self.path('run_log.txt')

        self.df = pd.read_csv(self.args.data)
        need = {'smiles', 'pIC50', 'split'}
        missing = need - set(self.df.columns)
        if missing:
            raise SystemExit(f'dataset is missing columns: {missing}')

        tr = self.df[self.df.split == 'train']
        self.y_mean = float(tr.pIC50.mean())
        self.y_std  = float(tr.pIC50.std())

        self.log(f'--- run started {datetime.datetime.now():%Y-%m-%d %H:%M} ---')
        self.log(f'data {self.args.data}  outdir {self.outdir}')
        self.log(f'train {len(tr)}  val {(self.df.split=="val").sum()}  '
                 f'test {(self.df.split=="test").sum()}')
        self.log(f'target normalisation: mean {self.y_mean:.4f}  std {self.y_std:.4f}')

    # ---------------- paths & logging ----------------
    def path(self, *parts):
        return os.path.join(self.outdir, *parts)

    def log(self, msg):
        line = str(msg)
        print(line, flush=True)
        with open(self.log_txt, 'a') as f:
            f.write(line + '\n')

    # ---------------- data ----------------
    def splits(self):
        d = self.df
        return d[d.split == 'train'], d[d.split == 'val'], d[d.split == 'test']

    def norm(self, y):
        return (np.asarray(y, dtype=float) - self.y_mean) / self.y_std

    def denorm(self, z):
        return np.asarray(z, dtype=float) * self.y_std + self.y_mean

    # ---------------- saving ----------------
    def record(self, model, representation, y_true, y_pred, extra=None):
        """Append one row to results.csv IMMEDIATELY, and store predictions."""
        from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        row = {
            'model': model,
            'representation': representation,
            'RMSE': round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
            'MAE':  round(float(mean_absolute_error(y_true, y_pred)), 4),
            'R2':   round(float(r2_score(y_true, y_pred)), 4),
            'spread_ratio': round(float(y_pred.std() / y_true.std()), 3),
            'n_test': len(y_true),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        }
        if extra:
            row.update(extra)

        old = pd.read_csv(self.results_csv) if os.path.exists(self.results_csv) else pd.DataFrame()
        # replace an existing row with the same model name, so reruns update
        if len(old) and 'model' in old.columns:
            old = old[old.model != model]
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(
            self.results_csv, index=False)

        self.save_predictions(model, y_true, y_pred)
        self.log(f"[saved] {model:28s} RMSE {row['RMSE']:.4f}  R2 {row['R2']:.4f}  "
                 f"spread {row['spread_ratio']:.2f}")
        return row

    def save_predictions(self, model, y_true, y_pred):
        col = 'pred_' + ''.join(ch if ch.isalnum() else '_' for ch in model.lower())
        if os.path.exists(self.preds_csv):
            p = pd.read_csv(self.preds_csv)
            if len(p) != len(y_pred):
                self.log(f'  ! prediction length {len(y_pred)} != stored {len(p)}; '
                         f'writing {col} to a separate file')
                pd.DataFrame({'y_true': y_true, col: y_pred}).to_csv(
                    self.path(f'preds_{col}.csv'), index=False)
                return
        else:
            _, _, test = self.splits()
            p = pd.DataFrame({'smiles': test.smiles.values[:len(y_true)],
                              'y_true': y_true})
        p[col] = y_pred
        p.to_csv(self.preds_csv, index=False)

    def already_done(self, model):
        if not os.path.exists(self.results_csv):
            return False
        r = pd.read_csv(self.results_csv)
        return 'model' in r.columns and (r.model == model).any()

    # ---------------- figures ----------------
    def scatter(self, model, y_true, y_pred, fname=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score

        fname = fname or f"scatter_{''.join(c if c.isalnum() else '_' for c in model.lower())}.png"
        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        ax.scatter(y_true, y_pred, s=7, alpha=0.3)
        lo = min(y_true.min(), y_pred.min()) - 0.3
        hi = max(y_true.max(), y_pred.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('measured pIC50'); ax.set_ylabel('predicted pIC50')
        ax.set_title(f'{model}  (R2 = {r2_score(y_true, y_pred):.3f})')
        fig.tight_layout()
        out = self.path('figures', fname)
        fig.savefig(out, dpi=300); plt.close(fig)
        self.log(f'  figure -> {out}')
        return out

    def curve(self, model, history, fname=None):
        """history: list of dicts with keys epoch, train, val, r2"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        h = pd.DataFrame(history)
        h.to_csv(self.path(f"history_{''.join(c if c.isalnum() else '_' for c in model.lower())}.csv"),
                 index=False)
        fname = fname or f"curve_{''.join(c if c.isalnum() else '_' for c in model.lower())}.png"
        fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
        ax1.plot(h.epoch, h.train, label='train loss')
        ax1.plot(h.epoch, h.val, label='val loss')
        ax1.set_xlabel('epoch'); ax1.set_ylabel('MSE (normalised)')
        ax2 = ax1.twinx(); ax2.plot(h.epoch, h.r2, ls=':', color='tab:green', label='val R2')
        ax2.set_ylabel('val R2')
        ax1.legend(loc='upper right'); ax1.set_title(model)
        fig.tight_layout()
        out = self.path('figures', fname)
        fig.savefig(out, dpi=300); plt.close(fig)
        self.log(f'  figure -> {out}')
        return out

    def subgroup(self, model, y_true, y_pred, mask, labels=('has pharmacophore', 'has none')):
        """Score separately on molecules with and without a known binding motif."""
        from sklearn.metrics import mean_squared_error, r2_score
        out = {}
        for label, m in [(labels[0], mask), (labels[1], ~mask)]:
            if m.sum() < 20:
                continue
            out[label] = {
                'n': int(m.sum()),
                'RMSE': round(float(mean_squared_error(y_true[m], y_pred[m]) ** 0.5), 4),
                'R2': round(float(r2_score(y_true[m], y_pred[m])), 4),
            }
            self.log(f"  {label:22s} n={m.sum():4d}  RMSE {out[label]['RMSE']:.3f}  "
                     f"R2 {out[label]['R2']:.3f}")
        path = self.path('subgroup_scores.json')
        allj = json.load(open(path)) if os.path.exists(path) else {}
        allj[model] = out
        json.dump(allj, open(path, 'w'), indent=2)
        return out

    def summary(self):
        if not os.path.exists(self.results_csv):
            self.log('no results yet'); return
        r = pd.read_csv(self.results_csv).sort_values('R2', ascending=False)
        self.log('\n=== results so far ===')
        self.log(r[['model', 'representation', 'RMSE', 'MAE', 'R2', 'spread_ratio']]
                 .to_string(index=False))
        return r
