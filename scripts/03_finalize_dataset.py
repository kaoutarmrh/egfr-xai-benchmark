"""
03_finalize_dataset.py
Take the recovered egfr_clean.csv, fix the scaffold leakage in the old split,
re-attach the pharmacophore annotation, and write the final dataset that every
later experiment will use.

Run once, then never touch the split again. Every model in the paper must be
evaluated on the same test set or the comparison is meaningless.

    python 03_finalize_dataset.py egfr_clean.csv
Output: egfr_final.csv
"""

import sys, json
import pandas as pd
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, ".")
from importlib import import_module
ann = import_module("02_annotate_pharmacophore") if False else None
RDLogger.DisableLog("rdApp.*")

SEED = 42
FRACS = (0.80, 0.10, 0.10)


def scaffold_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""


def strict_scaffold_split(df):
    """Every molecule sharing a Bemis-Murcko scaffold lands in the same fold.
    Zero leakage by construction. Largest scaffold families go to train, so the
    test set is enriched in rarer chemotypes - the harder, honest evaluation."""
    buckets = defaultdict(list)
    for i, s in enumerate(df["scaffold"]):
        buckets[s].append(i)

    groups = sorted(buckets.values(), key=len, reverse=True)
    n = len(df)
    n_tr, n_va = int(FRACS[0] * n), int(FRACS[1] * n)

    train, val, test = [], [], []
    for g in groups:
        if len(train) + len(g) <= n_tr:
            train += g
        elif len(val) + len(g) <= n_va:
            val += g
        else:
            test += g

    split = ["train"] * n
    for i in val:
        split[i] = "val"
    for i in test:
        split[i] = "test"
    return split


def main(path="egfr_clean.csv"):
    df = pd.read_csv(path)
    print(f"loaded {len(df)} molecules from {path}")

    df["scaffold"] = [scaffold_of(s) for s in df["smiles"]]
    df["split"] = strict_scaffold_split(df)

    # verify: no scaffold may appear in more than one fold
    chk = df.groupby("scaffold")["split"].nunique()
    assert chk.max() == 1, "leakage remains"
    print("leakage check passed: every scaffold lives in exactly one fold")

    # re-attach pharmacophore annotation
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pharm", "02_annotate_pharmacophore.py")
    pharm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pharm)

    rows = [pharm.annotate(s) or {} for s in df["smiles"]]
    df["core_atoms"] = [json.dumps(r.get("core_atoms", [])) for r in rows]
    df["control_atoms"] = [json.dumps(r.get("control_atoms", [])) for r in rows]
    df["core_fraction"] = [r.get("core_fraction", 0.0) for r in rows]
    df["has_pharmacophore"] = [r.get("has_pharmacophore", False) for r in rows]
    for name in pharm.PHARMACOPHORE:
        df[f"feat_{name}"] = [name in r.get("features", {}) for r in rows]

    df.to_csv("egfr_final.csv", index=False)

    print("\n--- final dataset ---")
    print(df.split.value_counts().to_dict())
    print(f"unique scaffolds : {df.scaffold.nunique()}")
    print(f"pIC50            : {df.pIC50.mean():.2f} +/- {df.pIC50.std():.2f}")
    print(f"with pharmacophore: {df.has_pharmacophore.mean():.1%}")
    for s in ["train", "val", "test"]:
        sub = df[df.split == s]
        print(f"  {s:5s} n={len(sub):5d}  pIC50={sub.pIC50.mean():.2f}  "
              f"pharm={sub.has_pharmacophore.mean():.1%}")
    print("\nsaved -> egfr_final.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "egfr_clean.csv")
