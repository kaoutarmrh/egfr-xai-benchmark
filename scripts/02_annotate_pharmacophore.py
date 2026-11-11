"""
02_annotate_pharmacophore.py
Label, for every molecule, which atoms belong to substructures that medicinal
chemistry already knows to matter for EGFR inhibition.

This file is the scientific core of the paper. It turns "the model highlighted
some atoms" into "the model highlighted the RIGHT atoms", because it gives you
a ground truth that does not depend on the model.

Chemical justification (all from published EGFR crystal structures):
  - 4-anilinoquinazoline: the scaffold of gefitinib, erlotinib, lapatinib.
    Quinazoline N1 accepts the hydrogen bond from the Met793 backbone NH in the
    kinase hinge. Without it the series loses its anchor.
  - acrylamide warhead: the Michael acceptor of afatinib, osimertinib,
    neratinib. It forms a covalent bond to Cys797.
  - anilino NH linker: places the aryl group into the hydrophobic back pocket
    controlled by the Thr790 gatekeeper.
  - solubilising tail (morpholine, dimethylaminoalkyl): improves PK, points to
    solvent, contributes little to binding. This is a NEGATIVE control: a model
    that puts its attribution here has learned the wrong thing.

Run:
    python 02_annotate_pharmacophore.py            # self-test on reference drugs
    python 02_annotate_pharmacophore.py data/egfr_wt.csv
"""

import sys, json
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------------------
# Pharmacophore definitions. All patterns verified against reference drugs.
# role: "core"    -> binding determinant, attribution here is CORRECT
#       "control" -> solubilising / peripheral, attribution here is SUSPECT
# ----------------------------------------------------------------------
PHARMACOPHORE = {
    "quinazoline_core": {
        "smarts": "c1ccc2ncncc2c1",
        "role": "core",
        "note": "gefitinib/erlotinib/lapatinib/afatinib scaffold",
    },
    "anilinoquinazoline": {
        "smarts": "[NX3;H1](c1ccccc1)c1ncnc2ccccc12",
        "role": "core",
        "note": "4-anilinoquinazoline, the classic EGFR chemotype",
    },
    "hinge_N_acceptor": {
        "smarts": "n1cnc2ccccc2c1",
        "role": "core",
        "note": "N1 H-bond acceptor to Met793 backbone",
    },
    "acrylamide_warhead": {
        "smarts": "[CX3]=[CX3][CX3](=O)[NX3]",
        "role": "core",
        "note": "Michael acceptor, covalent bond to Cys797",
    },
    "anilino_NH_linker": {
        "smarts": "[c]-[NX3;H1]-[c]",
        "role": "core",
        "note": "directs aryl into the Thr790 back pocket",
    },
    "quinoline_carbonitrile": {
        "smarts": "N#Cc1cnc2ccccc2c1",
        "role": "core",
        "note": "neratinib / pelitinib chemotype",
    },
    "aminopyrimidine": {
        "smarts": "[NX3;H1]-c1ncccn1",
        "role": "core",
        "note": "osimertinib-type hinge binder",
    },
    "morpholine_tail": {
        "smarts": "C1COCCN1",
        "role": "control",
        "note": "solubilising group, solvent-exposed",
    },
    "dimethylamino_tail": {
        "smarts": "[CX4][NX3]([CH3])[CH3]",
        "role": "control",
        "note": "solubilising group, solvent-exposed",
    },
}

REFERENCE_DRUGS = {
    "gefitinib":   "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "erlotinib":   "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "afatinib":    "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1",
    "osimertinib": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-n2cnc3ccccc32)n1",
    "lapatinib":   "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    "neratinib":   "CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C",
    "aspirin":     "CC(=O)Oc1ccccc1C(=O)O",          # must match nothing
    "caffeine":    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",      # must match nothing
}

_COMPILED = {k: (Chem.MolFromSmarts(v["smarts"]), v)
             for k, v in PHARMACOPHORE.items()}
for k, (p, _) in _COMPILED.items():
    assert p is not None, f"invalid SMARTS for {k}"


def annotate(smiles):
    """Return which pharmacophore features a molecule contains and which atom
    indices they occupy.

    core_atoms / control_atoms are the sets you will compare your XAI
    attributions against.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features, core, control = {}, set(), set()
    for name, (patt, meta) in _COMPILED.items():
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        atoms = {a for m in matches for a in m}
        features[name] = sorted(atoms)
        (core if meta["role"] == "core" else control).update(atoms)

    control -= core          # a shared atom counts as core
    n = mol.GetNumAtoms()
    return {
        "n_atoms": n,
        "features": features,
        "core_atoms": sorted(core),
        "control_atoms": sorted(control),
        "core_fraction": round(len(core) / n, 4),
        "has_pharmacophore": len(core) > 0,
    }


def attribution_score(attributions, core_atoms, n_atoms):
    """THE METRIC OF THE PAPER.

    attributions : list of |importance| per atom, produced by your XAI method
    core_atoms   : atom indices from annotate()

    Returns:
      pcs  Pharmacophore Concentration Score. Share of total attribution mass
           landing on known binding atoms. 1.0 = perfect, 0.0 = none.
      enr  Enrichment over chance. >1 means the model concentrates on the
           pharmacophore more than a random attribution would. This is the
           number to report: it corrects for big cores in small molecules.
    """
    total = sum(abs(a) for a in attributions)
    if total == 0 or not core_atoms:
        return {"pcs": 0.0, "enrichment": 0.0}
    on_core = sum(abs(attributions[i]) for i in core_atoms if i < len(attributions))
    pcs = on_core / total
    baseline = len(core_atoms) / n_atoms
    return {"pcs": round(pcs, 4),
            "enrichment": round(pcs / baseline, 4) if baseline else 0.0}


def self_test():
    print("=== reference drugs ===")
    for name, smi in REFERENCE_DRUGS.items():
        a = annotate(smi)
        feats = ", ".join(a["features"]) or "(none)"
        print(f"{name:12s} core={len(a['core_atoms']):3d}/{a['n_atoms']:3d} "
              f"ctrl={len(a['control_atoms']):2d}  {feats}")

    print("\n=== metric demo: gefitinib ===")
    a = annotate(REFERENCE_DRUGS["gefitinib"])
    n = a["n_atoms"]
    good = [1.0 if i in a["core_atoms"] else 0.05 for i in range(n)]
    bad  = [1.0 if i in a["control_atoms"] else 0.05 for i in range(n)]
    flat = [1.0] * n
    for label, attr in [("focuses on core", good),
                        ("focuses on tail", bad),
                        ("uniform (random)", flat)]:
        s = attribution_score(attr, a["core_atoms"], n)
        print(f"  {label:18s} PCS={s['pcs']:.3f}  enrichment={s['enrichment']:.2f}")


def run_dataset(path):
    df = pd.read_csv(path)
    rows = []
    for smi in df["smiles"]:
        a = annotate(smi)
        rows.append(a if a else {"n_atoms": 0, "features": {}, "core_atoms": [],
                                 "control_atoms": [], "core_fraction": 0.0,
                                 "has_pharmacophore": False})
    df["core_atoms"] = [json.dumps(r["core_atoms"]) for r in rows]
    df["control_atoms"] = [json.dumps(r["control_atoms"]) for r in rows]
    df["core_fraction"] = [r["core_fraction"] for r in rows]
    df["has_pharmacophore"] = [r["has_pharmacophore"] for r in rows]

    out = path.replace(".csv", "_annotated.csv")
    df.to_csv(out, index=False)

    print(f"molecules                     : {len(df)}")
    print(f"carrying >=1 core feature     : {df.has_pharmacophore.mean():.1%}")
    print("\nfeature frequency:")
    for name in PHARMACOPHORE:
        k = sum(name in r["features"] for r in rows)
        print(f"  {name:24s} {k:6d}  ({k/len(df):5.1%})  [{PHARMACOPHORE[name]['role']}]")
    if "pIC50" in df:
        hi = df[df.has_pharmacophore].pIC50.mean()
        lo = df[~df.has_pharmacophore].pIC50.mean()
        print(f"\nmean pIC50 with pharmacophore : {hi:.2f}")
        print(f"mean pIC50 without            : {lo:.2f}")
        print("(the gap is your sanity check: the annotation should track potency)")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dataset(sys.argv[1])
    else:
        self_test()
