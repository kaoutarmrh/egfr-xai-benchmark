"""
01_build_dataset.py
Rebuild the EGFR bioactivity dataset from ChEMBL, clean it, and make a
scaffold-based train/val/test split.

Run in Google Colab:
    !pip install -q rdkit chembl_webresource_client pandas
    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/drive/MyDrive/conf
    !python 01_build_dataset.py

Output: data/egfr_wt.csv, data/egfr_mutant.csv, data/split_report.txt
Runtime: 10-20 min (the ChEMBL download is the slow part).
"""

import os, re, json
import pandas as pd
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

TARGET = "CHEMBL203"          # human EGFR / erbB1
OUTDIR = "data"
RAW_CACHE = os.path.join(OUTDIR, "egfr_raw.csv")
SEED = 42

# Assay descriptions containing these strings are mutant assays, not wild type.
MUTANT_PATTERNS = [
    r"L858R", r"T790M", r"T\s*790", r"L\s*858", r"C797S", r"G719[SAC]?",
    r"exon\s*19", r"del\s*19", r"E746", r"A750", r"delE746", r"L861Q",
    r"S768I", r"insertion", r"exon\s*20", r"double\s*mutant", r"triple\s*mutant",
    r"\bmutant\b", r"\bmutation\b",
]
MUTANT_RE = re.compile("|".join(MUTANT_PATTERNS), flags=re.I)


# ----------------------------------------------------------------------
# 1. Download
# ----------------------------------------------------------------------
def download():
    """Pull IC50 activities for EGFR from the ChEMBL web service.

    If the REST API is down (it returns HTTP 500 from time to time), use the
    FTP mirror instead:
        !pip install chembl_downloader
        import chembl_downloader
        df = chembl_downloader.query(SQL)    # see README note
    """
    if os.path.exists(RAW_CACHE):
        print(f"[1] using cached {RAW_CACHE}")
        return pd.read_csv(RAW_CACHE)

    from chembl_webresource_client.new_client import new_client

    print(f"[1] downloading activities for {TARGET} ...")
    act = new_client.activity
    res = act.filter(
        target_chembl_id=TARGET,
        standard_type="IC50",
        assay_type="B",                     # binding/functional biochemical assays
    ).only([
        "molecule_chembl_id", "canonical_smiles", "standard_value",
        "standard_units", "standard_relation", "pchembl_value",
        "assay_chembl_id", "assay_description", "target_organism",
        "assay_variant_mutation", "document_chembl_id",
    ])

    df = pd.DataFrame(list(res))
    os.makedirs(OUTDIR, exist_ok=True)
    df.to_csv(RAW_CACHE, index=False)
    print(f"    {len(df)} raw measurements -> {RAW_CACHE}")
    return df


# ----------------------------------------------------------------------
# 2. Filter measurements
# ----------------------------------------------------------------------
def filter_measurements(df):
    n0 = len(df)
    log = [f"raw measurements: {n0}"]

    df = df[df["canonical_smiles"].notna()]
    df = df[df["pchembl_value"].notna()]
    log.append(f"has SMILES + pchembl_value: {len(df)}")

    # keep only exact measurements; ">10000" tells us nothing about potency
    df = df[df["standard_relation"] == "="]
    log.append(f"standard_relation '=': {len(df)}")

    df = df[df["standard_units"] == "nM"]
    log.append(f"units nM: {len(df)}")

    df = df[df["target_organism"].fillna("").str.contains("sapiens", case=False)]
    log.append(f"human target: {len(df)}")

    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df[df["pchembl_value"].between(2, 12)]
    log.append(f"pchembl in [2,12]: {len(df)}")

    return df, log


# ----------------------------------------------------------------------
# 3. Split wild type from mutants
# ----------------------------------------------------------------------
def tag_variant(df):
    """A single ChEMBL target id mixes wild-type and mutant assays. Mixing them
    silently is a serious flaw: a compound can be nanomolar on WT and inactive
    on T790M. We separate them using the mutation field and the assay text."""
    mut_field = df.get("assay_variant_mutation", pd.Series([None] * len(df)))
    desc = df["assay_description"].fillna("")
    is_mut = mut_field.notna() | desc.str.contains(MUTANT_RE)
    df = df.copy()
    df["variant"] = ["mutant" if m else "wild_type" for m in is_mut]
    return df


# ----------------------------------------------------------------------
# 4. Standardise structures
# ----------------------------------------------------------------------
_lfc = rdMolStandardize.LargestFragmentChooser()
_unc = rdMolStandardize.Uncharger()

def standardise(smiles):
    """Salt strip -> neutralise -> canonical SMILES. Returns None if unusable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _lfc.choose(mol)
        mol = _unc.uncharge(mol)
        mol = rdMolStandardize.Cleanup(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumHeavyAtoms() < 6:
        return None
    # drop anything with no carbon, or with elements outside organic chemistry
    allowed = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"}
    syms = {a.GetSymbol() for a in mol.GetAtoms()}
    if "C" not in syms or not syms.issubset(allowed):
        return None
    return Chem.MolToSmiles(mol)


# ----------------------------------------------------------------------
# 5. Aggregate duplicates
# ----------------------------------------------------------------------
def aggregate(df, max_spread=1.0):
    """One row per unique structure. Median of replicates; a compound whose
    replicate measurements disagree by more than `max_spread` log units is
    discarded as unreliable."""
    g = df.groupby("smiles")["pchembl_value"]
    agg = g.agg(["median", "min", "max", "count"]).reset_index()
    agg["spread"] = agg["max"] - agg["min"]
    keep = agg[(agg["count"] == 1) | (agg["spread"] <= max_spread)]
    out = keep[["smiles", "median", "count", "spread"]].rename(
        columns={"median": "pIC50", "count": "n_measurements"})
    # carry one ChEMBL id per structure for traceability
    ids = df.groupby("smiles")["molecule_chembl_id"].first().reset_index()
    out = out.merge(ids, on="smiles", how="left")
    dropped = len(agg) - len(keep)
    return out, dropped


# ----------------------------------------------------------------------
# 6. Scaffold split
# ----------------------------------------------------------------------
def scaffold_split(df, frac=(0.8, 0.1, 0.1), seed=SEED):
    """Bemis-Murcko scaffold split.

    Do NOT use a random split here. ChEMBL is full of analogue series: a random
    split puts near-identical molecules in train and test, R2 looks excellent
    and means nothing. Reviewers in this field check this first.
    """
    buckets = defaultdict(list)
    for i, smi in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = ""
        buckets[scaf].append(i)

    # largest scaffold groups go to train -> test contains rarer chemotypes
    groups = sorted(buckets.values(), key=len, reverse=True)

    n = len(df)
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    train, val, test = [], [], []
    for grp in groups:
        if len(train) + len(grp) <= n_train:
            train += grp
        elif len(val) + len(grp) <= n_val:
            val += grp
        else:
            test += grp

    split = pd.Series(["train"] * n, index=df.index)
    split.iloc[val] = "val"
    split.iloc[test] = "test"
    out = df.copy()
    out["split"] = split.values
    out["scaffold"] = [
        MurckoScaffold.MurckoScaffoldSmiles(smi, includeChirality=False)
        if Chem.MolFromSmiles(smi) else "" for smi in out["smiles"]
    ]
    return out, len(buckets)


# ----------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    raw = download()

    df, log = filter_measurements(raw)
    df = tag_variant(df)
    log.append(f"wild_type rows: {(df.variant=='wild_type').sum()}  "
               f"mutant rows: {(df.variant=='mutant').sum()}")

    print("[2] standardising structures ...")
    df["smiles"] = [standardise(s) for s in df["canonical_smiles"]]
    df = df[df["smiles"].notna()]
    log.append(f"after standardisation: {len(df)}")

    report = list(log)
    for variant in ["wild_type", "mutant"]:
        sub = df[df["variant"] == variant]
        if len(sub) < 100:
            report.append(f"\n[{variant}] only {len(sub)} rows - skipped")
            continue
        agg, dropped = aggregate(sub)
        agg, n_scaf = scaffold_split(agg)

        path = os.path.join(OUTDIR, f"egfr_{'wt' if variant=='wild_type' else 'mutant'}.csv")
        agg.to_csv(path, index=False)

        counts = agg["split"].value_counts().to_dict()
        active = (agg["pIC50"] >= 6).mean()
        report += [
            f"\n[{variant}]",
            f"  unique structures      : {len(agg)}",
            f"  discarded (spread >1)  : {dropped}",
            f"  distinct scaffolds     : {n_scaf}",
            f"  split                  : {counts}",
            f"  pIC50 mean/std         : {agg.pIC50.mean():.2f} / {agg.pIC50.std():.2f}",
            f"  fraction pIC50 >= 6    : {active:.1%}",
            f"  saved                  : {path}",
        ]

    txt = "\n".join(report)
    with open(os.path.join(OUTDIR, "split_report.txt"), "w") as f:
        f.write(txt)
    print("\n" + txt)
    print(f"\n[done] report -> {OUTDIR}/split_report.txt")


if __name__ == "__main__":
    main()
