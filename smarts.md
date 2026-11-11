# Pharmacophore definitions

The structural reference used to score attribution maps. It is fixed from
crystallography before any model is trained, so it cannot be tuned to the
models it evaluates.

## Narrow target — direct binding determinants

Atoms that make the principal interactions. Direct and water-mediated contacts
are kept distinct, because only the former is a direct determinant.

| motif | SMARTS | atoms taken | interaction | structure |
|---|---|---|---|---|
| quinazoline hinge nitrogens | `n1cnc2ccccc2c1` | ring N atoms (see below) | N1 accepts a hydrogen bond directly from the Met793 backbone amide | 1M17 (erlotinib), 2ITY (gefitinib) |
| acrylamide warhead | `[CX3]=[CX3][CX3](=O)[NX3]` | match indices 0 and 1 (the two alkene carbons) | Michael addition to Cys797 | 4ZAU (osimertinib) |

### Separating N1 from N3

The two quinazoline nitrogens are **not** equivalent. N1 makes the direct hinge
hydrogen bond; N3 reaches Thr790 and Thr854 only through bridging water
molecules. They are distinguished by ring membership, not by atom order, so the
assignment does not depend on how a SMILES is written:

```python
def quinazoline_nitrogens(mol):
    """N1 is the ring nitrogen bonded to the ring-fusion carbon; N3 is the other."""
    ri = mol.GetRingInfo()
    n1, n3 = set(), set()
    for match in mol.GetSubstructMatches(QUINAZOLINE):
        for a in [x for x in match if mol.GetAtomWithIdx(x).GetSymbol() == 'N']:
            fused = any(ri.NumAtomRings(nb.GetIdx()) >= 2
                        for nb in mol.GetAtomWithIdx(a).GetNeighbors())
            (n1 if fused else n3).add(a)
    return n1, n3
```

Two narrow targets are reported:

- **N1 + N3 + warhead carbons** — 7.0% of heavy atoms per molecule (Table 1)
- **N1 + warhead carbons only**, the strict direct-contact target — 5.2%

The paper's conclusions are unchanged between them, which is the point of
reporting both.

### Validation against reference inhibitors

Checked at motif *and* atom level before use:

| compound | quinazoline motif | N1 | N3 | warhead carbons |
|---|---|---|---|---|
| gefitinib | match | 1 | 1 | 0 |
| erlotinib | match | 1 | 1 | 0 |
| lapatinib | match | 1 | 1 | 0 |
| afatinib | match | 1 | 1 | 2 |
| osimertinib | **no match** (pyrimidine core) | 0 | 0 | 2 |
| neratinib | — | — | — | match |
| aspirin | no match | 0 | 0 | 0 |
| caffeine | no match | 0 | 0 | 0 |

Osimertinib failing the quinazoline pattern is the informative case: it is a
potent EGFR inhibitor that does not carry the motif, so a pattern that matched
it would be wrong.

## Broad target — established EGFR chemotypes

The broad target is the union of the two narrow motifs with five further
scaffold and linker motifs, covering about 27.5% of heavy atoms per molecule.
These are the exact patterns used, from `PHARMACOPHORE` in
`scripts/02_annotate_pharmacophore.py`:

| name | SMARTS | role | note |
|---|---|---|---|
| `hinge_N_acceptor` | `n1cnc2ccccc2c1` | core | N1 hydrogen-bond acceptor to the Met793 backbone |
| `acrylamide_warhead` | `[CX3]=[CX3][CX3](=O)[NX3]` | core | Michael acceptor, covalent bond to Cys797 |
| `quinazoline_core` | `c1ccc2ncncc2c1` | core | gefitinib / erlotinib / lapatinib / afatinib scaffold |
| `anilinoquinazoline` | `[NX3;H1](c1ccccc1)c1ncnc2ccccc12` | core | 4-anilinoquinazoline, the classic EGFR chemotype |
| `anilino_NH_linker` | `[c]-[NX3;H1]-[c]` | core | directs the aryl group into the Thr790 back pocket |
| `quinoline_carbonitrile` | `N#Cc1cnc2ccccc2c1` | core | neratinib / pelitinib chemotype |
| `aminopyrimidine` | `[NX3;H1]-c1ncccn1` | core | osimertinib-type hinge binder |
| `morpholine_tail` | `C1COCCN1` | control | solubilising group, solvent-exposed |
| `dimethylamino_tail` | `[CX4][NX3]([CH3])[CH3]` | control | solubilising group, solvent-exposed |

The first two are the narrow target; the next five are what the broad target
adds. The two `control` motifs are solvent-exposed solubilising groups, used
during annotation development to confirm that the annotation does not simply
mark any recurring substructure.

### Reference compounds used for validation

```
gefitinib    COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1
erlotinib    C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1
afatinib     CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1
osimertinib  COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-n2cnc3ccccc32)n1
lapatinib    CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1
neratinib    CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C
aspirin      CC(=O)Oc1ccccc1C(=O)O            must match nothing
caffeine     Cn1c(=O)c2c(ncn2C)n(C)c1=O       must match nothing
```

### Why the breadth matters

Scoring the same attribution against both targets is the paper's second
finding. On the matched 236 compounds, AttentiveFP attribution scores *below*
chance against the broad annotation (0.92) and roughly twofold above chance
against the narrow one (1.96–2.16). A broad reference is not a safer choice
than a narrow one; it is a different question.

## Annotation sanity check

Independently of any model, compounds matching at least one broad motif are
more potent than non-matching compounds by 1.12 log10 units
(Mann–Whitney p = 1.6e-239, n = 5,855 versus 2,127). The annotation therefore
captures activity-relevant chemistry rather than arbitrary substructure.

## Enrichment

For non-negative atom-level attribution `a` and target `T` in a molecule of
`n` heavy atoms:

```
E(a, T) = [ sum_{i in T} a_i / sum_i a_i ] / [ |T| / n ]
```

Scale-invariant and normalised for target size. Being a ratio of dependent
random quantities its null expectation is not exactly one, which is why it is
calibrated against two empirical controls rather than against 1.0:

- **random atoms** — tests the metric
- **random values through the bit-to-atom map** — tests the projection, since
  atoms inside many overlapping environments would otherwise accumulate mass,
  and ring systems are exactly where the pharmacophore lives

Both controls are generated per molecule, so models are compared against them
by paired test within compound, not against a nominal value.
