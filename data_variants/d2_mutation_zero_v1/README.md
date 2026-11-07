# d2_mutation_zero_v1

D0 48 + D2 48 = 96 dim Gene.feature. Mapping-blocked D2 blanks filled with 0 as missing placeholders.
Original `feature_observed.tsv` keeps those cells as unobserved (0).
Reusable blocks/splits/gene_toids are hard-linked or copied from D0; do not write through them.

Load-only check:

```powershell
python -c "from run_ablation import load_data; from pathlib import Path; x,g,s,f=load_data(Path('data_variants/d2_mutation_zero_v1')); print(tuple(x.shape), len(g), len(s))"
```
