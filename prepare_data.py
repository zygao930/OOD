import os
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.makedirs("data", exist_ok=True)
os.makedirs("results/splits", exist_ok=True)

from data_utils import pair_genes


###############################################################################
# 1. Norman
###############################################################################

print("=" * 60)
print("1. Norman (CRISPRa, K562)")
print("=" * 60)

from gears import PertData

pert_data = PertData("./data")
pert_data.load(data_name="norman")
adata = pert_data.adata
print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

single_perts, pair_perts, single_gene_names = [], [], set()
for c in adata.obs["condition"].unique():
    if c == "ctrl":
        continue
    genes = pair_genes(c)
    if len(genes) == 1:
        single_perts.append(c)
        single_gene_names.add(genes[0])
    elif len(genes) == 2:
        pair_perts.append(c)

comp_eligible = [
    p for p in pair_perts
    if all(g in single_gene_names for g in pair_genes(p))
]
non_eligible = [p for p in pair_perts if p not in comp_eligible]

print(f"  Singles: {len(single_perts)}")
print(f"  Pairs: {len(pair_perts)} (eligible: {len(comp_eligible)})")

for seed in [42, 43, 44]:
    rng = np.random.RandomState(seed)
    eligible = list(comp_eligible)
    rng.shuffle(eligible)
    n = len(eligible)
    n_test = max(1, int(n * 0.2))
    n_val = max(1, int(n * 0.2))

    split = {
        "test": eligible[:n_test],
        "val": eligible[n_test:n_test + n_val],
        "train": single_perts + eligible[n_test + n_val:] + non_eligible + ["ctrl"],
    }

    path = f"results/splits/norman_seed{seed}.pkl"
    with open(path, "wb") as f:
        pickle.dump(split, f)
    print(f"  Seed {seed}: test={len(split['test'])}, "
          f"val={len(split['val'])}, train={len(split['train'])}")


###############################################################################
# 2. Adamson
###############################################################################

print("\n" + "=" * 60)
print("2. Adamson (CRISPRi, K562)")
print("=" * 60)

try:
    pert_data_adam = PertData("./data")
    pert_data_adam.load(data_name="adamson")
    adata_a = pert_data_adam.adata
    print(f"  Cells: {adata_a.n_obs}, Genes: {adata_a.n_vars}")

    single_genes_a = set()
    for c in adata_a.obs["condition"].unique():
        if c == "ctrl":
            continue
        genes = pair_genes(c)
        if len(genes) == 1:
            single_genes_a.add(genes[0])

    gene_list = sorted(single_genes_a)
    print(f"  Unique single-gene perturbations: {len(gene_list)}")

    all_conds = list(adata_a.obs["condition"].unique())
    for seed in [42, 43, 44]:
        rng = np.random.RandomState(seed)
        shuffled = list(gene_list)
        rng.shuffle(shuffled)

        n_test_genes = max(1, int(len(shuffled) * 0.2))
        n_val_genes = max(1, int(len(shuffled) * 0.1))

        test_genes = set(shuffled[:n_test_genes])
        val_genes = set(shuffled[n_test_genes:n_test_genes + n_val_genes])
        train_genes = set(shuffled[n_test_genes + n_val_genes:])

        def conds_for(gene_set):
            return [
                c for c in all_conds
                if c != "ctrl" and any(g in gene_set for g in pair_genes(c))
            ]

        train_conds = ["ctrl"] + conds_for(train_genes)
        val_conds = conds_for(val_genes)
        test_conds = conds_for(test_genes)

        split = {
            "train": train_conds,
            "val": val_conds,
            "test": test_conds,
            "test_genes": sorted(test_genes),
            "val_genes": sorted(val_genes),
            "train_genes": sorted(train_genes),
            "held_out_genes": sorted(test_genes),  
        }
        path = f"results/splits/adamson_seed{seed}.pkl"
        with open(path, "wb") as f:
            pickle.dump(split, f)
        print(f"  Seed {seed}: train_genes={len(train_genes)} "
              f"val_genes={len(val_genes)} test_genes={len(test_genes)} "
              f"| val_conds={len(val_conds)} test_conds={len(test_conds)}")

except Exception as e:
    print(f"  Adamson loading failed: {e}")
    print("  Skipping Adamson splits.")


###############################################################################
# 3. Joung-Zhang 2023
###############################################################################

print("\n" + "=" * 60)
print("3. Joung-Zhang 2023 (combinatorial)")
print("=" * 60)

try:
    import anndata as ad

    DATA_PATH = "data/joungzhang/JoungZhang2023_combinatorial.h5ad"
    if os.path.exists(DATA_PATH):
        adata_c = ad.read_h5ad(DATA_PATH)
        print(f"  Cells: {adata_c.n_obs}, Genes: {adata_c.n_vars}")
        print("  Joung-Zhang 2023 uses leave-one-out CV (splits created at runtime).")
    else:
        print(f"  File not found: {DATA_PATH}")
        print("  Download JoungZhang2023 combinatorial data manually.")
except Exception as e:
    print(f"  Joung-Zhang 2023 setup: {e}")


print("\n" + "=" * 60)
print("DONE: splits saved to results/splits/")
print("=" * 60)
