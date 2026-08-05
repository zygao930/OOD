"""
Gene-level failure analysis (Section 3.3, Figure 3 data, Table 6).

Decomposes pair-level predictions into per-gene interaction corrections
to identify where GEARS's corrections go wrong.

Outputs:
    results/gene_level/gene_level_results.json
    results/gene_level/pair_comparison.json   (Figure 3 data)

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_gene_level_analysis.py
"""

import os
import json
import pickle
import warnings
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

from data_utils import pair_genes

warnings.filterwarnings("ignore")
os.makedirs("results/gene_level", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [42, 43, 44]


###############################################################################
# Load data and GEARS
###############################################################################

print("=" * 60)
print("Loading Norman dataset")
print("=" * 60)

from gears import PertData, GEARS

pert_data = PertData("./data")
pert_data.load(data_name="norman")
adata = pert_data.adata
n_genes_out = adata.n_vars

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

ctrl_mask = adata.obs["condition"] == "ctrl"
ctrl_X = adata[ctrl_mask].X
ctrl_mean = np.array(
    ctrl_X.toarray().mean(0) if hasattr(ctrl_X, "toarray") else ctrl_X.mean(0)
).flatten()


def get_effect(cond):
    mask = adata.obs["condition"] == cond
    if mask.sum() == 0:
        return None
    X = adata[mask].X
    return np.array(
        X.toarray().mean(0) if hasattr(X, "toarray") else X.mean(0)
    ).flatten() - ctrl_mean


single_effects = {}
for c in single_perts:
    eff = get_effect(c)
    if eff is not None:
        genes = pair_genes(c)
        if genes:
            single_effects[genes[0]] = eff

all_pair_effects = {}
for c in pair_perts:
    eff = get_effect(c)
    if eff is not None:
        all_pair_effects[c] = eff

gene_names = (
    list(adata.var["gene_name"]) if "gene_name" in adata.var.columns
    else list(adata.var_names)
)


###############################################################################
# Co-expression node degree (from the GEARS co-expression graph)
###############################################################################

# Match the original order: prepare the split and dataloaders and load the
# GEARS model FIRST, which is what populates pert_data.gene_sim_network, then
# read node degrees from that graph. Only if the real graph is genuinely
# absent do we fall back to a data-derived cosine graph, and we say so loudly.
coexpress_degree = np.zeros(n_genes_out)

_seed42_split = "results/splits/norman_seed42.pkl"
_seed42_gears = "results/norman/gears_seed42"
if not os.path.exists(_seed42_gears):
    _seed42_gears = "results/gears_seed42"

if os.path.exists(_seed42_split):
    try:
        pert_data.prepare_split(split="custom", seed=1, split_dict_path=_seed42_split)
        pert_data.get_dataloader(batch_size=64, test_batch_size=256)
        if os.path.exists(_seed42_gears):
            _gm = GEARS(pert_data, device=device)
            _gm.model_initialize(hidden_size=64)
            _gm.load_pretrained(_seed42_gears)
    except Exception as e:
        print(f"  Warning: could not prepare split / load GEARS for graph: {e}")

try:
    sim = getattr(pert_data, "gene_sim_network", None)
    if sim is not None and hasattr(sim, "edge_index"):
        ei = sim.edge_index
        if isinstance(ei, torch.Tensor):
            ei = ei.cpu().numpy()
        for i in range(ei.shape[1]):
            src, dst = int(ei[0, i]), int(ei[1, i])
            if src < n_genes_out:
                coexpress_degree[src] += 1
            if dst < n_genes_out:
                coexpress_degree[dst] += 1
        print(f"  Co-expression edges from GEARS graph: {ei.shape[1]}")
    else:
        print("  Warning: pert_data.gene_sim_network unavailable after init.")
except Exception as e:
    print(f"  Could not extract GEARS co-expression graph: {e}")

if coexpress_degree.max() == 0:
    print("  WARNING: GEARS co-expression graph not found; falling back to a "
          "data-derived cosine graph. This is NOT identical to the GEARS graph "
          "and the resulting degree correlations may differ from the paper.")
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        nonctrl = adata[adata.obs["condition"] != "ctrl"]
        X = nonctrl.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        if X.shape[0] > 5000:
            idx = np.random.RandomState(42).choice(X.shape[0], 5000, replace=False)
            X = X[idx]
        chunk = 500
        for start in range(0, n_genes_out, chunk):
            end = min(start + chunk, n_genes_out)
            sim_chunk = cosine_similarity(X[:, start:end].T, X.T)
            for i in range(end - start):
                coexpress_degree[start + i] = np.sum(np.abs(sim_chunk[i]) > 0.3) - 1
        print(f"  Fallback max co-expression degree: {coexpress_degree.max():.0f}")
    except Exception as e:
        print(f"  Co-expression fallback failed: {e}")


###############################################################################
# Per-gene analysis across seeds
###############################################################################

pair_data = []
pair_gene_data = []                        # per-pair Top20-vs-rest statistics
gene_corrections = defaultdict(list)       # per-gene accumulation across pairs
# Per-gene running sums for the gene-level correction-vs-error correlation.
gene_corr_sum = np.zeros(n_genes_out)
gene_err_sum = np.zeros(n_genes_out)
gene_count = np.zeros(n_genes_out)

for seed in SEEDS:
    print(f"\nSeed {seed}")

    split_path = f"results/splits/norman_seed{seed}.pkl"
    if not os.path.exists(split_path):
        continue

    with open(split_path, "rb") as f:
        split = pickle.load(f)
    test_pairs = split["test"]

    gears_dir = f"results/norman/gears_seed{seed}"
    if not os.path.exists(gears_dir):
        gears_dir = f"results/gears_seed{seed}"
    if not os.path.exists(gears_dir):
        print(f"  GEARS model not found. Skipping.")
        continue

    pert_data.prepare_split(split="custom", seed=1, split_dict_path=split_path)
    pert_data.get_dataloader(batch_size=64, test_batch_size=256)
    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(hidden_size=64)
    gears_model.load_pretrained(gears_dir)

    for p in test_pairs:
        genes = pair_genes(p)
        if not (len(genes) == 2 and genes[0] in single_effects
                and genes[1] in single_effects and p in all_pair_effects):
            continue

        true_eff = all_pair_effects[p]
        add_pred = single_effects[genes[0]] + single_effects[genes[1]]
        # Top-20 DE genes are recomputed PER PAIR from this pair's true effect.
        top20 = np.argsort(np.abs(true_eff))[-20:]
        rest_idx = np.delete(np.arange(len(true_eff)), top20)

        try:
            res = gears_model.predict([genes])
            gears_pred = np.asarray(next(iter(res.values()))).reshape(-1) - ctrl_mean
        except Exception:
            continue

        # Per-gene interaction correction and prediction error for THIS pair.
        interaction_correction = np.abs(gears_pred - add_pred)
        pred_error = np.abs(gears_pred - true_eff)
        additive_error = np.abs(add_pred - true_eff)

        # Accumulate per-gene sums across all pairs (gene-level correlation).
        gene_corr_sum += interaction_correction
        gene_err_sum += pred_error
        gene_count += 1

        # Store lightweight per-gene entries with this pair's Top20 status.
        for g_idx in range(n_genes_out):
            gene_corrections[g_idx].append({
                "correction": float(gears_pred[g_idx] - add_pred[g_idx]),
                "error": float(pred_error[g_idx]),
                "is_top20": bool(g_idx in top20),
            })

        # Per-pair Top20-vs-rest statistics (drives the 9.8x ratio, rho, wins).
        pair_gene_data.append({
            "pair": p, "seed": seed,
            "mean_correction_top20": float(np.mean(interaction_correction[top20])),
            "mean_correction_rest": float(np.mean(interaction_correction[rest_idx])),
            "mean_error_top20": float(np.mean(pred_error[top20])),
            "mean_error_rest": float(np.mean(pred_error[rest_idx])),
            "mean_additive_err_top20": float(np.mean(additive_error[top20])),
            "mean_additive_err_rest": float(np.mean(additive_error[rest_idx])),
            "pair_ad": float(np.mean(interaction_correction)),
            "pair_error": float(np.mean(pred_error)),
        })

        # Store pair-level comparison
        gears_mae = float(np.mean(np.abs(gears_pred - true_eff)))
        add_mae = float(np.mean(np.abs(add_pred - true_eff)))
        gears_top20 = float(np.mean(np.abs(gears_pred[top20] - true_eff[top20])))
        add_top20 = float(np.mean(np.abs(add_pred[top20] - true_eff[top20])))

        pair_data.append({
            "seed": seed,
            "pair": p,
            "gears_mae": gears_mae,
            "additive_mae": add_mae,
            "gears_top20": gears_top20,
            "additive_top20": add_top20,
            "ad": float(np.mean(np.abs(gears_pred - add_pred))),
            "gears_improves": gears_top20 < add_top20,
        })


###############################################################################
# Results: Section 3.3
###############################################################################

print("\n" + "=" * 60)
print("GENE-LEVEL ANALYSIS (Section 3.3)")
print("=" * 60)

# Per-pair Top-20 vs rest correction ratio, correlation, and wins.
mean_corr_top20 = mean_corr_other = ratio = None
corr_t20 = corr_rest = None
gears_wins_t20 = gears_wins_rest = None
n_pairs_pg = len(pair_gene_data)

if pair_gene_data:
    t20_corrections = np.array([d["mean_correction_top20"] for d in pair_gene_data])
    rest_corrections = np.array([d["mean_correction_rest"] for d in pair_gene_data])
    t20_errors = np.array([d["mean_error_top20"] for d in pair_gene_data])
    rest_errors = np.array([d["mean_error_rest"] for d in pair_gene_data])
    t20_add_errors = np.array([d["mean_additive_err_top20"] for d in pair_gene_data])
    rest_add_errors = np.array([d["mean_additive_err_rest"] for d in pair_gene_data])

    mean_corr_top20 = float(np.mean(t20_corrections))
    mean_corr_other = float(np.mean(rest_corrections))
    ratio = mean_corr_top20 / max(mean_corr_other, 1e-10)

    corr_t20 = stats.spearmanr(t20_corrections, t20_errors)[0]
    corr_rest = stats.spearmanr(rest_corrections, rest_errors)[0]

    gears_wins_t20 = int(np.sum(t20_errors < t20_add_errors))
    gears_wins_rest = int(np.sum(rest_errors < rest_add_errors))

    print(f"\n  Per-pair mean |correction| Top-20: {mean_corr_top20:.6f}")
    print(f"  Per-pair mean |correction| rest:   {mean_corr_other:.6f}")
    print(f"  Ratio: {ratio:.1f}x")
    print(f"\n  Per-pair Spearman(correction, error):")
    print(f"    Top-20 DE genes: {corr_t20:.3f}")
    print(f"    Remaining genes: {corr_rest:.3f}")
    print(f"\n  GEARS beats additive (Top-20): {gears_wins_t20}/{n_pairs_pg} "
          f"({100 * gears_wins_t20 / n_pairs_pg:.1f}%)")
    print(f"  GEARS beats additive (rest):   {gears_wins_rest}/{n_pairs_pg} "
          f"({100 * gears_wins_rest / n_pairs_pg:.1f}%)")

# Gene-level correction vs error correlation (accumulated across all pairs).
rho = None
valid = gene_count > 0
if valid.sum() > 0:
    gene_corr_mean = np.zeros(n_genes_out)
    gene_err_mean = np.zeros(n_genes_out)
    gene_corr_mean[valid] = gene_corr_sum[valid] / gene_count[valid]
    gene_err_mean[valid] = gene_err_sum[valid] / gene_count[valid]
    rho = stats.spearmanr(gene_corr_mean[valid], gene_err_mean[valid])[0]
    print(f"\n  Gene-level Spearman(correction, error): rho = {rho:.3f}")

# Co-expression node degree vs correction and error (mechanistic analysis).
rho_degree_correction = rho_degree_error = None
if valid.sum() > 0 and coexpress_degree.max() > 0:
    deg_v = coexpress_degree[valid]
    corr_v = gene_corr_mean[valid]
    err_v = gene_err_mean[valid]
    rho_degree_correction = float(stats.spearmanr(deg_v, corr_v)[0])
    rho_degree_error = float(stats.spearmanr(deg_v, err_v)[0])
    print(f"  Spearman(co-expr degree, correction): rho = {rho_degree_correction:.3f}")
    print(f"  Spearman(co-expr degree, error):      rho = {rho_degree_error:.3f}")

    # Degree-binned means (quartiles), matching the original's summary table.
    print(f"\n  {'Degree bin':>15s} {'Mean correction':>18s} {'Mean error':>15s} {'N':>8s}")
    pct = [0, 25, 50, 75, 100]
    for i in range(len(pct) - 1):
        lo = np.percentile(deg_v, pct[i])
        hi = np.percentile(deg_v, pct[i + 1])
        mask_bin = (deg_v >= lo) & (deg_v <= hi) if i == len(pct) - 2 \
            else (deg_v >= lo) & (deg_v < hi)
        if mask_bin.sum() > 0:
            print(f"  {f'{pct[i]}-{pct[i+1]}th':>15s} "
                  f"{np.mean(corr_v[mask_bin]):>18.6f} "
                  f"{np.mean(err_v[mask_bin]):>15.6f} {int(mask_bin.sum()):>8d}")

# Pair-level: GEARS vs additive on Top-20 DE genes
n_improves = n_total = None
if pair_data:
    n_improves = sum(1 for p in pair_data if p["gears_improves"])
    n_total = len(pair_data)
    print(f"\n  GEARS improves on top-20 DE genes: {n_improves}/{n_total} "
          f"({100 * n_improves / n_total:.0f}%) of test pairs")


# Table 6: failure cases
print("\n  Representative failure cases (Table 6):")
sorted_pairs = sorted(pair_data, key=lambda p: p["gears_mae"], reverse=True)
print(f"\n  {'Pair':<25} {'GEARS':>8} {'Add.':>8} {'AD':>8}")
print(f"  High-interaction failures:")
for p in sorted_pairs[:5]:
    print(f"  {p['pair']:<25} {p['gears_mae']:>8.3f} {p['additive_mae']:>8.3f} {p['ad']:>8.3f}")
print(f"  Low-interaction noise:")
sorted_low = sorted([p for p in pair_data if p["ad"] < 0.04],
                     key=lambda p: p["gears_mae"], reverse=True)
for p in sorted_low[:5]:
    print(f"  {p['pair']:<25} {p['gears_mae']:>8.3f} {p['additive_mae']:>8.3f} {p['ad']:>8.3f}")


###############################################################################
# Save
###############################################################################

results = {
    "gene_stats_summary": {
        "top20_mean_correction": mean_corr_top20,
        "other_mean_correction": mean_corr_other,
        "top20_rest_ratio": ratio,
        "correction_error_correlation": rho,
        "per_pair_corr_error_top20": corr_t20,
        "per_pair_corr_error_rest": corr_rest,
        "gears_wins_top20": gears_wins_t20,
        "gears_wins_rest": gears_wins_rest,
        "n_pairs": n_pairs_pg,
        "gears_improves_fraction": (n_improves / n_total) if pair_data else None,
        "coexpress_degree_correction_rho": rho_degree_correction,
        "coexpress_degree_error_rho": rho_degree_error,
        "max_coexpress_degree": float(coexpress_degree.max()),
    },
}

with open("results/gene_level/pair_gene_data.json", "w") as f:
    json.dump(pair_gene_data, f, indent=2)

with open("results/gene_level/gene_level_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open("results/gene_level/pair_comparison.json", "w") as f:
    json.dump(pair_data, f, indent=2)

print(f"\nSaved results to results/gene_level/")
