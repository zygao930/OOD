"""
Signal combination and metric sensitivity analysis (Appendix D).

Produces:
    Table 11: Signal combination ablation. Each pair of signals is combined by
              taking the arithmetic mean of their min-max normalized scores
              (NOT a fitted logistic regression), matching the source code.
    Table 12: Reliability signal AUROC across evaluation metrics

Reads saved GEARS predictions. Requires GPU only if recomputing predictions.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_metric_sensitivity.py
"""

import os
import json
import pickle
import warnings

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from data_utils import pair_genes

warnings.filterwarnings("ignore")
os.makedirs("results/metric_sensitivity", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


###############################################################################
# Load Norman data and GEARS predictions
###############################################################################

print("=" * 60)
print("Loading Norman data and predictions")
print("=" * 60)

from data_utils import load_norman

data = load_norman()
single_effects = data["single_effects"]
all_pair_effects = data["all_pair_effects"]

# Load GEARS predictions
from gears import PertData, GEARS

pert_data = data["pert_data"]
ctrl_mean = data["ctrl_mean"]
n_genes_out = data["n_genes_out"]

SEEDS = [42, 43, 44]
records = []

# Build a GEARS embedding-space kNN / Mahalanobis structure from training
# pairs so the signal set matches Table 11 (which includes kNN + Mahalanobis).
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis as _mahalanobis

gene2idx_full = data["gene2idx"]


def _gears_pair_emb(model, genes):
    try:
        mdl = getattr(model, "best_model", model.model)
        w = mdl.gene_emb.weight.data.cpu().numpy()
    except Exception:
        return None
    embs = [w[gene2idx_full[g]] for g in genes
            if g in gene2idx_full and gene2idx_full[g] < w.shape[0]]
    return np.mean(embs, axis=0) if embs else None


for seed in SEEDS:
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
        continue

    pert_data.prepare_split(split="custom", seed=1, split_dict_path=split_path)
    pert_data.get_dataloader(batch_size=64, test_batch_size=256)
    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(hidden_size=64)
    gears_model.load_pretrained(gears_dir)

    # Build GEARS-embedding kNN + Mahalanobis structures from training pairs.
    train_pairs = [c for c in split["train"]
                   if c != "ctrl" and len(pair_genes(c)) == 2]
    tr_embs = []
    for c in train_pairs:
        e = _gears_pair_emb(gears_model, pair_genes(c))
        if e is not None:
            tr_embs.append(e)
    nn_seed = mahal_seed = None
    if len(tr_embs) >= 2:
        tr_arr = np.array(tr_embs)
        nn_seed = NearestNeighbors(
            n_neighbors=min(5, len(tr_arr)), metric="cosine").fit(tr_arr)
        if len(tr_arr) > 2:
            try:
                cov = EmpiricalCovariance().fit(tr_arr)
                mean = np.mean(tr_arr, axis=0)
                cov_inv = np.linalg.inv(cov.covariance_ + 1e-6 * np.eye(tr_arr.shape[1]))
                mahal_seed = (mean, cov_inv)
            except Exception:
                mahal_seed = None

    for p in test_pairs:
        genes = pair_genes(p)
        if not (len(genes) == 2 and genes[0] in single_effects
                and genes[1] in single_effects and p in all_pair_effects):
            continue

        true_eff = all_pair_effects[p]
        add_pred = single_effects[genes[0]] + single_effects[genes[1]]
        top20 = np.argsort(np.abs(true_eff))[-20:]

        try:
            res = gears_model.predict([genes])
            pred = np.asarray(next(iter(res.values()))).reshape(-1) - ctrl_mean
        except Exception:
            continue

        rec = {
            "seed": seed,
            "pair": p,
            "pred": pred,
            "true": true_eff,
            "additive": add_pred,
            "top20_idx": top20,
            "error_full_mae": float(np.mean(np.abs(pred - true_eff))),
            "error_top20_mae": float(np.mean(np.abs(pred[top20] - true_eff[top20]))),
            "error_full_mse": float(np.mean((pred - true_eff) ** 2)),
            "error_top20_mse": float(np.mean((pred[top20] - true_eff[top20]) ** 2)),
            "ad_l1": float(np.mean(np.abs(pred - add_pred))),
            "ad_l2": float(np.sqrt(np.mean((pred - add_pred) ** 2))),
            "pm": float(np.mean(np.abs(pred))),
            "gv": float(np.var(pred)),
        }
        # kNN + Mahalanobis in GEARS embedding space.
        emb = _gears_pair_emb(gears_model, genes)
        if emb is not None and nn_seed is not None:
            rec["knn"] = float(nn_seed.kneighbors(emb.reshape(1, -1))[0].mean())
            if mahal_seed is not None:
                try:
                    rec["mahalanobis"] = float(
                        _mahalanobis(emb, mahal_seed[0], mahal_seed[1]))
                except Exception:
                    rec["mahalanobis"] = None
        records.append(rec)

print(f"Loaded {len(records)} test predictions")


###############################################################################
# Table 11: Signal combination ablation
###############################################################################

print("\n" + "=" * 60)
print("TABLE 11: Signal combination ablation")
print("=" * 60)

if records:
    # Failure = top 25% error (75th percentile), matching the original.
    errors = np.array([r["error_full_mae"] for r in records])
    thresh = np.percentile(errors, 75)
    y_bin = (errors > thresh).astype(int)

    # Full signal set including kNN and Mahalanobis (Table 11).
    signal_defs = [
        ("AD", "ad_l1"),
        ("PM", "pm"),
        ("GV", "gv"),
        ("kNN", "knn"),
        ("Mahalanobis", "mahalanobis"),
    ]

    def _norm(a):
        """Min-max normalize to [0, 1]; constant arrays map to zeros."""
        a = np.asarray(a, dtype=float)
        lo, hi = np.nanmin(a), np.nanmax(a)
        if not np.isfinite(lo) or hi - lo < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    # Collect normalized signal arrays over records that have all needed keys.
    signal_arrays = {}
    for name, key in signal_defs:
        vals = [r.get(key) for r in records]
        if any(v is None for v in vals):
            continue
        signal_arrays[name] = _norm(vals)

    print(f"\n  Individual signals (failure = top 25% error):")
    print(f"  {'Signal':<28} {'AUROC':>7}")
    print(f"  {'~' * 38}")
    ind_aurocs = {}
    if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
        for name, arr in signal_arrays.items():
            auroc = roc_auc_score(y_bin, arr)
            ind_aurocs[name] = auroc
            print(f"  {name:<28} {auroc:>7.3f}")

    # Pairwise combinations by AVERAGING normalized scores (not logistic reg).
    print(f"\n  Pairwise combinations (mean of normalized signals):")
    print(f"  {'Combination':<28} {'AUROC':>7}")
    print(f"  {'~' * 38}")
    from itertools import combinations as combs
    combo_results = []
    names = list(signal_arrays.keys())
    if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
        for a, b in combs(names, 2):
            combined = (signal_arrays[a] + signal_arrays[b]) / 2
            auroc = roc_auc_score(y_bin, combined)
            combo_results.append((f"{a} + {b}", auroc))
        combo_results.sort(key=lambda x: x[1], reverse=True)
        for label, auroc in combo_results:
            print(f"  {label:<28} {auroc:>7.3f}")

        if ind_aurocs and combo_results:
            print(f"\n  Best individual: {max(ind_aurocs.values()):.3f}, "
                  f"Best combo: {combo_results[0][1]:.3f}")


###############################################################################
# Table 12: AUROC across metrics
###############################################################################

print("\n" + "=" * 60)
print("TABLE 12: Reliability signal AUROC across evaluation metrics")
print("=" * 60)

if records:
    metrics = [
        ("Full MAE", "error_full_mae"),
        ("Top20 MAE", "error_top20_mae"),
        ("Full MSE", "error_full_mse"),
        ("Top20 MSE", "error_top20_mse"),
    ]

    signals_eval = [
        ("AD (L1)", "ad_l1"),
        ("AD (L2)", "ad_l2"),
        ("PM", "pm"),
        ("GV", "gv"),
        ("kNN", "knn"),
        ("Mahalanobis", "mahalanobis"),
    ]

    print(f"\n  {'Signal':<20}", end="")
    for mname, _ in metrics:
        print(f" {mname:>10}", end="")
    print()
    print(f"  {'~' * 62}")

    for sname, skey in signals_eval:
        vals = [r.get(skey) for r in records]
        if any(v is None for v in vals):
            continue
        print(f"  {sname:<20}", end="")
        s = np.array(vals, dtype=float)
        for mname, mkey in metrics:
            errs = np.array([r[mkey] for r in records])
            thresh = np.percentile(errs, 75)   # top 25% error = failure
            yb = (errs > thresh).astype(int)
            if yb.sum() == 0 or yb.sum() == len(yb):
                print(f" {'n/a':>10}", end="")
            else:
                auroc = roc_auc_score(yb, s)
                print(f" {auroc:>10.3f}", end="")
        print()

    # Clean up large arrays from records before saving
    for r in records:
        for k in ["pred", "true", "additive", "top20_idx"]:
            if k in r:
                del r[k]

    with open("results/metric_sensitivity/results.json", "w") as f:
        json.dump(records, f, indent=2)

print("\nDONE")
