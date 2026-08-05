"""
GEARS architecture ablation (Table 5).

Ablation configurations:
    Full GEARS        Reuses the Table 1 full model (not retrained)
    No GO graph       Remove GO similarity edges, retrain
    No co-expression  Remove co-expression edges, retrain
    No graphs         Remove both, retrain
    Singles-only      Full GEARS trained without any combinatorial data

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_gears_ablation.py
"""

import os
import json
import pickle
import warnings
import argparse

import numpy as np
import torch
from scipy import stats

from data_utils import pair_genes

warnings.filterwarnings("ignore")
os.makedirs("results/gears_ablation", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    choices=[42, 43, 44],
    required=True,
    help="Run one Norman split seed",
)
args = parser.parse_args()

SEEDS = [args.seed]
EPOCHS = 15


###############################################################################
# Load data
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

comp_eligible = [p for p in pair_perts if all(g in single_gene_names for g in pair_genes(p))]

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


###############################################################################
# Graph ablation helper
###############################################################################

def ablate_graphs(gears_obj, remove_go=False, remove_coexpress=False):
    """
    Remove graph edges before training by replacing adjacency with
    self-loops and zeroing/freezing corresponding transformation layers.
    """
    m = gears_obj.model if hasattr(gears_obj, "model") else gears_obj
    changes = []

    tensor_info = {}
    for attr in sorted(dir(m)):
        if attr.startswith("_"):
            continue
        obj = getattr(m, attr, None)
        if isinstance(obj, torch.Tensor) and obj.numel() > 1:
            tensor_info[attr] = obj

    def classify(attr):
        lo = attr.lower()
        is_go = attr in ("G_sim", "G_go", "G_sim_weight", "G_go_weight") or (
            "sim" in lo and "coexpress" not in lo)
        is_coex = attr in ("G_coexpress", "G_co", "G_coexpress_weight") or "coexpress" in lo
        return is_go, is_coex

    replaced_edge_counts = {}
    for attr, obj in tensor_info.items():
        is_go, is_coex = classify(attr)
        should_remove = (is_go and remove_go) or (is_coex and remove_coexpress)
        if not should_remove:
            continue

        if obj.dim() == 2 and obj.shape[0] == obj.shape[1]:
            n = obj.shape[0]
            setattr(m, attr, torch.eye(n, device=obj.device, dtype=obj.dtype))
            changes.append(f"{attr}: adj({n}) -> identity")
        elif obj.dim() == 2 and obj.shape[0] == 2:
            old_e = obj.shape[1]
            n_nodes = int(obj.max().item()) + 1
            loops = torch.arange(n_nodes, device=obj.device).unsqueeze(0).repeat(2, 1)
            setattr(m, attr, loops)
            replaced_edge_counts[old_e] = n_nodes
            changes.append(f"{attr}: edges({old_e}) -> self-loops({n_nodes})")
        elif obj.dim() == 1 and obj.shape[0] in replaced_edge_counts:
            n_nodes = replaced_edge_counts[obj.shape[0]]
            setattr(m, attr, torch.ones(n_nodes, device=obj.device, dtype=obj.dtype))
            changes.append(f"{attr}: weights -> ones({n_nodes})")
        else:
            setattr(m, attr, torch.zeros_like(obj))
            changes.append(f"{attr}: -> zeros")

    # Pass 2 (deferred correction): a 1D weight tensor may have been visited
    # before its edge index, so it would have hit the zeros fallback above.
    # Now that all edge counts are known, fix any such weight vector to ones.
    for attr, obj in tensor_info.items():
        is_go, is_coex = classify(attr)
        should_remove = (is_go and remove_go) or (is_coex and remove_coexpress)
        if not should_remove:
            continue
        current = getattr(m, attr, None)
        if current is None:
            continue
        if current.data_ptr() == obj.data_ptr() and obj.dim() == 1 \
                and obj.shape[0] in replaced_edge_counts:
            n_nodes = replaced_edge_counts[obj.shape[0]]
            setattr(m, attr, torch.ones(n_nodes, device=obj.device, dtype=obj.dtype))
            changes.append(f"{attr}: weights({obj.shape[0]}) -> ones({n_nodes}) [deferred]")

    for layer_name, should_freeze in [
        ("emb_trans", remove_go), ("sim_layers", remove_go),
        ("emb_trans_v2", remove_coexpress), ("cross_gene_state", remove_coexpress),
    ]:
        layer = getattr(m, layer_name, None)
        if layer is not None and should_freeze and isinstance(layer, torch.nn.Module):
            with torch.no_grad():
                for p in layer.parameters():
                    p.zero_()
                    p.requires_grad = False
            changes.append(f"{layer_name}: zeroed+frozen")

    for c in changes:
        print(f"    {c}")
    return changes


###############################################################################
# Ablation loop
###############################################################################

CONFIGS = [
    ("full", "Full GEARS", False, False, True),
    ("no_go", "No GO graph", True, False, True),
    ("no_coexpress", "No co-expression", False, True, True),
    ("no_graphs", "No graphs", True, True, True),
    ("singles_only", "Singles-only GEARS", False, False, False),
]

all_records = []

for seed in SEEDS:
    print(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")

    split_path = f"results/splits/norman_seed{seed}.pkl"
    if not os.path.exists(split_path):
        print(f"  Split not found: {split_path}. Run prepare_data.py first.")
        continue

    with open(split_path, "rb") as f:
        split = pickle.load(f)
    test_pairs = split["test"]

    for config_id, config_name, rm_go, rm_coex, use_pairs in CONFIGS:
        print(f"\n  {config_name}")
        model_dir = f"results/gears_ablation/{config_id}_seed{seed}"

        if config_id == "singles_only":
            # Create split without combinatorial training data
            singles_split = {
                "train": [c for c in split["train"]
                          if c == "ctrl" or len(pair_genes(c)) == 1],
                "val": split["val"],
                "test": split["test"],
            }
            tmp_path = f"/tmp/singles_split_seed{seed}.pkl"
            with open(tmp_path, "wb") as f:
                pickle.dump(singles_split, f)
            pert_data.prepare_split(split="custom", seed=1, split_dict_path=tmp_path)
        else:
            pert_data.prepare_split(split="custom", seed=1, split_dict_path=split_path)

        pert_data.get_dataloader(batch_size=64, test_batch_size=256)
        gears_model = GEARS(pert_data, device=device)

        if os.path.exists(model_dir):
            gears_model.model_initialize(hidden_size=64)
            gears_model.load_pretrained(model_dir)
        elif config_id == "full" and (
                os.path.exists(f"results/norman/gears_seed{seed}")
                or os.path.exists(f"results/gears_seed{seed}")):
            # Reuse the SAME full GEARS model from Table 1 rather than
            # retraining, so Table 5's first row matches Table 1 exactly.
            full_dir = (f"results/norman/gears_seed{seed}"
                        if os.path.exists(f"results/norman/gears_seed{seed}")
                        else f"results/gears_seed{seed}")
            print(f"    Reusing Table 1 full GEARS from {full_dir}")
            gears_model.model_initialize(hidden_size=64)
            gears_model.load_pretrained(full_dir)
            gears_model.save_model(model_dir)
        else:
            gears_model.model_initialize(hidden_size=64)
            if rm_go or rm_coex:
                ablate_graphs(gears_model, remove_go=rm_go, remove_coexpress=rm_coex)
            gears_model.train(epochs=EPOCHS, lr=1e-3)
            gears_model.save_model(model_dir)

        def predict(pname):
            genes = pair_genes(pname)
            try:
                res = gears_model.predict([genes])
                pred = np.asarray(next(iter(res.values()))).reshape(-1)
                return pred - ctrl_mean
            except Exception:
                return None

        for p in test_pairs:
            genes = pair_genes(p)
            if not (len(genes) == 2 and genes[0] in single_effects
                    and genes[1] in single_effects and p in all_pair_effects):
                continue
            true_eff = all_pair_effects[p]
            add_pred = single_effects[genes[0]] + single_effects[genes[1]]
            pred = predict(p)
            if pred is None:
                continue
            top20 = np.argsort(np.abs(true_eff))[-20:]
            all_records.append({
                "seed": seed,
                "pair": p,
                "config": config_id,
                "error_model": float(np.mean(np.abs(pred - true_eff))),
                "error_additive": float(np.mean(np.abs(add_pred - true_eff))),
                "error_model_top20": float(np.mean(np.abs(pred[top20] - true_eff[top20]))),
                "ad": float(np.mean(np.abs(pred - add_pred))),
                "interaction_magnitude": float(np.mean(np.abs(true_eff - add_pred))),
            })


###############################################################################
# Results: Table 5
###############################################################################

print("\n" + "=" * 60)
print("TABLE 5: GEARS ablations")
print("=" * 60)

print(f"\n  {'Config':<25} {'MAE':>8} {'Wins/Add':>10}")
print(f"  {'~' * 45}")

for cid, cname, _, _, _ in CONFIGS:
    cr = [r for r in all_records if r["config"] == cid]
    if not cr:
        continue
    es = [r["error_model"] for r in cr]
    ae = [r["error_additive"] for r in cr]
    w = sum(1 for e, a in zip(es, ae) if e < a)
    print(f"  {cname:<25} {np.mean(es):>8.4f} {w:>3}/{len(cr)}")

ae = [r["error_additive"] for r in all_records if r["config"] == "full"]
print(f"  {'Additive':<25} {np.mean(ae):>8.4f}")

output_path = f"results/gears_ablation/records_seed{args.seed}.json"
with open(output_path, "w") as f:
    json.dump(all_records, f, indent=2)
print(f"\nSaved {output_path} ({len(all_records)} records)")