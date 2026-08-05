"""
Norman compositional OOD: prediction baselines and reliability analysis.

Produces:
    Table 1 (Norman columns): MAE, Top20 MAE, Wins/Add. for all 8 models
    Table 2: Nine reliability signals for GEARS
    Table 9: Selective prediction under abstention (Appendix C)
    Table 10: Interaction correction analysis across models

Models: Additive, Ridge, MLP, Transformer, CPA, scGPT, GEARS+scGPT, GEARS
Signals: AD, PM, GV, Ensemble, kNN, Mahalanobis, NN error, Cosine, Random

Usage:
    # Run all seeds sequentially
    CUDA_VISIBLE_DEVICES=0 python run_norman_baselines.py

    # Run one seed per GPU
    CUDA_VISIBLE_DEVICES=0 python run_norman_baselines.py --seed 42
    CUDA_VISIBLE_DEVICES=1 python run_norman_baselines.py --seed 43
    CUDA_VISIBLE_DEVICES=2 python run_norman_baselines.py --seed 44

    # Merge per-seed record files after all jobs finish
    python run_norman_baselines.py --merge-only
"""

import os
import json
import pickle
import warnings
import argparse

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.spatial.distance import mahalanobis as mahal_dist
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance

from data_utils import load_norman, pair_genes, PertDatasetML, train_pytorch, train_cpa
from models import PertMLP, PertTransformer, CompositionalPertVAE, ScGPTPertPredictor
from reliability import (
    compute_additive_disagreement,
    compute_prediction_magnitude,
    compute_genewise_variance,
    compute_knn_distance,
    compute_ensemble_variance,
    failure_auroc,
    spearman_correlation,
    abstention_improvement,
)

warnings.filterwarnings("ignore")
os.makedirs("results/norman", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

ALL_SEEDS = [42, 43, 44]

parser = argparse.ArgumentParser(description="Run Norman baselines and reliability analysis.")
parser.add_argument(
    "--seed",
    type=int,
    choices=ALL_SEEDS,
    default=None,
    help="Run only one seed. Use one seed per GPU for safe parallel execution.",
)
parser.add_argument(
    "--merge-only",
    action="store_true",
    help="Merge results/norman/records_seed*.json into all_records.json and exit.",
)
args = parser.parse_args()


def merge_seed_records():
    merged = []
    missing = []
    for seed in ALL_SEEDS:
        path = f"results/norman/records_seed{seed}.json"
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as f:
            merged.extend(json.load(f))

    if missing:
        raise FileNotFoundError(
            "Cannot merge because the following files are missing: " + ", ".join(missing)
        )

    merged.sort(key=lambda r: (int(r.get("seed", -1)), str(r.get("pair", ""))))
    out_path = "results/norman/all_records.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(merged)} records into {out_path}")


if args.merge_only:
    merge_seed_records()
    raise SystemExit(0)

SEEDS = [args.seed] if args.seed is not None else ALL_SEEDS
GEARS_EPOCHS = 15
MLP_EPOCHS = 200
CPA_EPOCHS = 300
N_ENSEMBLE = 3
SCGPT_DIR = "data/scgpt_pretrained"


###############################################################################
# 1. Load data
###############################################################################

print("=" * 60)
print("Loading Norman dataset")
print("=" * 60)

data = load_norman()
adata = data["adata"]
pert_data = data["pert_data"]
n_genes_out = data["n_genes_out"]
n_pert_genes = data["n_pert_genes"]
pert_gene2idx = data["pert_gene2idx"]
gene2idx = data["gene2idx"]
gene_names = data["gene_names"]
single_effects = data["single_effects"]
all_pair_effects = data["all_pair_effects"]
all_effects = data["all_effects"]
single_perts = data["single_perts"]
pair_perts = data["pair_perts"]
comp_eligible = data["comp_eligible"]
ctrl_mean = data["ctrl_mean"]

non_eligible = [p for p in pair_perts if p not in comp_eligible]
print(f"Singles: {len(single_perts)}, Pairs: {len(pair_perts)}, "
      f"Eligible: {len(comp_eligible)}")


###############################################################################
# 2. Load scGPT pretrained embeddings (if available)
###############################################################################

scgpt_embs = None
if os.path.exists(os.path.join(SCGPT_DIR, "vocab.json")):
    print("\nLoading scGPT pretrained embeddings...")
    with open(os.path.join(SCGPT_DIR, "vocab.json")) as f:
        scgpt_vocab = json.load(f)
    state = torch.load(os.path.join(SCGPT_DIR, "best_model.pt"),
                       map_location="cpu", weights_only=True)
    emb_key = None
    for key in sorted(state.keys()):
        shape = state[key].shape
        if len(shape) == 2 and shape[0] > 10000 and shape[1] >= 32:
            if "encoder" in key.lower() or emb_key is None:
                emb_key = key
    pretrained_emb = state[emb_key].numpy()

    vocab_lower = {k.lower(): v for k, v in scgpt_vocab.items()}
    scgpt_embs = np.zeros((n_pert_genes, pretrained_emb.shape[1]), dtype=np.float32)
    mapped = 0
    for g, idx in pert_gene2idx.items():
        vidx = scgpt_vocab.get(g, vocab_lower.get(g.lower()))
        if vidx is not None and vidx < pretrained_emb.shape[0]:
            scgpt_embs[idx] = pretrained_emb[vidx]
            mapped += 1
    print(f"  Mapped {mapped}/{n_pert_genes} genes")

    # Also build embeddings for ALL output genes, so GEARS+scGPT can inject
    # pretrained embeddings into every output-gene position (matching the
    # original run_scgpt_norman.py), not just the perturbation genes.
    scgpt_embs_out = np.zeros((n_genes_out, pretrained_emb.shape[1]), dtype=np.float32)
    mapped_out = 0
    for i, gn in enumerate(gene_names):
        vidx = scgpt_vocab.get(gn, vocab_lower.get(gn.lower()))
        if vidx is not None and vidx < pretrained_emb.shape[0]:
            scgpt_embs_out[i] = pretrained_emb[vidx]
            mapped_out += 1
    print(f"  Mapped {mapped_out}/{n_genes_out} output genes for GEARS+scGPT")
else:
    print("\nscGPT pretrained weights not found; skipping scGPT and GEARS+scGPT.")


###############################################################################
# 3. Main experiment loop
###############################################################################

from gears import GEARS

all_records = []

for seed in SEEDS:
    print(f"\n{'=' * 60}")
    print(f"SEED {seed}")
    print(f"{'=' * 60}")

    # Load or create split
    split_path = f"results/splits/norman_seed{seed}.pkl"
    if not os.path.exists(split_path):
        from data_utils import create_splits
        split = create_splits(comp_eligible, pair_perts, single_perts, seed)
        os.makedirs("results/splits", exist_ok=True)
        with open(split_path, "wb") as f:
            pickle.dump(split, f)
    else:
        with open(split_path, "rb") as f:
            split = pickle.load(f)

    train_set = [c for c in split["train"] if c != "ctrl"]
    val_pairs = split["val"]
    test_pairs = split["test"]

    rng = np.random.RandomState(seed)
    elig = list(comp_eligible)
    rng.shuffle(elig)
    n_t = max(1, int(len(elig) * 0.2))
    n_v = max(1, int(len(elig) * 0.2))
    train_eligible = elig[n_t + n_v:]
    train_pairs_all = train_eligible + non_eligible

    print(f"  Train: {len(train_set)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # ----- GEARS -----
    gears_dir = f"results/norman/gears_seed{seed}"
    pert_data.prepare_split(split="custom", seed=1, split_dict_path=split_path)
    pert_data.get_dataloader(batch_size=64, test_batch_size=256)
    gears_model = GEARS(pert_data, device=device)

    if os.path.exists(gears_dir):
        gears_model.model_initialize(hidden_size=64)
        gears_model.load_pretrained(gears_dir)
    else:
        gears_model.model_initialize(hidden_size=64)
        gears_model.train(epochs=GEARS_EPOCHS, lr=1e-3)
        gears_model.save_model(gears_dir)

    def predict_gears(pname):
        genes = pair_genes(pname)
        try:
            res = gears_model.predict([genes])
            pred = np.asarray(next(iter(res.values()))).reshape(-1)
            return pred - ctrl_mean
        except Exception:
            return None

    # GEARS gene embeddings for kNN
    trained_gears = getattr(gears_model, "best_model", gears_model.model)
    with torch.no_grad():
        gears_gene_emb = trained_gears.gene_emb.weight.cpu().numpy()

    def pert_embedding_gears(pname):
        genes = pair_genes(pname)
        embs = [gears_gene_emb[gene2idx[g]] for g in genes if g in gene2idx]
        return np.mean(embs, axis=0).astype(np.float32) if embs else None

    # kNN model for embedding-based signals
    train_pair_embs, train_pair_names = [], []
    for p in train_pairs_all:
        emb = pert_embedding_gears(p)
        if emb is not None and p in all_pair_effects:
            train_pair_embs.append(emb)
            train_pair_names.append(p)
    train_embs_arr = np.array(train_pair_embs) if train_pair_embs else np.zeros((1, 64))
    nn_model = NearestNeighbors(
        n_neighbors=min(5, len(train_embs_arr)), metric="cosine"
    ).fit(train_embs_arr)

    # Mahalanobis
    mahal_data = None
    if len(train_pair_embs) > 2:
        try:
            cov_est = EmpiricalCovariance().fit(train_embs_arr)
            train_mean = np.mean(train_embs_arr, axis=0)
            cov_inv = np.linalg.inv(
                cov_est.covariance_ + 1e-6 * np.eye(train_embs_arr.shape[1])
            )
            mahal_data = (train_mean, cov_inv)
        except Exception:
            pass

    # Ensemble GEARS
    ensemble_models = [gears_model]
    for ens_idx in range(1, N_ENSEMBLE):
        ens_dir = f"results/norman/gears_ens{ens_idx}_seed{seed}"
        pert_data.prepare_split(split="custom", seed=1, split_dict_path=split_path)
        pert_data.get_dataloader(batch_size=64, test_batch_size=256)
        ens = GEARS(pert_data, device=device)
        if os.path.exists(ens_dir):
            ens.model_initialize(hidden_size=64)
            ens.load_pretrained(ens_dir)
        else:
            ens.model_initialize(hidden_size=64)
            torch.manual_seed(seed * 1000 + ens_idx * 100)
            ens.train(epochs=GEARS_EPOCHS, lr=1e-3)
            ens.save_model(ens_dir)
        ensemble_models.append(ens)

    ensemble_preds = {}
    for p in val_pairs + test_pairs:
        preds = []
        for em in ensemble_models:
            genes = pair_genes(p)
            try:
                res = em.predict([genes])
                pred = np.asarray(next(iter(res.values()))).reshape(-1)
                preds.append(pred - ctrl_mean)
            except Exception:
                pass
        if preds:
            ensemble_preds[p] = preds

    # ----- MLP -----
    mlp_path = f"results/norman/mlp_seed{seed}.pt"
    torch.manual_seed(seed)
    mlp = PertMLP(n_pert_genes, n_genes_out).to(device)
    if os.path.exists(mlp_path):
        mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    else:
        train_ds = PertDatasetML(train_set, all_effects, pert_gene2idx)
        val_ds = PertDatasetML(val_pairs, all_effects, pert_gene2idx)
        mlp = train_pytorch(mlp, train_ds, val_ds, seed, device=device, epochs=MLP_EPOCHS)
        torch.save(mlp.state_dict(), mlp_path)
    mlp.eval()

    def predict_mlp(pname):
        genes = pair_genes(pname)
        if not all(g in pert_gene2idx for g in genes):
            return None
        idx = [pert_gene2idx[g] for g in genes]
        while len(idx) < 2:
            idx.append(-1)
        with torch.no_grad():
            return mlp(torch.tensor([idx[:2]], dtype=torch.long, device=device)).cpu().numpy().flatten()

    # ----- Transformer -----
    trans_path = f"results/norman/transformer_seed{seed}.pt"
    torch.manual_seed(seed)
    transformer = PertTransformer(n_pert_genes, n_genes_out).to(device)
    if os.path.exists(trans_path):
        transformer.load_state_dict(torch.load(trans_path, map_location=device, weights_only=True))
    else:
        train_ds = PertDatasetML(train_set, all_effects, pert_gene2idx)
        val_ds = PertDatasetML(val_pairs, all_effects, pert_gene2idx)
        transformer = train_pytorch(transformer, train_ds, val_ds, seed, device=device, epochs=MLP_EPOCHS)
        torch.save(transformer.state_dict(), trans_path)
    transformer.eval()

    def predict_transformer(pname):
        genes = pair_genes(pname)
        if not all(g in pert_gene2idx for g in genes):
            return None
        idx = [pert_gene2idx[g] for g in genes]
        while len(idx) < 2:
            idx.append(-1)
        with torch.no_grad():
            return transformer(torch.tensor([idx[:2]], dtype=torch.long, device=device)).cpu().numpy().flatten()

    # ----- CPA -----
    cpa_path = f"results/norman/cpa_seed{seed}.pt"
    torch.manual_seed(seed)
    cpa = CompositionalPertVAE(n_pert_genes, n_genes_out).to(device)
    if os.path.exists(cpa_path):
        cpa.load_state_dict(torch.load(cpa_path, map_location=device, weights_only=True))
    else:
        train_ds = PertDatasetML(train_set, all_effects, pert_gene2idx)
        val_ds = PertDatasetML(val_pairs, all_effects, pert_gene2idx)
        cpa = train_cpa(cpa, train_ds, val_ds, seed, device=device, epochs=CPA_EPOCHS)
        torch.save(cpa.state_dict(), cpa_path)
    cpa.eval()

    def predict_cpa(pname):
        genes = pair_genes(pname)
        if not all(g in pert_gene2idx for g in genes):
            return None
        idx = [pert_gene2idx[g] for g in genes]
        while len(idx) < 2:
            idx.append(-1)
        with torch.no_grad():
            return cpa(torch.tensor([idx[:2]], dtype=torch.long, device=device)).cpu().numpy().flatten()

    # ----- scGPT -----
    predict_scgpt = None
    if scgpt_embs is not None:
        scgpt_path = f"results/norman/scgpt_seed{seed}.pt"
        torch.manual_seed(seed)
        scgpt_model = ScGPTPertPredictor(n_pert_genes, n_genes_out, scgpt_embs).to(device)
        if os.path.exists(scgpt_path):
            scgpt_model.load_state_dict(torch.load(scgpt_path, map_location=device, weights_only=True))
        else:
            train_ds = PertDatasetML(train_set, all_effects, pert_gene2idx)
            val_ds = PertDatasetML(val_pairs, all_effects, pert_gene2idx)
            scgpt_model = train_pytorch(scgpt_model, train_ds, val_ds, seed,
                                        device=device, epochs=MLP_EPOCHS, lr=5e-4,
                                        use_adamw=True, weight_decay=1e-4)
            torch.save(scgpt_model.state_dict(), scgpt_path)
        scgpt_model.eval()

        def predict_scgpt(pname):
            genes = pair_genes(pname)
            if not all(g in pert_gene2idx for g in genes):
                return None
            idx = [pert_gene2idx[g] for g in genes]
            while len(idx) < 2:
                idx.append(-1)
            with torch.no_grad():
                return scgpt_model(torch.tensor([idx[:2]], dtype=torch.long, device=device)).cpu().numpy().flatten()

    # ----- GEARS+scGPT -----
    predict_gears_scgpt = None
    if scgpt_embs is not None:
        from sklearn.decomposition import PCA
        gs_dir = f"results/norman/gears_scgpt_seed{seed}"
        # Initialize GEARS with scGPT embeddings projected via PCA
        pert_data.prepare_split(split="custom", seed=1, split_dict_path=split_path)
        pert_data.get_dataloader(batch_size=64, test_batch_size=256)
        gs_model = GEARS(pert_data, device=device)
        if os.path.exists(gs_dir):
            gs_model.model_initialize(hidden_size=64)
            gs_model.load_pretrained(gs_dir)
        else:
            gs_model.model_initialize(hidden_size=64)
            # Inject scGPT embeddings for ALL output genes (matching the
            # original run_scgpt_norman.py): PCA the full output-gene
            # embedding matrix to the GEARS embedding width, then copy into
            # each output-gene position of gene_emb.
            gears_net = gs_model.model
            model_emb_dim = gears_net.gene_emb.weight.shape[1]
            if scgpt_embs_out.shape[1] != model_emb_dim:
                proj_out = PCA(n_components=model_emb_dim,
                               random_state=seed).fit_transform(scgpt_embs_out)
            else:
                proj_out = scgpt_embs_out
            with torch.no_grad():
                gene_emb_w = gears_net.gene_emb.weight
                n_copy = min(gene_emb_w.shape[0], proj_out.shape[0])
                gene_emb_w[:n_copy] = torch.tensor(
                    proj_out[:n_copy], dtype=torch.float32, device=gene_emb_w.device)
            gs_model.train(epochs=GEARS_EPOCHS, lr=1e-3)
            gs_model.save_model(gs_dir)

        def predict_gears_scgpt(pname):
            genes = pair_genes(pname)
            try:
                res = gs_model.predict([genes])
                pred = np.asarray(next(iter(res.values()))).reshape(-1)
                return pred - ctrl_mean
            except Exception:
                return None

    # ----- Ridge -----
    def pair_to_onehot(pname):
        genes = pair_genes(pname)
        vec = np.zeros(n_pert_genes, dtype=np.float32)
        for g in genes:
            if g in pert_gene2idx:
                vec[pert_gene2idx[g]] = 1.0
        return vec

    X_train, Y_train = [], []
    for p in train_set:
        if p not in all_effects:
            continue
        X_train.append(pair_to_onehot(p))
        Y_train.append(all_effects[p])
    ridge = Ridge(alpha=1.0).fit(np.array(X_train), np.array(Y_train))

    def predict_ridge(pname):
        return ridge.predict(pair_to_onehot(pname).reshape(1, -1)).flatten()

    # ----- NN error proxy setup -----
    proxy_errors = []
    for p in train_pair_names:
        gp = predict_gears(p)
        if gp is not None and p in all_pair_effects:
            proxy_errors.append(float(np.mean(np.abs(gp - all_pair_effects[p]))))
        else:
            proxy_errors.append(0.0)
    proxy_errors_arr = np.array(proxy_errors)
    nn_proxy = NearestNeighbors(
        n_neighbors=min(3, len(train_embs_arr)), metric="cosine"
    ).fit(train_embs_arr)

    # ----- Per-model embedding kNN structures (Table 4 routing) -----
    # Each learned model gets its own embedding space, matching the original
    # (Transformer/CPA/scGPT/MLP embedding kNN) rather than reusing the GEARS
    # embedding for every model.
    def _torch_embed(model, pname):
        genes = pair_genes(pname)
        if not all(g in pert_gene2idx for g in genes):
            return None
        idx = [pert_gene2idx[g] for g in genes]
        while len(idx) < 2:
            idx.append(-1)
        gi = torch.tensor([idx[:2]], dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            t = gi.clone(); t[t < 0] = model.pad_idx
            if hasattr(model, "get_embedding"):          # Transformer
                return model.get_embedding(gi).cpu().numpy().flatten()
            if hasattr(model, "get_pert_embedding"):     # CPA
                return model.get_pert_embedding(gi).cpu().numpy().flatten()
            if hasattr(model, "pretrained_emb"):         # scGPT
                raw = model.pretrained_emb[t]
                proj = model.emb_proj(raw)
                comb = proj + model.pert_emb(t) + model.pos_emb.unsqueeze(0)
                pad = (gi < 0)
                out = model.transformer(comb, src_key_padding_mask=pad)
                mf = (~pad).unsqueeze(-1).float()
                return ((out * mf).sum(1) / mf.sum(1).clamp(min=1)).cpu().numpy().flatten()
            # MLP: mean-pooled embedding
            embs = model.emb(t)
            mask = (gi >= 0).unsqueeze(-1).float()
            return (embs * mask).sum(1).cpu().numpy().flatten()

    model_handles = {"mlp": mlp, "transformer": transformer, "cpa": cpa}
    if predict_scgpt is not None:
        model_handles["scgpt"] = scgpt_model

    per_model_nn = {}
    for mkey, mdl in model_handles.items():
        embs = []
        for p in train_pair_names:
            e = _torch_embed(mdl, p)
            if e is not None:
                embs.append(e)
        if len(embs) >= 2:
            arr = np.array(embs)
            per_model_nn[mkey] = (mdl, NearestNeighbors(
                n_neighbors=min(5, len(arr)), metric="cosine").fit(arr))

    # GEARS+scGPT gets its OWN gene-embedding kNN (not the standard GEARS one),
    # matching the original which reads the trained gs_model's gene_emb.
    gs_pair_emb = None
    gs_nn = None
    if predict_gears_scgpt is not None:
        try:
            gs_gene_emb = gs_model.model.gene_emb.weight.data.cpu().numpy()

            def gs_pair_emb(pname):
                genes = pair_genes(pname)
                es = [gs_gene_emb[gene2idx[g]] for g in genes if g in gene2idx]
                return np.mean(es, axis=0).astype(np.float32) if es else None

            gs_train_embs = []
            for p in train_pair_names:
                e = gs_pair_emb(p)
                if e is not None:
                    gs_train_embs.append(e)
            if len(gs_train_embs) >= 2:
                gs_nn = NearestNeighbors(
                    n_neighbors=min(5, len(gs_train_embs)), metric="cosine"
                ).fit(np.array(gs_train_embs))
        except Exception:
            gs_pair_emb = None
            gs_nn = None

    # =====================================================================
    # Collect records for test pairs
    # =====================================================================

    for p in test_pairs:
        genes = pair_genes(p)
        if not (len(genes) == 2 and genes[0] in single_effects
                and genes[1] in single_effects and p in all_pair_effects):
            continue

        true_eff = all_pair_effects[p]
        add_pred = single_effects[genes[0]] + single_effects[genes[1]]
        top20 = np.argsort(np.abs(true_eff))[-20:]

        gears_pred = predict_gears(p)
        mlp_pred = predict_mlp(p)
        trans_pred = predict_transformer(p)
        cpa_pred = predict_cpa(p)
        ridge_pred = predict_ridge(p)
        scgpt_pred = predict_scgpt(p) if predict_scgpt else None
        gs_pred = predict_gears_scgpt(p) if predict_gears_scgpt else None

        record = {
            "seed": seed,
            "pair": p,
            "interaction_magnitude": float(np.mean(np.abs(true_eff - add_pred))),
        }

        # Prediction errors and top20
        predictors = [
            ("additive", add_pred),
            ("ridge", ridge_pred),
            ("mlp", mlp_pred),
            ("transformer", trans_pred),
            ("cpa", cpa_pred),
            ("scgpt", scgpt_pred),
            ("gears_scgpt", gs_pred),
            ("gears", gears_pred),
        ]

        for name, pred in predictors:
            if pred is not None:
                record[f"error_{name}"] = float(np.mean(np.abs(pred - true_eff)))
                record[f"top20_{name}"] = float(np.mean(np.abs(pred[top20] - true_eff[top20])))
                record[f"ad_{name}"] = float(np.mean(np.abs(pred - add_pred)))
                record[f"pm_{name}"] = float(np.mean(np.abs(pred)))
                record[f"gv_{name}"] = float(np.var(pred))

        # Embedding-based signals (GEARS embedding for the shared kNN/Mahal)
        emb = pert_embedding_gears(p)
        if emb is not None:
            record["knn_dist"] = float(nn_model.kneighbors(emb.reshape(1, -1))[0].mean())
            record["knn_gears"] = record["knn_dist"]
            if mahal_data:
                try:
                    record["mahalanobis"] = float(mahal_dist(emb, mahal_data[0], mahal_data[1]))
                except Exception:
                    pass

            # NN error proxy (computed once, in GEARS embedding space).
            dists_p, idxs_p = nn_proxy.kneighbors(emb.reshape(1, -1))
            w = 1.0 / (dists_p[0] + 1e-8)
            w = w / w.sum()
            record["nn_error_proxy"] = float(np.sum(w * proxy_errors_arr[idxs_p[0]]))

        # Per-model embedding kNN (Transformer/CPA/scGPT/MLP own spaces).
        for mkey, (mdl, nn_m) in per_model_nn.items():
            e = _torch_embed(mdl, p)
            if e is not None:
                record[f"knn_{mkey}"] = float(nn_m.kneighbors(e.reshape(1, -1))[0].mean())

        # GEARS+scGPT kNN in its own gene-embedding space.
        if gs_nn is not None and gs_pair_emb is not None:
            ge = gs_pair_emb(p)
            if ge is not None:
                record["knn_gears_scgpt"] = float(
                    gs_nn.kneighbors(ge.reshape(1, -1))[0].mean())

        # Ensemble variance
        if p in ensemble_preds and len(ensemble_preds[p]) >= 2:
            record["ensemble_var"] = compute_ensemble_variance(ensemble_preds[p])

        # Cosine to nearest
        if gears_pred is not None:
            best_cos = float("inf")
            for tp in train_pair_names[:50]:
                if tp in all_pair_effects:
                    tp_pred = predict_gears(tp)
                    if tp_pred is not None:
                        cs = np.dot(gears_pred, tp_pred) / (
                            np.linalg.norm(gears_pred) * np.linalg.norm(tp_pred) + 1e-8)
                        if 1.0 - cs < best_cos:
                            best_cos = 1.0 - cs
            if best_cos < float("inf"):
                record["cosine_to_nearest"] = float(best_cos)

        record["random"] = float(rng.rand())

        all_records.append(record)

    seed_records = [r for r in all_records if r["seed"] == seed]
    seed_records_path = f"results/norman/records_seed{seed}.json"
    with open(seed_records_path, "w") as f:
        json.dump(seed_records, f, indent=2)
    print(f"  Records: {len(seed_records)}")
    print(f"  Saved: {seed_records_path}")


###############################################################################
# 4. Print results
###############################################################################

print("\n" + "=" * 60)
print("TABLE 1: Prediction performance (Norman)")
print("=" * 60)

model_keys = [
    ("Additive", "additive"),
    ("Ridge", "ridge"),
    ("MLP", "mlp"),
    ("Transformer", "transformer"),
    ("CPA", "cpa"),
    ("scGPT", "scgpt"),
    ("GEARS+scGPT", "gears_scgpt"),
    ("GEARS", "gears"),
]

print(f"\n  {'Model':<18} {'MAE':>8} {'Top20':>8} {'Wins/Add':>10}")
print(f"  {'~' * 48}")

for name, key in model_keys:
    valid = [r for r in all_records if f"error_{key}" in r]
    if not valid:
        continue
    errs = [r[f"error_{key}"] for r in valid]
    t20 = [r[f"top20_{key}"] for r in valid if f"top20_{key}" in r]
    add_errs = [r["error_additive"] for r in valid]
    wins = sum(1 for e, a in zip(errs, add_errs) if e < a) if key != "additive" else "n/a"
    total = len(valid)
    t20_str = f"{np.mean(t20):.4f}" if t20 else "n/a"
    w_str = f"{wins}/{total}" if isinstance(wins, int) else wins
    print(f"  {name:<18} {np.mean(errs):>8.4f} {t20_str:>8} {w_str:>10}")


print("\n" + "=" * 60)
print("TABLE 2: Reliability signals for GEARS")
print("=" * 60)

valid_gears = [r for r in all_records if "error_gears" in r]
if valid_gears:
    gears_errors = np.array([r["error_gears"] for r in valid_gears])

    signals = [
        ("Additive disagreement", "ad_gears"),
        ("Prediction magnitude", "pm_gears"),
        ("Gene-wise variance", "gv_gears"),
        ("Ensemble variance", "ensemble_var"),
        ("NN error proxy", "nn_error_proxy"),
        ("kNN distance", "knn_dist"),
        ("Mahalanobis distance", "mahalanobis"),
        ("Cosine to nearest", "cosine_to_nearest"),
        ("Random", "random"),
    ]

    for pctl_label, pctl in [("Top50%", 50), ("Top25%", 75), ("Top10%", 90)]:
        thresh = np.percentile(gears_errors, 100 - int(pctl_label[3:-1]))
        y_bin = (gears_errors > thresh).astype(int)

        print(f"\n  Failure AUROC ({pctl_label}):")
        print(f"  {'Signal':<28} {'AUROC':>7} {'Spearman':>10}")
        for sname, skey in signals:
            scores = [r.get(skey) for r in valid_gears]
            mask = [s is not None for s in scores]
            if sum(mask) < 10:
                continue
            s = np.array([scores[i] for i in range(len(scores)) if mask[i]])
            y = y_bin[np.array(mask)]
            e = gears_errors[np.array(mask)]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            auroc = roc_auc_score(y, s)
            rho = stats.spearmanr(s, e)[0]
            print(f"  {sname:<28} {auroc:>7.3f} {rho:>10.3f}")


print("\n" + "=" * 60)
print("TABLE 9: Selective prediction under abstention (GEARS)")
print("=" * 60)

if valid_gears:
    abs_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"\n  {'Signal':<28}", end="")
    for ar in abs_rates:
        print(f" {'Abs' + str(int(ar * 100)) + '%':>10}", end="")
    print()

    for sname, skey in signals:
        valid = [r for r in valid_gears if r.get(skey) is not None]
        if len(valid) < 10:
            continue
        sorted_v = sorted(valid, key=lambda r: r[skey])
        full_mae = np.mean([r["error_gears"] for r in sorted_v])

        print(f"  {sname:<28}", end="")
        for ar in abs_rates:
            n_keep = max(1, int((1 - ar) * len(sorted_v)))
            kept = sorted_v[:n_keep]
            mae = np.mean([r["error_gears"] for r in kept])
            imp = (1 - mae / full_mae) * 100
            print(f" {mae:.4f}({imp:+.0f}%)", end="")
        print()


print("\n" + "=" * 60)
print("TABLE 10: Interaction correction analysis")
print("=" * 60)

print(f"\n  {'Model':<18} {'Mean AD':>8} {'Median AD':>10} {'90th':>8} {'Corr':>8}")
for name, key in model_keys:
    if key == "additive":
        print(f"  {'Additive':<18} {'0':>8} {'0':>10} {'0':>8} {'n/a':>8}")
        continue
    valid = [r for r in all_records if f"ad_{key}" in r and f"error_{key}" in r]
    if not valid:
        continue
    ads = np.array([r[f"ad_{key}"] for r in valid])
    errs = np.array([r[f"error_{key}"] for r in valid])
    rho = stats.spearmanr(ads, errs)[0]
    print(f"  {name:<18} {np.mean(ads):>8.4f} {np.median(ads):>10.4f} "
          f"{np.percentile(ads, 90):>8.4f} {rho:>8.3f}")


###############################################################################
# 5. Save
###############################################################################

if args.seed is None:
    # Sequential all-seed run: canonical combined output is safe to write directly.
    all_records.sort(key=lambda r: (int(r.get("seed", -1)), str(r.get("pair", ""))))
    with open("results/norman/all_records.json", "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved results/norman/all_records.json ({len(all_records)} records)")
else:
    print(
        f"\nSeed {args.seed} completed. After all three seed jobs finish, run:\n"
        "  /root/miniconda3/envs/vcell/bin/python run_norman_baselines.py --merge-only"
    )
print("DONE")