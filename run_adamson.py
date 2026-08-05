import os
import json
import pickle
import warnings
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from data_utils import pair_genes
from models import PertMLP, PertTransformer, CompositionalPertVAE, ScGPTPertPredictor
from reliability import abstention_improvement

warnings.filterwarnings("ignore")
os.makedirs("results/adamson", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [42, 43, 44]
EPOCHS = 200
SCGPT_DIR = "data/scgpt_pretrained"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seeds", type=int, nargs="+", default=SEEDS,
    help="Seed(s) to run, e.g. --seeds 42 43"
)
parser.add_argument(
    "--merge-only", action="store_true",
    help="Merge per-seed shard JSON files into results/adamson/all_records.json"
)
args = parser.parse_args()

requested_seeds = list(dict.fromkeys(args.seeds))
invalid_seeds = [s for s in requested_seeds if s not in SEEDS]
if invalid_seeds:
    parser.error(f"Unsupported seed(s): {invalid_seeds}; choose from {SEEDS}")

if args.merge_only:
    shard_paths = sorted(glob.glob("results/adamson/records_seeds_*.json"))
    merged = {}
    for shard_path in shard_paths:
        with open(shard_path, "r") as f:
            for record in json.load(f):
                key = (int(record["seed"]), str(record["condition"]))
                merged[key] = record

    all_records = [merged[k] for k in sorted(merged)]
    os.makedirs("results/adamson", exist_ok=True)
    with open("results/adamson/all_records.json", "w") as f:
        json.dump(all_records, f, indent=2)

    present_seeds = sorted({int(r["seed"]) for r in all_records})
    missing_seeds = [s for s in SEEDS if s not in present_seeds]
    print(f"Merged {len(shard_paths)} shard file(s): {len(all_records)} records")
    print(f"Present seeds: {present_seeds}")
    if missing_seeds:
        print(f"WARNING: missing seeds: {missing_seeds}")
    else:
        print("All seeds are present.")
    print("Saved results/adamson/all_records.json")
    raise SystemExit(0)

if requested_seeds == SEEDS:
    output_path = "results/adamson/all_records.json"
else:
    seed_tag = "_".join(str(s) for s in requested_seeds)
    output_path = f"results/adamson/records_seeds_{seed_tag}.json"


###############################################################################
# Load Adamson data
###############################################################################

print("=" * 60)
print("Loading Adamson dataset")
print("=" * 60)

from gears import PertData, GEARS

pert_data = PertData("./data")
pert_data.load(data_name="adamson")
adata = pert_data.adata
n_genes_out = adata.n_vars
print(f"Cells: {adata.n_obs}, Genes: {n_genes_out}")

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


conditions = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
all_effects = {}
for c in conditions:
    eff = get_effect(c)
    if eff is not None:
        all_effects[c] = eff

all_genes = sorted(set(g for c in all_effects for g in pair_genes(c)))
pg2idx = {g: i for i, g in enumerate(all_genes)}
n_pg = len(all_genes)

gene_names_full = (
    list(adata.var["gene_name"]) if "gene_name" in adata.var.columns
    else list(adata.var_names)
)
gene2idx_full = {g: i for i, g in enumerate(gene_names_full)}
print(f"Conditions: {len(conditions)}, Perturbation genes: {n_pg}")


class PertDS(Dataset):
    def __init__(self, conds, effs, pg2i):
        self.items = []
        for c in conds:
            if c not in effs:
                continue
            gs = pair_genes(c)
            if not gs or not all(g in pg2i for g in gs):
                continue
            idx = [pg2i[g] for g in gs]
            while len(idx) < 2:
                idx.append(-1)
            self.items.append((np.array(idx[:2], np.int64), effs[c].astype(np.float32)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


###############################################################################
# Load scGPT pretrained embeddings (optional)
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
    scgpt_embs = np.zeros((n_pg, pretrained_emb.shape[1]), dtype=np.float32)
    mapped = 0
    for g, idx in pg2idx.items():
        vidx = scgpt_vocab.get(g, vocab_lower.get(g.lower()))
        if vidx is not None and vidx < pretrained_emb.shape[0]:
            scgpt_embs[idx] = pretrained_emb[vidx]
            mapped += 1
    print(f"  Mapped {mapped}/{n_pg} perturbation genes")
else:
    print("\nscGPT weights not found; skipping scGPT.")


###############################################################################
# Embedding extraction + reliability signal helpers
###############################################################################

from scipy.spatial.distance import cdist
from sklearn.covariance import EmpiricalCovariance

N_MC_SAMPLES = 20       
N_NEIGHBORS = 5         


def embed_mlp(model, gi):
    model.eval()
    with torch.no_grad():
        idx = gi.clone(); idx[idx < 0] = model.pad_idx
        embs = model.emb(idx)
        mask = (gi >= 0).unsqueeze(-1).float()
        return (embs * mask).sum(dim=1).cpu().numpy()


def embed_transformer(model, gi):
    return model.get_embedding(gi).cpu().numpy()


def embed_cpa(model, gi):
    model.eval()
    with torch.no_grad():
        return model.get_pert_embedding(gi).cpu().numpy()


def embed_scgpt(model, gi):
    model.eval()
    with torch.no_grad():
        idx = gi.clone(); idx[idx < 0] = model.pad_idx
        raw = model.pretrained_emb[idx]
        proj = model.emb_proj(raw)
        combined = proj + model.pert_emb(idx) + model.pos_emb.unsqueeze(0)
        pad = (gi < 0)
        out = model.transformer(combined, src_key_padding_mask=pad)
        mf = (~pad).unsqueeze(-1).float()
        return ((out * mf).sum(1) / mf.sum(1).clamp(min=1)).cpu().numpy()


def compute_mahalanobis(test_emb, train_embs):
    try:
        cov = EmpiricalCovariance().fit(train_embs)
        return float(cov.mahalanobis(test_emb.reshape(1, -1))[0])
    except Exception:
        return None


def compute_cosine_to_nearest(test_emb, train_embs):
    """Cosine similarity to the nearest training embedding (higher = less OOD)."""
    try:
        sims = 1.0 - cdist(test_emb.reshape(1, -1), train_embs, metric="cosine").flatten()
        return float(np.max(sims))
    except Exception:
        return None


def compute_knn_dist(test_emb, nn_model):
    try:
        d, _ = nn_model.kneighbors(test_emb.reshape(1, -1))
        return float(d.mean())
    except Exception:
        return None


def compute_nn_error_proxy(test_emb, train_embs, train_errors, k=N_NEIGHBORS):
    """Weighted error of the k nearest training neighbors (cosine-distance weights)."""
    try:
        dists = cdist(test_emb.reshape(1, -1), train_embs, metric="cosine").flatten()
        order = np.argsort(dists)[:min(k, len(dists))]
        w = 1.0 / (dists[order] + 1e-8)
        w = w / w.sum()
        return float(np.sum(w * np.asarray(train_errors)[order]))
    except Exception:
        return None


def compute_ensemble_variance(mc_preds):
    """Mean per-gene standard deviation across MC-dropout forward passes."""
    if len(mc_preds) < 2:
        return None
    return float(np.mean(np.std(np.array(mc_preds), axis=0)))


def mc_dropout_preds(model, gi, n=N_MC_SAMPLES):
    """Run n stochastic forward passes with dropout enabled."""
    was_training = model.training
    model.train() 
    preds = []
    with torch.no_grad():
        for _ in range(n):
            preds.append(model(gi).cpu().numpy().flatten())
    if not was_training:
        model.eval()
    return preds


def build_train_embeddings(embed_fn, model, train_ds):
    """Return (train_embs, train_errors) for a model over the training set."""
    embs, errs = [], []
    for gi_np, tgt_np in train_ds.items:
        gi = torch.tensor([gi_np], dtype=torch.long, device=device)
        e = embed_fn(model, gi)[0]
        with torch.no_grad():
            pred = model(gi).cpu().numpy().flatten()
        embs.append(e)
        errs.append(float(np.mean(np.abs(pred - tgt_np))))
    return (np.array(embs) if embs else np.zeros((0, 1))), np.array(errs)


###############################################################################
# Main loop
###############################################################################

if os.path.exists(output_path):
    with open(output_path, "r") as f:
        all_records = json.load(f)
    print(f"Resuming from {output_path}: {len(all_records)} records")
else:
    all_records = []

completed_seeds = {int(r["seed"]) for r in all_records}

for seed in requested_seeds:
    if seed in completed_seeds:
        print(f"Skipping completed seed {seed}")
        continue
    print(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")

    split_path = f"results/splits/adamson_seed{seed}.pkl"
    if not os.path.exists(split_path):
        print(f"  Not found: {split_path}. Run prepare_data.py first.")
        continue

    with open(split_path, "rb") as f:
        split = pickle.load(f)

    train_conds = split["train"]
    test_conds = split["test"]
    val_conds = split.get("val", [])
    held_out_genes = set(split.get("test_genes", split.get("held_out_genes", [])))

    print(f"  Train: {len(train_conds)}, Val: {len(val_conds)}, "
          f"Test: {len(test_conds)}, Held-out test genes: {len(held_out_genes)}")

    train_ds = PertDS(train_conds, all_effects, pg2idx)
    val_ds = PertDS(val_conds, all_effects, pg2idx)
    if len(val_ds) == 0:
        val_ds = PertDS(train_conds[-20:], all_effects, pg2idx)

    # MLP
    torch.manual_seed(seed)
    mlp = PertMLP(n_pg, n_genes_out).to(device)
    from data_utils import train_pytorch
    mlp = train_pytorch(mlp, train_ds, val_ds, seed, device=device, epochs=EPOCHS)
    mlp.eval()

    # Transformer
    torch.manual_seed(seed)
    trans = PertTransformer(n_pg, n_genes_out).to(device)
    trans = train_pytorch(trans, train_ds, val_ds, seed, device=device, epochs=EPOCHS)
    trans.eval()

    # CPA
    torch.manual_seed(seed)
    cpa_model = CompositionalPertVAE(n_pg, n_genes_out).to(device)
    from data_utils import train_cpa
    cpa_model = train_cpa(cpa_model, train_ds, val_ds, seed, device=device)
    cpa_model.eval()

    # scGPT (fine-tuned)
    scgpt_model = None
    if scgpt_embs is not None:
        torch.manual_seed(seed)
        scgpt_model = ScGPTPertPredictor(n_pg, n_genes_out, scgpt_embs).to(device)
        scgpt_opt = torch.optim.AdamW(scgpt_model.parameters(), lr=5e-4, weight_decay=1e-4)
        scgpt_sch = torch.optim.lr_scheduler.CosineAnnealingLR(scgpt_opt, EPOCHS)
        tl = DataLoader(train_ds, 32, shuffle=True)
        vl = DataLoader(val_ds, 128)
        bv, bs, w = float("inf"), None, 0
        for ep in range(1, EPOCHS + 1):
            scgpt_model.train()
            for gi, tgt in tl:
                loss = nn.functional.mse_loss(scgpt_model(gi.to(device)), tgt.to(device))
                scgpt_opt.zero_grad(); loss.backward(); scgpt_opt.step()
            scgpt_sch.step()
            scgpt_model.eval(); vv, nv = 0, 0
            with torch.no_grad():
                for gi, tgt in vl:
                    vv += nn.functional.mse_loss(scgpt_model(gi.to(device)), tgt.to(device)).item()
                    nv += 1
            avg = vv / max(nv, 1)
            if avg < bv:
                bv = avg; bs = {k: v.clone() for k, v in scgpt_model.state_dict().items()}; w = 0
            else:
                w += 1
            if w >= 30:
                break
        if bs:
            scgpt_model.load_state_dict(bs)
        scgpt_model.eval()

    gears_split = {"train": train_conds, "val": val_conds, "test": test_conds}
    tmp_path = f"/tmp/adamson_gears_seed{seed}.pkl"
    with open(tmp_path, "wb") as f:
        pickle.dump(gears_split, f)
    pert_data.prepare_split(split="custom", seed=1, split_dict_path=tmp_path)
    pert_data.get_dataloader(batch_size=64, test_batch_size=256)
    gears_model = GEARS(pert_data, device=device)
    gears_dir = f"results/adamson/gears_seed{seed}"
    if os.path.exists(gears_dir):
        gears_model.model_initialize(hidden_size=64)
        gears_model.load_pretrained(gears_dir)
    else:
        gears_model.model_initialize(hidden_size=64)
        gears_model.train(epochs=15, lr=1e-3)
        gears_model.save_model(gears_dir)

    print("  Building reliability structures (kNN/Mahalanobis/NN-error)...")
    train_structs = {}
    for mkey, efn, mdl in [
        ("mlp", embed_mlp, mlp),
        ("transformer", embed_transformer, trans),
        ("cpa", embed_cpa, cpa_model),
    ] + ([("scgpt", embed_scgpt, scgpt_model)] if scgpt_model is not None else []):
        t_embs, t_errs = build_train_embeddings(efn, mdl, train_ds)
        nn_model = None
        if len(t_embs) >= 2:
            nn_model = NearestNeighbors(
                n_neighbors=min(N_NEIGHBORS, len(t_embs)), metric="cosine"
            ).fit(t_embs)
        train_structs[mkey] = (efn, mdl, t_embs, t_errs, nn_model)

    # GEARS embedding structure (gene-embedding space).
    gears_gene_emb = None
    try:
        gears_gene_emb = gears_model.model.gene_emb.weight.data.cpu().numpy()
    except Exception:
        gears_gene_emb = None

    def gears_pair_emb(genes):
        if gears_gene_emb is None:
            return None
        embs = [gears_gene_emb[gene2idx_full[g]] for g in genes
                if g in gene2idx_full and gene2idx_full[g] < gears_gene_emb.shape[0]]
        return np.mean(embs, axis=0) if embs else None

    print("  Computing GEARS training errors for NN error proxy...")
    gears_train_errors = {}
    for c in train_conds:
        if c == "ctrl" or c not in all_effects:
            continue
        try:
            res = gears_model.predict([pair_genes(c)])
            pr = np.asarray(next(iter(res.values()))).reshape(-1)
            if pr.shape[0] == n_genes_out:
                gears_train_errors[c] = float(np.mean(np.abs((pr - ctrl_mean) - all_effects[c])))
        except Exception:
            pass

    gears_train_embs, gears_train_errs = [], []
    for c in train_conds:
        if c == "ctrl" or c not in all_effects:
            continue
        e = gears_pair_emb(pair_genes(c))
        if e is not None:
            gears_train_embs.append(e)
            gears_train_errs.append(gears_train_errors.get(c, 0.0))
    gears_train_embs = np.array(gears_train_embs) if gears_train_embs else np.zeros((0, 1))
    gears_train_errs = np.array(gears_train_errs)
    gears_nn = None
    if len(gears_train_embs) >= 2:
        gears_nn = NearestNeighbors(
            n_neighbors=min(N_NEIGHBORS, len(gears_train_embs)), metric="cosine"
        ).fit(gears_train_embs)

    # Seed the random signal per seed (matching the original's np.random.seed).
    np.random.seed(seed)

    # Evaluate
    for c in test_conds:
        if c not in all_effects:
            continue
        true_eff = all_effects[c]
        genes = pair_genes(c)
        if not genes:
            continue

        idx = [pg2idx[g] for g in genes if g in pg2idx]
        while len(idx) < 2:
            idx.append(-1)
        gi = torch.tensor([idx[:2]], dtype=torch.long, device=device)

        record = {"seed": seed, "condition": c, "random": float(np.random.rand())}

        def add_reliability(mkey, pred):
            """Attach PM, GV, and the embedding-based signals for a model."""
            record[f"pm_{mkey}"] = float(np.mean(np.abs(pred)))
            record[f"gv_{mkey}"] = float(np.var(pred))
            if mkey not in train_structs:
                return
            efn, mdl, t_embs, t_errs, nn_model = train_structs[mkey]
            if len(t_embs) < 2:
                return
            test_emb = efn(mdl, gi)[0]
            record[f"knn_{mkey}"] = compute_knn_dist(test_emb, nn_model)
            record[f"mahalanobis_{mkey}"] = compute_mahalanobis(test_emb, t_embs)
            record[f"cosine_nearest_{mkey}"] = compute_cosine_to_nearest(test_emb, t_embs)
            record[f"nn_error_proxy_{mkey}"] = compute_nn_error_proxy(
                test_emb, t_embs, t_errs)
            record[f"ensemble_var_{mkey}"] = compute_ensemble_variance(
                mc_dropout_preds(mdl, gi))

        # MLP
        with torch.no_grad():
            mp = mlp(gi).cpu().numpy().flatten()
        record["error_mlp"] = float(np.mean(np.abs(mp - true_eff)))
        add_reliability("mlp", mp)

        # Transformer
        with torch.no_grad():
            tp = trans(gi).cpu().numpy().flatten()
        record["error_transformer"] = float(np.mean(np.abs(tp - true_eff)))
        add_reliability("transformer", tp)

        # CPA
        with torch.no_grad():
            cp = cpa_model(gi).cpu().numpy().flatten()
        record["error_cpa"] = float(np.mean(np.abs(cp - true_eff)))
        add_reliability("cpa", cp)

        # scGPT
        if scgpt_model is not None:
            with torch.no_grad():
                sp = scgpt_model(gi).cpu().numpy().flatten()
            record["error_scgpt"] = float(np.mean(np.abs(sp - true_eff)))
            add_reliability("scgpt", sp)

        # GEARS
        try:
            res = gears_model.predict([genes])
            gp = np.asarray(next(iter(res.values()))).reshape(-1) - ctrl_mean
            record["error_gears"] = float(np.mean(np.abs(gp - true_eff)))
            record["pm_gears"] = float(np.mean(np.abs(gp)))
            record["gv_gears"] = float(np.var(gp))
            g_emb = gears_pair_emb(genes)
            if g_emb is not None and gears_nn is not None:
                record["knn_gears"] = compute_knn_dist(g_emb, gears_nn)
                record["mahalanobis_gears"] = compute_mahalanobis(g_emb, gears_train_embs)
                record["cosine_nearest_gears"] = compute_cosine_to_nearest(
                    g_emb, gears_train_embs)
                if len(gears_train_errs) > 0:
                    record["nn_error_proxy_gears"] = compute_nn_error_proxy(
                        g_emb, gears_train_embs, gears_train_errs)
            record["ensemble_var_gears"] = float("nan")
        except Exception:
            pass

        all_records.append(record)

    print(f"  Records: {sum(1 for r in all_records if r['seed'] == seed)}")

    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"  Checkpoint saved: {output_path}")


###############################################################################
# Results: Table 3 Adamson rows
###############################################################################

print("\n" + "=" * 60)
print("TABLE 3: Cross-dataset reliability (Adamson, single-gene OOD)")
print("=" * 60)

print(f"\n  {'Model':<15}  {'PM':>7}  {'GV':>7}  {'kNN':>7}  {'kNNAbs20':>8}  (AD undefined)")
print(f"  {'~' * 62}")

for model_name, key in [("GEARS", "gears"), ("MLP", "mlp"),
                         ("Transformer", "transformer"), ("CPA", "cpa"),
                         ("scGPT", "scgpt")]:
    valid = [r for r in all_records if f"error_{key}" in r]
    if not valid:
        continue
    errs = np.array([r[f"error_{key}"] for r in valid])
    pms = np.array([r[f"pm_{key}"] for r in valid])
    gvs = np.array([r[f"gv_{key}"] for r in valid])

    rho_pm = stats.spearmanr(pms, errs)[0]
    rho_gv = stats.spearmanr(gvs, errs)[0]

    knn_valid = [r for r in valid if r.get(f"knn_{key}") is not None]
    if len(knn_valid) >= 5:
        knn = np.array([r[f"knn_{key}"] for r in knn_valid])
        knn_errs = np.array([r[f"error_{key}"] for r in knn_valid])
        rho_knn = stats.spearmanr(knn, knn_errs)[0]
        abs20 = abstention_improvement(knn, knn_errs, abstention_rate=0.2)
        knn_str, abs20_str = f"{rho_knn:.3f}", f"{abs20:+.1f}"
    else:
        knn_str, abs20_str = "n/a", "n/a"
    print(f"  {model_name:<15}  {rho_pm:>7.3f}  {rho_gv:>7.3f}  "
          f"{knn_str:>7}  {abs20_str:>8}")


with open(output_path, "w") as f:
    json.dump(all_records, f, indent=2)
print(f"\nSaved {output_path} ({len(all_records)} records)")