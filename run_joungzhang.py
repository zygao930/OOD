"""
Joung-Zhang 2023 leave-one-out cross-validation (Table 1, Table 3 Joung-Zhang rows).

Evaluates all models on the JoungZhang2023 combinatorial dataset using
leave-one-out CV. Each double perturbation is held out once and the model
is retrained on the remaining data.

GFP is treated as a control/filler component: GFP+GENE is a single
perturbation, GENE1+GENE2 (no GFP) is a double, and the baseline is the
pure GFP control (or the mean of all GFP+GENE cells if none exists). This
yields the 44 leave-one-out evaluations reported in the paper.

For each model, records AD, PM, GV and a model-specific kNN distance so
Table 3 (AD, PM, GV, kNN, kNN-Abs20) and Table 4 kNN routing can be built.

Models: Ridge, MLP, Transformer, CPA, scGPT, GEARS, GEARS+scGPT

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_joungzhang.py --fold-start 0 --fold-end 15
    python run_joungzhang.py --merge-only
"""

import os
import json
import pickle
import warnings
import gc
import argparse
import glob
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from data_utils import pair_genes
from models import ScGPTPertPredictor
from reliability import abstention_improvement

warnings.filterwarnings("ignore")
os.makedirs("results/joungzhang", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(
    description="Run Joung-Zhang 2023 leave-one-out folds with resumable multi-GPU sharding."
)
parser.add_argument(
    "--fold-start", type=int, default=0,
    help="First fold index to run, inclusive (default: 0)."
)
parser.add_argument(
    "--fold-end", type=int, default=None,
    help="Last fold index to run, exclusive (default: number of eligible folds)."
)
parser.add_argument(
    "--merge-only", action="store_true",
    help="Merge completed shard JSON files into results/joungzhang/records.json and exit."
)
args = parser.parse_args()

EPOCHS = 200
SCGPT_DIR = "data/scgpt_pretrained"
GEARS_EPOCHS = 15
GEARS_CELLS_PER_COND = 32


###############################################################################
# Load Joung-Zhang 2023 data
###############################################################################

print("=" * 60)
print("Loading Joung-Zhang 2023 data")
print("=" * 60)

import anndata as ad

DATA_PATH = "data/joungzhang/JoungZhang2023_combinatorial.h5ad"
adata = ad.read_h5ad(DATA_PATH)
print(f"Loaded: {adata.shape}")

cond_key = "perturbation" if "perturbation" in adata.obs.columns else "condition"

# Joung-Zhang 2023 uses GFP as a control/filler component: GFP+GENE is a
# single perturbation, GENE1+GENE2 (no GFP) is a double perturbation, and
# a pure GFP condition (or the mean of all GFP+GENE cells when none exists)
# provides the control baseline. This mirrors the original source exactly.
CTRL_GENE = "GFP"
ctrl_labels = ["ctrl", "control", "non-targeting"]


def cp_genes(cond):
    """Extract real perturbation genes from a Joung-Zhang condition string,
    stripping both the GFP filler and any control labels."""
    return [
        g for g in str(cond).split("+")
        if g != CTRL_GENE and g not in ctrl_labels and g != "ctrl"
    ]


def parse_joungzhang_cond(cond):
    """Classify a Joung-Zhang condition as ctrl / single / double / skip."""
    s = str(cond)
    parts = [p.strip() for p in s.split("+") if p.strip()]
    if len(parts) == 1 and (parts[0] == CTRL_GENE or parts[0] in ctrl_labels):
        return "ctrl", []
    real = [p for p in parts if p != CTRL_GENE and p not in ctrl_labels]
    if len(real) == 0:
        return "ctrl", []
    elif len(real) == 1 and (CTRL_GENE in parts or any(p in ctrl_labels for p in parts)):
        return "single", real
    elif len(real) == 1 and len(parts) == 1:
        # bare single-gene label (no explicit GFP/ctrl component)
        return "single", real
    elif len(real) == 2 and CTRL_GENE not in parts:
        return "double", sorted(real)
    else:
        return "skip", real


# Identify controls, singles, and doubles.
ctrl_name = None
singles, doubles = [], []          # lists of condition strings
single_cond_by_gene = {}           # gene -> condition string
single_gene_names = set()
for c in adata.obs[cond_key].unique():
    ctype, genes = parse_joungzhang_cond(c)
    if ctype == "ctrl":
        ctrl_name = c
    elif ctype == "single":
        singles.append(c)
        single_cond_by_gene[genes[0]] = c
        single_gene_names.add(genes[0])
    elif ctype == "double":
        doubles.append(c)

n_genes = adata.n_vars

# Control mean: use the pure GFP control if present, otherwise the mean of
# all GFP+GENE (single) cells as the baseline.
if ctrl_name is not None and (adata.obs[cond_key] == ctrl_name).sum() > 0:
    ctrl_mask = adata.obs[cond_key] == ctrl_name
    ctrl_X = adata[ctrl_mask].X
    ctrl_mean = np.array(
        ctrl_X.toarray().mean(0) if hasattr(ctrl_X, "toarray") else ctrl_X.mean(0)
    ).flatten()
    print(f"Control: pure control condition '{ctrl_name}'")
else:
    ctrl_cells = np.zeros(n_genes)
    n_ctrl = 0
    for gene, cond in single_cond_by_gene.items():
        mask = adata.obs[cond_key] == cond
        if mask.sum() > 0:
            X = adata[mask].X
            ctrl_cells += np.array(
                X.toarray().mean(0) if hasattr(X, "toarray") else X.mean(0)
            ).flatten() * mask.sum()
            n_ctrl += mask.sum()
    ctrl_mean = ctrl_cells / max(n_ctrl, 1)
    print(f"Control: mean of {n_ctrl} GFP+GENE single-perturbation cells")


def get_effect(cond_name):
    mask = adata.obs[cond_key] == cond_name
    if mask.sum() == 0:
        return None
    X = adata[mask].X
    return np.array(
        X.toarray().mean(0) if hasattr(X, "toarray") else X.mean(0)
    ).flatten() - ctrl_mean


# AnnData-insertion-order eligible pairs (used to build each fold's validation
# pool, matching the original's eligible.items() pre-shuffle order).
eligible_anndata_order = [
    d for d in doubles if all(g in single_gene_names for g in cp_genes(d))
]
# Fold iteration order is lexicographic by sorted gene tuple, matching the
# original's enumerate(sorted(pair_effects.keys())) so fold_idx maps to the
# same held-out pair (and thus the same shuffle/model seeds).
eligible = sorted(eligible_anndata_order, key=lambda c: tuple(sorted(cp_genes(c))))
print(f"Singles: {len(singles)}, Doubles: {len(doubles)}, Eligible: {len(eligible)}")

# Resolve the requested fold shard after the eligible fold count is known.
fold_start = max(0, args.fold_start)
fold_end = len(eligible) if args.fold_end is None else min(args.fold_end, len(eligible))
if fold_start >= fold_end and not args.merge_only:
    raise ValueError(
        f"Invalid fold range [{fold_start}, {fold_end}); eligible folds: 0..{len(eligible) - 1}"
    )

def merge_shard_records():
    """Merge non-overlapping or partially overlapping shard files by fold index."""
    paths = sorted(glob.glob("results/joungzhang/records_folds_*.json"))
    canonical = "results/joungzhang/records.json"
    if not paths and os.path.exists(canonical):
        print(f"No shard files found; keeping existing {canonical}.")
        return
    merged = {}
    for path in paths:
        try:
            with open(path, "r") as f:
                rows = json.load(f)
            for row in rows:
                if "fold" in row:
                    merged[int(row["fold"])] = row
        except Exception as e:
            print(f"Warning: could not read {path}: {e}")
    ordered = [merged[k] for k in sorted(merged)]
    with open(canonical, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"Merged {len(paths)} shard files into {canonical} ({len(ordered)} folds).")
    missing = sorted(set(range(len(eligible))) - set(merged))
    if missing:
        print(f"Missing folds: {missing}")
    else:
        print("All folds are present.")

if args.merge_only:
    merge_shard_records()
    raise SystemExit(0)

# A full single-process run keeps the historical canonical path. Sharded runs
# write separate files so concurrent GPU processes never overwrite each other.
if fold_start == 0 and fold_end == len(eligible):
    records_path = "results/joungzhang/records.json"
else:
    records_path = f"results/joungzhang/records_folds_{fold_start:02d}_{fold_end:02d}.json"

if os.path.exists(records_path):
    try:
        with open(records_path, "r") as f:
            all_records = json.load(f)
    except Exception:
        all_records = []
else:
    all_records = []
completed_folds = {int(r["fold"]) for r in all_records if "fold" in r}
print(
    f"Fold shard: [{fold_start}, {fold_end}); output: {records_path}; "
    f"resuming with {len(completed_folds)} completed folds."
)

# Precompute effects
single_effects = {}
for c in singles:
    eff = get_effect(c)
    if eff is not None:
        g = cp_genes(c)[0]
        single_effects[g] = eff

all_effects = {}
for c in singles + doubles:
    eff = get_effect(c)
    if eff is not None:
        all_effects[c] = eff

all_pert_genes = sorted(set(g for c in all_effects for g in cp_genes(c)))
pg2idx = {g: i for i, g in enumerate(all_pert_genes)}
n_pg = len(all_pert_genes)


###############################################################################
# Model definitions (compact, for LOO retraining)
###############################################################################

class MLPPredictor(nn.Module):
    def __init__(self, n_pg, n_out, emb=64, hid=256):
        super().__init__()
        self.emb = nn.Embedding(n_pg + 1, emb, padding_idx=n_pg)
        self.pad = n_pg
        self.pos = nn.Parameter(torch.randn(2, emb) * 0.02)
        self.net = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(), nn.LayerNorm(hid), nn.Dropout(0.1),
            nn.Linear(hid, hid), nn.ReLU(), nn.LayerNorm(hid), nn.Dropout(0.1),
            nn.Linear(hid, n_out))

    def forward(self, x):
        idx = x.clone(); idx[idx < 0] = self.pad
        e = self.emb(idx) + self.pos.unsqueeze(0)
        m = (x >= 0).unsqueeze(-1).float()
        return self.net((e * m).sum(1) / m.sum(1).clamp(min=1))


class TransformerPredictor(nn.Module):
    def __init__(self, n_pg, n_out, emb=64, heads=4, layers=2, hid=256):
        super().__init__()
        self.emb = nn.Embedding(n_pg + 1, emb, padding_idx=n_pg)
        self.pad = n_pg
        self.pos = nn.Parameter(torch.randn(2, emb) * 0.02)
        el = nn.TransformerEncoderLayer(emb, heads, hid, 0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(el, layers)
        self.head = nn.Sequential(
            nn.Linear(emb, hid), nn.GELU(), nn.LayerNorm(hid), nn.Dropout(0.1),
            nn.Linear(hid, n_out))

    def forward(self, x):
        idx = x.clone(); idx[idx < 0] = self.pad
        e = self.emb(idx) + self.pos.unsqueeze(0)
        pm = x < 0
        o = self.tf(e, src_key_padding_mask=pm)
        m = (~pm).unsqueeze(-1).float()
        return self.head((o * m).sum(1) / m.sum(1).clamp(min=1))


class CPAPredictor(nn.Module):
    def __init__(self, n_pg, n_out, emb=64, lat=64, hid=256):
        super().__init__()
        self.pad = n_pg
        self.emb = nn.Embedding(n_pg + 1, emb, padding_idx=n_pg)
        self.encoder = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(), nn.LayerNorm(hid))
        self.mu = nn.Linear(hid, lat)
        self.logvar = nn.Linear(hid, lat)
        self.decoder = nn.Sequential(
            nn.Linear(lat, hid), nn.ReLU(), nn.LayerNorm(hid), nn.Dropout(0.1),
            nn.Linear(hid, hid), nn.ReLU(), nn.LayerNorm(hid), nn.Dropout(0.1),
            nn.Linear(hid, n_out))

    def forward(self, x, tgt=None):
        idx = x.clone(); idx[idx < 0] = self.pad
        e = self.emb(idx)
        m = (x >= 0).unsqueeze(-1).float()
        pooled = (e * m).sum(1) / m.sum(1).clamp(min=1)
        h = self.encoder(pooled)
        mu = self.mu(h); logvar = self.logvar(h)
        if self.training:
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z = mu
        return self.decoder(z)


class PertDS(Dataset):
    def __init__(self, conds, effs, pg2i):
        self.items = []
        for c in conds:
            if c not in effs:
                continue
            gs = cp_genes(c)
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


def train_model(model, tds, vds, seed, epochs=EPOCHS, patience=30):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tl = DataLoader(tds, 32, shuffle=True)
    vl = DataLoader(vds, 128)
    bv, bs, w = float("inf"), None, 0

    for ep in range(1, epochs + 1):
        model.train()
        for gi, tgt in tl:
            gi, tgt = gi.to(device), tgt.to(device)
            loss = nn.functional.mse_loss(model(gi), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        model.eval(); vv, nv = 0, 0
        with torch.no_grad():
            for gi, tgt in vl:
                p = model(gi.to(device))
                vv += nn.functional.mse_loss(p, tgt.to(device)).item(); nv += 1
        avg = vv / max(nv, 1)
        if avg < bv:
            bv = avg; bs = {k: v.clone() for k, v in model.state_dict().items()}; w = 0
        else:
            w += 1
        if w >= patience:
            break

    if bs:
        model.load_state_dict(bs)
    return model


###############################################################################
# Model-specific kNN distance helpers (Table 3 / Table 4 routing signal)
###############################################################################

from sklearn.metrics.pairwise import cosine_distances


def get_embedding(model, gene_idx_tensor):
    """Extract the pre-head pooled embedding from MLP / Transformer / CPA."""
    model.eval()
    with torch.no_grad():
        idx = gene_idx_tensor.clone()
        idx[idx < 0] = model.pad
        e = model.emb(idx)
        if hasattr(model, "pos"):
            e = e + model.pos.unsqueeze(0)
        mask = (gene_idx_tensor >= 0).unsqueeze(-1).float()
        if hasattr(model, "tf"):
            pad_mask = (gene_idx_tensor < 0)
            out = model.tf(e, src_key_padding_mask=pad_mask)
            mf = (~pad_mask).unsqueeze(-1).float()
            pooled = (out * mf).sum(1) / mf.sum(1).clamp(min=1)
        elif hasattr(model, "encoder"):
            pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = model.encoder(pooled)
        else:
            pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
    return pooled.cpu().numpy()


def compute_knn_emb(model, ho_tensor, train_ds, k=5):
    """Mean cosine distance to k nearest training pairs in embedding space."""
    try:
        ho_emb = get_embedding(model, ho_tensor)
        train_embs = []
        for gi, _ in DataLoader(train_ds, batch_size=128, shuffle=False):
            train_embs.append(get_embedding(model, gi.to(device)))
        if not train_embs:
            return None
        train_embs = np.vstack(train_embs)
        dists = cosine_distances(ho_emb, train_embs).flatten()
        return float(np.mean(np.sort(dists)[:min(k, len(dists))]))
    except Exception:
        return None


def compute_knn_scgpt(model, ho_tensor, train_ds, k=5):
    """kNN distance in scGPT's learned embedding space (Euclidean)."""
    try:
        model.eval()
        with torch.no_grad():
            def embed(t):
                idx = t.clone(); idx[idx < 0] = model.pad_idx
                raw = model.pretrained_emb[idx]
                proj = model.emb_proj(raw)
                combined = proj + model.pert_emb(idx) + model.pos_emb.unsqueeze(0)
                pad = (t < 0)
                out = model.transformer(combined, src_key_padding_mask=pad)
                mf = (~pad).unsqueeze(-1).float()
                return ((out * mf).sum(1) / mf.sum(1).clamp(min=1)).cpu().numpy()

            ho_emb = embed(ho_tensor)
            train_embs = []
            for gi, _ in DataLoader(train_ds, batch_size=128, shuffle=False):
                train_embs.append(embed(gi.to(device)))
        if not train_embs:
            return None
        train_embs = np.vstack(train_embs)
        dists = np.linalg.norm(train_embs - ho_emb, axis=1)
        return float(np.mean(np.sort(dists)[:min(k, len(dists))]))
    except Exception:
        return None


def compute_knn_gears(gears_model, ho_genes_local, train_conds_local, k=5):
    """kNN distance in GEARS's gene embedding space (Euclidean)."""
    try:
        emb_weight = gears_model.model.gene_emb.weight.data.cpu().numpy()

        def pair_emb(genes):
            embs = [emb_weight[pg2idx[g]] for g in genes
                    if g in pg2idx and pg2idx[g] < emb_weight.shape[0]]
            return np.mean(embs, axis=0) if embs else None

        ho_emb = pair_emb(ho_genes_local)
        if ho_emb is None:
            return None
        train_embs = []
        for c in train_conds_local:
            e = pair_emb(cp_genes(c))
            if e is not None:
                train_embs.append(e)
        if not train_embs:
            return None
        train_embs = np.array(train_embs)
        dists = np.linalg.norm(train_embs - ho_emb, axis=1)
        return float(np.mean(np.sort(dists)[:min(k, len(dists))]))
    except Exception:
        return None


def compute_gv_scgpt_mc(model, ho_tensor, n_passes=5):
    """Gene-wise variance for scGPT via MC dropout (mean per-gene variance
    across n stochastic forward passes), matching the original."""
    try:
        model.train()  # enable dropout
        mc_preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                mc_preds.append(model(ho_tensor).cpu().numpy().flatten())
        model.eval()
        return float(np.mean(np.var(np.array(mc_preds), axis=0)))
    except Exception:
        return None


def compute_gv_gears_percell(gears_model, gene_pair, device_local):
    """Gene-wise variance for GEARS: variance across per-cell predictions on
    control cells (mean over genes), matching the original."""
    try:
        from gears.utils import create_cell_graph_dataset_for_prediction
        try:
            from torch_geometric.loader import DataLoader as PyGLoader
        except Exception:
            from torch_geometric.data import DataLoader as PyGLoader
        ctrl_ad = gears_model.adata[gears_model.adata.obs["condition"] == "ctrl"]
        cg = create_cell_graph_dataset_for_prediction(
            list(gene_pair), ctrl_ad, gears_model.pert_list, device_local)
        loader = PyGLoader(cg, 64, shuffle=False)
        model_obj = gears_model.model
        model_obj.eval()
        all_p = []
        with torch.no_grad():
            for batch in loader:
                batch.to(device_local)
                p = model_obj(batch)
                all_p.append(p.detach().cpu().numpy())
                del p, batch
                torch.cuda.empty_cache()
        per_cell = np.vstack(all_p)
        return float(np.mean(np.var(per_cell, axis=0)))
    except Exception:
        return None


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

    # Also map output genes for GEARS+scGPT initialization.
    gene_names_out = list(adata.var_names)
    gene2idx_output = {g: i for i, g in enumerate(gene_names_out)}
    scgpt_embs_out = np.zeros((n_genes, pretrained_emb.shape[1]), dtype=np.float32)
    for i, gn in enumerate(gene_names_out):
        vidx = scgpt_vocab.get(gn, vocab_lower.get(gn.lower()))
        if vidx is not None and vidx < pretrained_emb.shape[0]:
            scgpt_embs_out[i] = pretrained_emb[vidx]
    scgpt_red_64 = PCA(n_components=64).fit_transform(scgpt_embs_out)
else:
    gene2idx_output = {}
    print("\nscGPT pretrained weights not found; skipping scGPT and GEARS+scGPT.")


###############################################################################
# Set up GEARS (optional)
###############################################################################

gears_available = False
pdata = None
try:
    from gears import PertData, GEARS
    gears_available = True
    print("\nGEARS package available.")
except ImportError:
    print("\nGEARS not installed; skipping GEARS and GEARS+scGPT.")


def build_gears_adata_joungzhang(adata_full, cond_key_local, held_out_cond,
                              n_pseudo_ctrl=256, max_cells=32, seed=0):
    """Build AnnData with GEARS-compatible 'condition' column for Joung-Zhang."""
    ac = adata_full.copy()
    cond_map = {}
    keep_mask = np.ones(ac.n_obs, dtype=bool)
    obs_vals = ac.obs[cond_key_local].to_numpy()
    for c in ac.obs[cond_key_local].unique():
        cs = str(c)
        ctype, gs = parse_joungzhang_cond(cs)
        if ctype == "ctrl" or cs in ctrl_labels or cs == CTRL_GENE:
            cond_map[cs] = "ctrl"
        elif ctype == "single":
            cond_map[cs] = cp_genes(cs)[0] + "+ctrl"
        elif ctype == "double":
            cond_map[cs] = "+".join(sorted(cp_genes(cs)))
        else:
            # 'skip' conditions (triples etc.) are removed entirely, matching
            # the original: they must not enter GEARS training or validation.
            keep_mask[obs_vals == c] = False
            cond_map[cs] = cs

    ac.obs["condition"] = [cond_map.get(str(x), str(x)) for x in ac.obs[cond_key_local]]
    ac.obs["condition_name"] = ac.obs["condition"].copy()
    ac = ac[keep_mask].copy()

    # Create pseudo-controls from single-gene cells if no ctrl cells exist
    cvals = ac.obs["condition"].astype(str).to_numpy()
    if not np.any(cvals == "ctrl"):
        single_idx = np.flatnonzero(
            ac.obs["condition"].astype(str).str.endswith("+ctrl").to_numpy()
        )
        if single_idx.size > 0:
            rng = np.random.default_rng(seed)
            take = rng.choice(single_idx, size=min(n_pseudo_ctrl, single_idx.size), replace=False)
            pseudo = ac[take].copy()
            pseudo.obs["condition"] = "ctrl"
            pseudo.obs["condition_name"] = "ctrl"
            pseudo.obs_names = [f"pseudo_ctrl_{i}" for i in range(pseudo.n_obs)]
            ac = ad.concat([ac, pseudo], axis=0, join="inner")

    # Subsample cells per condition
    rng = np.random.default_rng(seed)
    selected = []
    for cond in ac.obs["condition"].unique():
        idx = np.flatnonzero(ac.obs["condition"].to_numpy() == cond)
        if idx.size > max_cells:
            idx = rng.choice(idx, size=max_cells, replace=False)
        selected.append(idx)
    ac = ac[np.sort(np.concatenate(selected))].copy()

    if "gene_name" not in ac.var.columns:
        ac.var["gene_name"] = ac.var_names.tolist()
    ac.obs["cell_type"] = "joungzhang"
    for k in list(ac.obsm.keys()):
        del ac.obsm[k]
    ac.uns = {}
    return ac


###############################################################################
# Leave-one-out CV
###############################################################################

print(f"\nRunning fold shard [{fold_start}, {fold_end}) of {len(eligible)}-fold LOO CV")

# Prepare shared GEARS dataset once (if available)
if gears_available:
    try:
        gears_ds_name = f"joungzhang_gears_shared_c{GEARS_CELLS_PER_COND}"
        gears_ds_dir = os.path.abspath(os.path.join("./data", gears_ds_name))
        os.makedirs(gears_ds_dir, exist_ok=True)
        pdata = PertData("./data")
        gears_h5ad = os.path.join(gears_ds_dir, "perturb_processed.h5ad")
        # Multiple shards may start simultaneously. Serialize the one-time
        # GEARS preprocessing step to avoid corrupting the shared cache.
        import fcntl
        lock_path = os.path.join(gears_ds_dir, ".prepare.lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if not os.path.exists(gears_h5ad):
                print("Preparing shared GEARS Joung-Zhang dataset (one-time)...")
                gears_ad = build_gears_adata_joungzhang(
                    adata, cond_key, eligible[0],
                    max_cells=GEARS_CELLS_PER_COND)
                pdata.new_data_process(dataset_name=gears_ds_name, adata=gears_ad,
                                       skip_calc_de=False)
                del gears_ad
            else:
                pdata.load(data_path=gears_ds_dir)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        all_gears_conds = [
            str(c) for c in pdata.adata.obs["condition"].unique() if str(c) != "ctrl"
        ]
        print(f"  GEARS dataset: {pdata.adata.n_obs} cells, {len(all_gears_conds)} conditions")
    except Exception as e:
        print(f"  GEARS dataset prep failed: {e}")
        gears_available = False
        pdata = None

for fold_idx in range(fold_start, fold_end):
    held_out = eligible[fold_idx]
    if fold_idx in completed_folds:
        print(f"\n  Fold {fold_idx + 1}/{len(eligible)} already complete; skipping {held_out}")
        continue
    print(f"\n  Fold {fold_idx + 1}/{len(eligible)}: hold out {held_out}")

    if held_out not in all_effects:
        continue

    true_eff = all_effects[held_out]
    ho_genes = cp_genes(held_out)
    if not all(g in single_effects for g in ho_genes):
        continue

    add_pred = sum(single_effects[g] for g in ho_genes)
    top20 = np.argsort(np.abs(true_eff))[-20:]

    # Validation: the pool is the ELIGIBLE double pairs (in AnnData order,
    # matching the original's eligible.items()) minus the held-out pair.
    # Shuffle with the fold index as seed, then hold out ~1/5 as validation.
    remaining_doubles = [c for c in eligible_anndata_order if c != held_out]
    np.random.seed(fold_idx)
    shuffled_doubles = list(remaining_doubles)
    np.random.shuffle(shuffled_doubles)
    n_val = max(1, len(shuffled_doubles) // 5)
    val_conds = shuffled_doubles[:n_val]
    train_conds = singles + shuffled_doubles[n_val:]
    # Ridge and the kNN reference sets use ALL non-held-out conditions
    # (singles + eligible doubles), i.e. train + val, matching the original.
    all_nonho_conds = singles + remaining_doubles

    record = {
        "fold": fold_idx,
        "pair": held_out,
        "error_additive": float(np.mean(np.abs(add_pred - true_eff))),
        "top20_additive": float(np.mean(np.abs(add_pred[top20] - true_eff[top20]))),
    }

    # Ridge (trained on ALL non-held-out conditions: singles + doubles)
    X_r, Y_r = [], []
    for c in all_nonho_conds:
        if c not in all_effects:
            continue
        v = np.zeros(n_pg, np.float32)
        for g in cp_genes(c):
            if g in pg2idx:
                v[pg2idx[g]] = 1.0
        X_r.append(v)
        Y_r.append(all_effects[c])
    if X_r:
        ridge = Ridge(alpha=1.0).fit(np.array(X_r), np.array(Y_r))
        v = np.zeros(n_pg, np.float32)
        for g in ho_genes:
            if g in pg2idx:
                v[pg2idx[g]] = 1.0
        rp = ridge.predict(v.reshape(1, -1)).flatten()
        record["error_ridge"] = float(np.mean(np.abs(rp - true_eff)))
        record["top20_ridge"] = float(np.mean(np.abs(rp[top20] - true_eff[top20])))
        record["ad_ridge"] = float(np.mean(np.abs(rp - add_pred)))
        record["pm_ridge"] = float(np.mean(np.abs(rp)))
        record["gv_ridge"] = float(np.var(rp))
        # Ridge kNN: use Ridge coefficient columns as gene embeddings,
        # pair embedding = sum of the two gene embeddings.
        ridge_coef = ridge.coef_  # (n_genes_out, n_pert_genes)

        def ridge_pair_emb(genes):
            emb = np.zeros(n_genes)
            for g in genes:
                if g in pg2idx:
                    emb += ridge_coef[:, pg2idx[g]]
            return emb

        ho_r_emb = ridge_pair_emb(ho_genes).reshape(1, -1)
        train_pair_embs = [
            ridge_pair_emb(cp_genes(c)) for c in all_nonho_conds
            if c in all_effects and len(cp_genes(c)) == 2
        ]
        if train_pair_embs:
            d = cosine_distances(ho_r_emb, np.array(train_pair_embs)).flatten()
            record["knn_ridge"] = float(np.mean(np.sort(d)[:min(5, len(d))]))
        else:
            record["knn_ridge"] = None

    tds = PertDS(train_conds, all_effects, pg2idx)
    vds = PertDS(val_conds, all_effects, pg2idx)

    def make_idx(pname):
        gs = cp_genes(pname)
        idx = [pg2idx[g] for g in gs if g in pg2idx]
        while len(idx) < 2:
            idx.append(-1)
        return torch.tensor([idx[:2]], dtype=torch.long, device=device)

    # MLP (seed = fold_idx + 100, matching the original)
    torch.manual_seed(fold_idx + 100)
    mlp = MLPPredictor(n_pg, n_genes).to(device)
    mlp = train_model(mlp, tds, vds, fold_idx + 100)
    mlp.eval()
    with torch.no_grad():
        mp = mlp(make_idx(held_out)).cpu().numpy().flatten()
    record["error_mlp"] = float(np.mean(np.abs(mp - true_eff)))
    record["top20_mlp"] = float(np.mean(np.abs(mp[top20] - true_eff[top20])))
    record["ad_mlp"] = float(np.mean(np.abs(mp - add_pred)))
    record["pm_mlp"] = float(np.mean(np.abs(mp)))
    record["gv_mlp"] = float(np.var(mp))
    record["knn_mlp"] = compute_knn_emb(mlp, make_idx(held_out), tds)

    # Transformer (seed = fold_idx + 200)
    torch.manual_seed(fold_idx + 200)
    trans = TransformerPredictor(n_pg, n_genes).to(device)
    trans = train_model(trans, tds, vds, fold_idx + 200)
    trans.eval()
    with torch.no_grad():
        tp = trans(make_idx(held_out)).cpu().numpy().flatten()
    record["error_transformer"] = float(np.mean(np.abs(tp - true_eff)))
    record["top20_transformer"] = float(np.mean(np.abs(tp[top20] - true_eff[top20])))
    record["ad_transformer"] = float(np.mean(np.abs(tp - add_pred)))
    record["pm_transformer"] = float(np.mean(np.abs(tp)))
    record["gv_transformer"] = float(np.var(tp))
    record["knn_transformer"] = compute_knn_emb(trans, make_idx(held_out), tds)

    # CPA (seed = fold_idx + 300)
    torch.manual_seed(fold_idx + 300)
    cpa = CPAPredictor(n_pg, n_genes).to(device)
    cpa = train_model(cpa, tds, vds, fold_idx + 300, epochs=300, patience=40)
    cpa.eval()
    with torch.no_grad():
        cp = cpa(make_idx(held_out)).cpu().numpy().flatten()
    record["error_cpa"] = float(np.mean(np.abs(cp - true_eff)))
    record["top20_cpa"] = float(np.mean(np.abs(cp[top20] - true_eff[top20])))
    record["ad_cpa"] = float(np.mean(np.abs(cp - add_pred)))
    record["pm_cpa"] = float(np.mean(np.abs(cp)))
    record["gv_cpa"] = float(np.var(cp))
    record["knn_cpa"] = compute_knn_emb(cpa, make_idx(held_out), tds)

    del mlp, trans, cpa
    gc.collect(); torch.cuda.empty_cache()

    # scGPT (fine-tuned)
    if scgpt_embs is not None:
        torch.manual_seed(fold_idx + 1000)
        scgpt_m = ScGPTPertPredictor(n_pg, n_genes, scgpt_embs).to(device)
        scgpt_opt = torch.optim.AdamW(scgpt_m.parameters(), lr=5e-4, weight_decay=1e-4)
        scgpt_sch = torch.optim.lr_scheduler.CosineAnnealingLR(scgpt_opt, EPOCHS)
        tl = DataLoader(tds, 32, shuffle=True)
        vl = DataLoader(vds, 128)
        bv, bs, w = float("inf"), None, 0
        for ep in range(1, EPOCHS + 1):
            scgpt_m.train()
            for gi, tgt in tl:
                loss = nn.functional.mse_loss(scgpt_m(gi.to(device)), tgt.to(device))
                scgpt_opt.zero_grad(); loss.backward(); scgpt_opt.step()
            scgpt_sch.step()
            scgpt_m.eval(); vv, nv = 0, 0
            with torch.no_grad():
                for gi, tgt in vl:
                    vv += nn.functional.mse_loss(scgpt_m(gi.to(device)), tgt.to(device)).item()
                    nv += 1
            avg = vv / max(nv, 1)
            if avg < bv:
                bv = avg; bs = {k: v.clone() for k, v in scgpt_m.state_dict().items()}; w = 0
            else:
                w += 1
            if w >= 30:
                break
        if bs:
            scgpt_m.load_state_dict(bs)
        scgpt_m.eval()
        with torch.no_grad():
            sp = scgpt_m(make_idx(held_out)).cpu().numpy().flatten()
        record["error_scgpt"] = float(np.mean(np.abs(sp - true_eff)))
        record["top20_scgpt"] = float(np.mean(np.abs(sp[top20] - true_eff[top20])))
        record["ad_scgpt"] = float(np.mean(np.abs(sp - add_pred)))
        record["pm_scgpt"] = float(np.mean(np.abs(sp)))
        record["gv_scgpt"] = compute_gv_scgpt_mc(scgpt_m, make_idx(held_out))
        if record["gv_scgpt"] is None:
            record["gv_scgpt"] = float(np.var(sp))
        record["knn_scgpt"] = compute_knn_scgpt(scgpt_m, make_idx(held_out), tds)
        del scgpt_m; gc.collect(); torch.cuda.empty_cache()

    # GEARS
    if gears_available and pdata is not None:
        try:
            test_cond = "+".join(sorted(ho_genes))
            pair_conds = [c for c in all_gears_conds
                          if "ctrl" not in c and c != test_cond]
            val_gc = pair_conds[-2:] if len(pair_conds) > 2 else pair_conds[:1]
            train_gc = [c for c in all_gears_conds
                        if c not in val_gc and c != test_cond]
            split_dict = {"train": train_gc + ["ctrl"], "test": [test_cond], "val": val_gc}
            split_pkl = f"/tmp/joungzhang_gears_split_{fold_idx}.pkl"
            with open(split_pkl, "wb") as f:
                pickle.dump(split_dict, f)
            pdata.prepare_split(split="custom", seed=1, split_dict_path=split_pkl)
            pdata.get_dataloader(batch_size=64, test_batch_size=256)

            gears_obj = GEARS(pdata, device=device)
            gears_obj.model_initialize(hidden_size=64)
            gears_obj.train(epochs=GEARS_EPOCHS, lr=1e-3)
            torch.cuda.empty_cache(); gc.collect()

            # Compute kNN in GEARS embedding space BEFORE predict frees model.
            record["knn_gears"] = compute_knn_gears(gears_obj, ho_genes, train_gc)

            res = gears_obj.predict([list(ho_genes)])
            gp = np.asarray(next(iter(res.values()))).reshape(-1) - ctrl_mean
            if gp.shape[0] == n_genes:
                record["error_gears"] = float(np.mean(np.abs(gp - true_eff)))
                record["top20_gears"] = float(np.mean(np.abs(gp[top20] - true_eff[top20])))
                record["ad_gears"] = float(np.mean(np.abs(gp - add_pred)))
                record["pm_gears"] = float(np.mean(np.abs(gp)))
                record["gv_gears"] = compute_gv_gears_percell(gears_obj, ho_genes, device)
                if record["gv_gears"] is None:
                    record["gv_gears"] = float(np.var(gp))
            del gears_obj; torch.cuda.empty_cache(); gc.collect()

            # GEARS+scGPT
            if scgpt_embs is not None:
                gs_obj = GEARS(pdata, device=device)
                gs_obj.model_initialize(hidden_size=64)
                try:
                    gew = gs_obj.model.gene_emb.weight.data
                    gears_glist = list(pdata.adata.var_names)
                    inj = 0
                    for i, gn in enumerate(gears_glist):
                        if i >= gew.shape[0]:
                            break
                        oi = gene2idx_output.get(gn)
                        if oi is not None and oi < scgpt_red_64.shape[0]:
                            gew[i] = torch.tensor(scgpt_red_64[oi], dtype=torch.float32)
                            inj += 1
                except Exception:
                    pass
                gs_obj.train(epochs=GEARS_EPOCHS, lr=1e-3)
                torch.cuda.empty_cache(); gc.collect()

                record["knn_gears_scgpt"] = compute_knn_gears(gs_obj, ho_genes, train_gc)

                res = gs_obj.predict([list(ho_genes)])
                gsp = np.asarray(next(iter(res.values()))).reshape(-1) - ctrl_mean
                if gsp.shape[0] == n_genes:
                    record["error_gears_scgpt"] = float(np.mean(np.abs(gsp - true_eff)))
                    record["top20_gears_scgpt"] = float(np.mean(np.abs(gsp[top20] - true_eff[top20])))
                    record["ad_gears_scgpt"] = float(np.mean(np.abs(gsp - add_pred)))
                    record["pm_gears_scgpt"] = float(np.mean(np.abs(gsp)))
                    record["gv_gears_scgpt"] = compute_gv_gears_percell(gs_obj, ho_genes, device)
                    if record["gv_gears_scgpt"] is None:
                        record["gv_gears_scgpt"] = float(np.var(gsp))
                del gs_obj; torch.cuda.empty_cache(); gc.collect()

        except Exception as e:
            print(f"    GEARS error: {e}")

    all_records.append(record)

    # Save the shard immediately after every completed fold. If the process is
    # interrupted during a fold, only that current fold is repeated on restart.
    all_records = sorted(all_records, key=lambda r: int(r.get("fold", -1)))
    tmp_path = records_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(all_records, f, indent=2)
    os.replace(tmp_path, records_path)
    completed_folds.add(fold_idx)


###############################################################################
# Results
###############################################################################

print("\n" + "=" * 60)
print("TABLE 1: Prediction performance (Joung-Zhang 2023)")
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
    t20 = [r.get(f"top20_{key}") for r in valid]
    t20 = [x for x in t20 if x is not None]
    add_errs = [r["error_additive"] for r in valid]
    wins = sum(1 for e, a in zip(errs, add_errs) if e < a) if key != "additive" else "n/a"
    w_str = f"{wins}/{len(valid)}" if isinstance(wins, int) else wins
    t20_str = f"{np.mean(t20):.4f}" if t20 else "n/a"
    print(f"  {name:<18} {np.mean(errs):>8.4f} {t20_str:>8} {w_str:>10}")


print("\n" + "=" * 60)
print("TABLE 3: Cross-dataset reliability (Joung-Zhang 2023)")
print("=" * 60)
print(f"\n  {'Model':<15}  {'AD':>7}  {'PM':>7}  {'GV':>7}  {'kNN':>7}  {'kNNAbs20':>8}")
print(f"  {'~' * 62}")

for name, key in model_keys:
    if key == "additive":
        continue
    valid = [r for r in all_records if f"ad_{key}" in r and f"error_{key}" in r]
    if not valid:
        continue
    ads = np.array([r[f"ad_{key}"] for r in valid])
    pms = np.array([r[f"pm_{key}"] for r in valid])
    gvs = np.array([r[f"gv_{key}"] for r in valid])
    errs = np.array([r[f"error_{key}"] for r in valid])

    rho_ad = stats.spearmanr(ads, errs)[0]
    rho_pm = stats.spearmanr(pms, errs)[0]
    rho_gv = stats.spearmanr(gvs, errs)[0]

    # kNN rho and kNN Abs@20%: MAE improvement after abstaining on the
    # highest-risk 20% of predictions (per the paper's Abs20 definition).
    knn_valid = [r for r in valid if r.get(f"knn_{key}") is not None]
    if len(knn_valid) >= 5:
        knn = np.array([r[f"knn_{key}"] for r in knn_valid])
        knn_errs = np.array([r[f"error_{key}"] for r in knn_valid])
        rho_knn = stats.spearmanr(knn, knn_errs)[0]
        abs20 = abstention_improvement(knn, knn_errs, abstention_rate=0.2)
        knn_str = f"{rho_knn:.3f}"
        abs20_str = f"{abs20:+.1f}"
    else:
        knn_str = "n/a"
        abs20_str = "n/a"
    print(f"  {name:<15}  {rho_ad:>7.3f}  {rho_pm:>7.3f}  {rho_gv:>7.3f}  "
          f"{knn_str:>7}  {abs20_str:>8}")


print(f"\nSaved {records_path} ({len(all_records)} records)")