import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Perturbation parsing
# ---------------------------------------------------------------------------

def pair_genes(cond):
    """Extract gene names from a condition string like 'GENE1+GENE2' or 'GENE1+ctrl'."""
    return [g for g in cond.split("+") if g != "ctrl"]


# ---------------------------------------------------------------------------
# Norman dataset loading
# ---------------------------------------------------------------------------

def load_norman(data_dir="./data"):
    """
    Load the Norman (2019) dataset via the GEARS package.

    Returns a dictionary with:
        adata           AnnData object
        pert_data       GEARS PertData object
        ctrl_mean       Control mean expression (n_genes,)
        single_effects  {gene_name: effect_vector}
        all_pair_effects {condition: effect_vector} for pair perturbations
        all_effects     {condition: effect_vector} for all perturbations
        single_perts    List of single-gene condition strings
        pair_perts      List of pair condition strings
        comp_eligible   List of eligible pair conditions (both genes seen as singles)
        gene_names      List of output gene names
        gene2idx        {gene_name: index}
        pert_gene2idx   {perturbation_gene: index}
        n_pert_genes    Number of unique perturbation genes
        n_genes_out     Number of output genes
    """
    from gears import PertData

    pert_data = PertData(data_dir)
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

    comp_eligible = [
        p for p in pair_perts
        if all(g in single_gene_names for g in pair_genes(p))
    ]

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
        m = np.array(
            X.toarray().mean(0) if hasattr(X, "toarray") else X.mean(0)
        ).flatten()
        return m - ctrl_mean

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

    all_effects = {}
    for c in list(set(single_perts + pair_perts)):
        eff = get_effect(c)
        if eff is not None:
            all_effects[c] = eff

    gene_names = (
        list(adata.var["gene_name"]) if "gene_name" in adata.var.columns
        else list(adata.var_names)
    )
    gene2idx = {g: i for i, g in enumerate(gene_names)}

    all_pert_genes = sorted(set(g for c in all_effects for g in pair_genes(c)))
    pert_gene2idx = {g: i for i, g in enumerate(all_pert_genes)}

    return {
        "adata": adata,
        "pert_data": pert_data,
        "ctrl_mean": ctrl_mean,
        "single_effects": single_effects,
        "all_pair_effects": all_pair_effects,
        "all_effects": all_effects,
        "single_perts": single_perts,
        "pair_perts": pair_perts,
        "comp_eligible": comp_eligible,
        "gene_names": gene_names,
        "gene2idx": gene2idx,
        "pert_gene2idx": pert_gene2idx,
        "n_pert_genes": len(all_pert_genes),
        "n_genes_out": n_genes_out,
    }


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------

def create_splits(comp_eligible, pair_perts, single_perts, seed,
                  test_frac=0.2, val_frac=0.2):
    """
    Create train/val/test splits for compositional OOD evaluation.

    Eligible pairs are shuffled and partitioned into test (20%), validation
    (20%), and training (60%). All single-gene conditions and non-eligible
    pairs are always in training. Returns a split dictionary compatible
    with GEARS.
    """
    rng = np.random.RandomState(seed)
    eligible = list(comp_eligible)
    rng.shuffle(eligible)
    n = len(eligible)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_pairs = eligible[:n_test]
    val_pairs = eligible[n_test:n_test + n_val]
    train_eligible = eligible[n_test + n_val:]

    non_eligible = [p for p in pair_perts if p not in comp_eligible]
    train_set = single_perts + train_eligible + non_eligible + ["ctrl"]

    return {
        "train": train_set,
        "val": val_pairs,
        "test": test_pairs,
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PertDatasetML(Dataset):
    """Dataset wrapping perturbation gene indices and effect vectors."""

    def __init__(self, conditions, effects_dict, pert_gene2idx):
        self.items = []
        for c in conditions:
            if c not in effects_dict:
                continue
            genes = pair_genes(c)
            if not genes or not all(g in pert_gene2idx for g in genes):
                continue
            idx = [pert_gene2idx[g] for g in genes]
            while len(idx) < 2:
                idx.append(-1)
            self.items.append((
                np.array(idx[:2], dtype=np.int64),
                effects_dict[c].astype(np.float32),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_pytorch(model, train_ds, val_ds, seed, device="cpu",
                  lr=1e-3, epochs=200, patience=30, batch_size=32,
                  use_adamw=False, weight_decay=1e-5):
    """
    Generic training loop for MLP, Transformer, and scGPT models.

    Uses Adam (or AdamW when use_adamw=True) with cosine annealing
    and early stopping.
    """
    torch.manual_seed(seed)
    OptClass = torch.optim.AdamW if use_adamw else torch.optim.Adam
    optimizer = OptClass(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for gi, tgt in train_loader:
            loss = nn.functional.mse_loss(model(gi.to(device)), tgt.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for gi, tgt in val_loader:
                vl += nn.functional.mse_loss(
                    model(gi.to(device)), tgt.to(device)
                ).item()
                nv += 1
        avg_vl = vl / max(nv, 1)

        if avg_vl < best_val:
            best_val = avg_vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_cpa(model, train_ds, val_ds, seed, device="cpu",
              lr=1e-3, epochs=300, patience=40, batch_size=32):
    """
    CPA training loop with KL annealing.

    KL weight linearly increases from 0 to 0.1 over the first 100 epochs.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        kl_weight = min(0.1, ep / 100 * 0.1)

        model.train()
        for gi, tgt in train_loader:
            gi, tgt = gi.to(device), tgt.to(device)
            recon, direct, mu, logvar = model(gi, tgt)
            recon_loss = nn.functional.mse_loss(recon, tgt)
            direct_loss = nn.functional.mse_loss(direct, tgt)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + direct_loss + kl_weight * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        vv, nv = 0.0, 0
        with torch.no_grad():
            for gi, tgt in val_loader:
                pred = model(gi.to(device))
                vv += nn.functional.mse_loss(pred, tgt.to(device)).item()
                nv += 1
        avg = vv / max(nv, 1)

        if avg < best_val:
            best_val = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model
