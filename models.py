import numpy as np
import torch
import torch.nn as nn


class PertMLP(nn.Module):
    """
    Feedforward perturbation predictor.

    64-dimensional gene embeddings, 2x256 hidden layers with ReLU,
    LayerNorm, and dropout 0.1. Genes are mean-pooled before prediction.
    """

    def __init__(self, n_pert_genes, n_output_genes, emb_dim=64, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(n_pert_genes + 1, emb_dim, padding_idx=n_pert_genes)
        self.pad_idx = n_pert_genes
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, n_output_genes),
        )

    def forward(self, gene_idx):
        idx = gene_idx.clone()
        idx[idx < 0] = self.pad_idx
        embs = self.emb(idx)
        mask = (gene_idx >= 0).unsqueeze(-1).float()
        pooled = (embs * mask).sum(dim=1)
        return self.net(pooled)


class PertTransformer(nn.Module):
    """
    Transformer perturbation predictor.

    64-dimensional embeddings, 4 heads, 2 layers, 256-dimensional
    feedforward, pre-norm, learnable positional embeddings, mean pooling.
    """

    def __init__(self, n_pert_genes, n_output_genes, emb_dim=64,
                 n_heads=4, n_layers=2, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(n_pert_genes + 1, emb_dim, padding_idx=n_pert_genes)
        self.pad_idx = n_pert_genes
        self.pos_emb = nn.Parameter(torch.randn(2, emb_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=hidden,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, n_output_genes),
        )

    def forward(self, gene_idx):
        idx = gene_idx.clone()
        idx[idx < 0] = self.pad_idx
        embs = self.emb(idx) + self.pos_emb.unsqueeze(0)
        pad_mask = gene_idx < 0
        out = self.transformer(embs, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled)

    def get_embedding(self, gene_idx):
        idx = gene_idx.clone()
        idx[idx < 0] = self.pad_idx
        embs = self.emb(idx) + self.pos_emb.unsqueeze(0)
        pad_mask = gene_idx < 0
        out = self.transformer(embs, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).float()
        return ((out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).detach()


class CompositionalPertVAE(nn.Module):
    """
    CPA (Compositional Perturbation Autoencoder).

    VAE with per-gene perturbation embeddings composed additively in
    latent space: delta_{A+B} = delta_A + delta_B. 64-dimensional
    latent, KL annealing (max 0.1 over 100 epochs), 300 epochs,
    patience 40.
    """

    def __init__(self, n_pert_genes, n_output_genes, emb_dim=64,
                 latent_dim=64, hidden=256):
        super().__init__()
        self.pad_idx = n_pert_genes
        self.latent_dim = latent_dim

        self.pert_emb = nn.Embedding(n_pert_genes + 1, latent_dim,
                                     padding_idx=n_pert_genes)

        self.encoder = nn.Sequential(
            nn.Linear(n_output_genes, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_var = nn.Linear(hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, n_output_genes),
        )

        self.direct_head = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, n_output_genes),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def get_pert_embedding(self, gene_idx):
        idx = gene_idx.clone()
        idx[idx < 0] = self.pad_idx
        embs = self.pert_emb(idx)
        mask = (gene_idx >= 0).unsqueeze(-1).float()
        return (embs * mask).sum(dim=1)

    def forward(self, gene_idx, target=None):
        delta_pert = self.get_pert_embedding(gene_idx)
        if target is not None and self.training:
            mu, logvar = self.encode(target)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z + delta_pert)
            direct = self.direct_head(delta_pert)
            return recon, direct, mu, logvar
        else:
            return self.direct_head(delta_pert)


class ScGPTPertPredictor(nn.Module):
    """
    Perturbation predictor initialized with pretrained scGPT embeddings.

    Projects pretrained 512-dim embeddings to 128, adds a learnable
    perturbation-specific embedding, runs through 3-layer transformer,
    and predicts expression changes.
    """

    def __init__(self, n_pert_genes, n_output_genes, pretrained_embs,
                 emb_dim=128, n_heads=4, n_layers=3, hidden=256):
        super().__init__()
        self.pad_idx = n_pert_genes

        src_dim = pretrained_embs.shape[1]
        self.emb_proj = nn.Linear(src_dim, emb_dim)
        self.pretrained_emb = nn.Parameter(
            torch.tensor(np.vstack([
                pretrained_embs,
                np.zeros((1, src_dim), dtype=np.float32),
            ])),
            requires_grad=False,
        )

        self.pert_emb = nn.Embedding(n_pert_genes + 1, emb_dim,
                                     padding_idx=n_pert_genes)
        nn.init.normal_(self.pert_emb.weight, std=0.02)

        self.pos_emb = nn.Parameter(torch.randn(2, emb_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=hidden,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, n_output_genes),
        )

    def forward(self, gene_idx):
        idx = gene_idx.clone()
        idx[idx < 0] = self.pad_idx
        raw_emb = self.pretrained_emb[idx]
        proj_emb = self.emb_proj(raw_emb)
        combined = proj_emb + self.pert_emb(idx) + self.pos_emb.unsqueeze(0)
        pad_mask = gene_idx < 0
        out = self.transformer(combined, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled)
