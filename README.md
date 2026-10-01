# Do Perturbation Predictors Learn Gene Interactions? Diagnosing Extrapolation in Agentic Screening

Accepted at the NeurIPS 2026 Workshop ML4Molecules: Agentic Systems for Molecular Sciences.

## Repository structure

```
models.py                     Model architectures (Table 7)
data_utils.py                 Shared data loading, splits, training loops
reliability.py                Reliability signal computation (Table 8)

prepare_data.py               Download datasets and create evaluation splits
run_norman_baselines.py       Table 1 (Norman), Table 2, Table 9, Table 10
run_gears_ablation.py         Table 5: GEARS graph ablations
run_joungzhang.py             Table 1 (Joung-Zhang 2023), Table 3 (Joung-Zhang rows)
run_adamson.py                Table 3 (Adamson rows)
run_fallback_analysis.py      Table 4: Reliability-aware fallback
run_gene_level_analysis.py    Section 3.3: Gene-level analysis, Table 6
run_hypothesis_analysis.py    Section 4: Discussion hypotheses
run_metric_sensitivity.py     Table 11: Signal combinations, Table 12: Metric sensitivity
```

## Setup

### Dependencies

```bash
pip install gears torch numpy scipy scikit-learn anndata scanpy
```

### Datasets

Three perturbation datasets are used. Norman and Adamson are downloaded
automatically; the Joung-Zhang 2023 combinatorial dataset requires a manual download.

| Dataset | Cell line | Perturbation type | How to obtain |
|---------|-----------|-------------------|---------------|
| Norman et al. 2019 | K562 (CRISPRa) | 131 single + 131 double | Auto downloaded by `gears.PertData.load("norman")` |
| Adamson et al. 2016 | K562 (CRISPRi) | Single gene OOD | Auto downloaded by `gears.PertData.load("adamson")` |
| Joung-Zhang 2023 combinatorial | See source dataset metadata | Combinatorial perturbations | Manual download (see below) |

**Joung-Zhang 2023 combinatorial dataset**: Download `JoungZhang2023_combinatorial.h5ad` and place it at
`data/joungzhang/JoungZhang2023_combinatorial.h5ad`. Cite Joung, Zhang et al. (2023),
*A transcription factor atlas of directed differentiation* (Cell), as the dataset source.
If you skip this step, only `run_joungzhang.py` will be affected; all other
scripts work without it.

### Pretrained weights (optional)

scGPT and GEARS+scGPT experiments require pretrained scGPT embeddings. If the
weights are not present, these two models are skipped automatically and the
remaining six models still run normally.

To enable them, download the **scGPT whole-human** checkpoint from the
[scGPT GitHub](https://github.com/bowang-lab/scGPT) (Google Drive link in
their README) and place the files so that the directory looks like:

```
data/scgpt_pretrained/
    best_model.pt
    vocab.json
```

### Data preparation

```bash
python prepare_data.py
```

This downloads Norman and Adamson via the GEARS package (first run takes a few
minutes) and creates train/val/test splits for three seeds (42, 43, 44). Joung-Zhang 2023
uses leave-one-out CV so its splits are created at runtime.

## Running experiments

Each script corresponds to one or more tables in the paper. Scripts can be run independently; later scripts (fallback, hypothesis) read results saved by earlier ones.

### Core experiments

```bash
# Table 1 (Norman), Table 2, Table 9, Table 10
CUDA_VISIBLE_DEVICES=0 python run_norman_baselines.py

# Table 1 (Joung-Zhang 2023), Table 3 (Joung-Zhang rows)
CUDA_VISIBLE_DEVICES=0 python run_joungzhang.py

# Table 3 (Adamson rows)
CUDA_VISIBLE_DEVICES=0 python run_adamson.py

# Table 5: GEARS ablations
CUDA_VISIBLE_DEVICES=0 python run_gears_ablation.py
```

### Analysis scripts (no GPU needed for most)

```bash
# Table 4: Reliability-aware fallback
python run_fallback_analysis.py

# Section 3.3: Gene-level analysis, Table 6
CUDA_VISIBLE_DEVICES=0 python run_gene_level_analysis.py

# Section 4: Hypothesis testing
python run_hypothesis_analysis.py

# Table 11, Table 12: Signal combinations and metric sensitivity
CUDA_VISIBLE_DEVICES=0 python run_metric_sensitivity.py
```

## Script-to-table mapping

| Script | Paper content |
|--------|--------------|
| `run_norman_baselines.py` | Table 1 (Norman cols), Table 2, Table 9 (Appendix C), Table 10 (Appendix C) |
| `run_joungzhang.py` | Table 1 (Joung-Zhang cols), Table 3 (Joung-Zhang rows) |
| `run_adamson.py` | Table 3 (Adamson rows) |
| `run_fallback_analysis.py` | Table 4 |
| `run_gears_ablation.py` | Table 5 |
| `run_gene_level_analysis.py` | Section 3.3 gene-level, Table 6, Figure 3 data |
| `run_hypothesis_analysis.py` | Section 4 discussion |
| `run_metric_sensitivity.py` | Table 11 (Appendix D), Table 12 (Appendix D) |

## Models

All model architectures are defined in `models.py`:

| Model | Class | Description |
|-------|-------|-------------|
| MLP | `PertMLP` | 64-dim embeddings, 2x256 hidden, ReLU, LayerNorm, dropout 0.1 |
| Transformer | `PertTransformer` | 64-dim, 4 heads, 2 layers, pre-norm, mean pooling |
| CPA | `CompositionalPertVAE` | VAE with additive perturbation composition in latent space |
| scGPT (f.t.) | `ScGPTPertPredictor` | Transformer initialized from pretrained scGPT embeddings |
| GEARS | via `gears` package | GNN with GO + co-expression graphs |
| Ridge | `sklearn.linear_model.Ridge` | L2-regularized linear model on one-hot gene indicators |
| Additive | n/a | Sum of observed single-gene effects |

## Reliability signals

All nine signals (Table 8) are implemented in `reliability.py`:

| Signal | Function | Source |
|--------|----------|--------|
| Additive disagreement (AD) | `compute_additive_disagreement` | Prior-based |
| Prediction magnitude (PM) | `compute_prediction_magnitude` | Output-based |
| Gene-wise variance (GV) | `compute_genewise_variance` | Output-based |
| Ensemble variance | `compute_ensemble_variance` | Output-based |
| kNN distance | `compute_knn_distance` | Embedding-based |
| Mahalanobis distance | `compute_mahalanobis_distance` | Embedding-based |
| NN error proxy | `compute_nn_error_proxy` | Embedding-based |
| Cosine to nearest | `compute_cosine_to_nearest` | Embedding-based |
| Random | n/a | Baseline |
