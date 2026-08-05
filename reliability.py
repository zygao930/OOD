import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_additive_disagreement(pred, additive_pred):
    """AD = ||f(A,B) - (dA + dB)||_1  (Eq. 2)."""
    return float(np.mean(np.abs(pred - additive_pred)))


def compute_prediction_magnitude(pred):
    """PM = ||f(A,B)||_1."""
    return float(np.mean(np.abs(pred)))


def compute_genewise_variance(pred):
    """GV = Var_g[f(A,B)_g]."""
    return float(np.var(pred))


def compute_knn_distance(emb, nn_model):
    """Mean cosine distance to k nearest training pairs."""
    if emb is None:
        return None
    dists, _ = nn_model.kneighbors(emb.reshape(1, -1))
    return float(dists.mean())


def compute_mahalanobis_distance(emb, train_emb_mean, cov_inv):
    """Mahalanobis distance from training distribution."""
    if emb is None:
        return None
    try:
        return float(mahalanobis(emb, train_emb_mean, cov_inv))
    except Exception:
        return None


def compute_ensemble_variance(ensemble_preds):
    """Mean per-gene standard deviation across ensemble members."""
    if len(ensemble_preds) < 2:
        return None
    stack = np.array(ensemble_preds)
    return float(np.mean(np.std(stack, axis=0)))


def compute_nn_error_proxy(emb, proxy_nn, proxy_errors):
    """Weighted error of nearest training neighbors."""
    if emb is None:
        return None
    dists, idxs = proxy_nn.kneighbors(emb.reshape(1, -1))
    weights = 1.0 / (dists[0] + 1e-8)
    weights = weights / weights.sum()
    return float(np.sum(weights * proxy_errors[idxs[0]]))


def compute_cosine_to_nearest(pred, training_preds):
    """1 - cosine similarity to nearest training prediction."""
    best_cos_dist = float("inf")
    pred_norm = np.linalg.norm(pred)
    if pred_norm < 1e-12:
        return None
    for tp in training_preds:
        tp_norm = np.linalg.norm(tp)
        if tp_norm < 1e-12:
            continue
        cos_sim = np.dot(pred, tp) / (pred_norm * tp_norm)
        cos_dist = 1.0 - cos_sim
        if cos_dist < best_cos_dist:
            best_cos_dist = cos_dist
    return float(best_cos_dist) if best_cos_dist < float("inf") else None


# ---------------------------------------------------------------------------
# kNN model setup
# ---------------------------------------------------------------------------

def build_knn_model(embeddings, n_neighbors=5):
    """Fit a NearestNeighbors model on training pair embeddings."""
    arr = np.array(embeddings)
    k = min(n_neighbors, len(arr))
    return NearestNeighbors(n_neighbors=k, metric="cosine").fit(arr)


def build_mahalanobis(embeddings):
    """Fit covariance for Mahalanobis distance. Returns (mean, cov_inv) or None."""
    arr = np.array(embeddings)
    if len(arr) <= 2:
        return None
    try:
        cov_est = EmpiricalCovariance().fit(arr)
        mean = np.mean(arr, axis=0)
        cov_inv = np.linalg.inv(
            cov_est.covariance_ + 1e-6 * np.eye(arr.shape[1])
        )
        return mean, cov_inv
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def failure_auroc(risk_scores, errors, threshold_pctl=75):
    """
    Failure detection AUROC: how well does the risk score identify
    predictions above the given error percentile?
    """
    threshold = np.percentile(errors, threshold_pctl)
    y_bin = (errors > threshold).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return None
    return float(roc_auc_score(y_bin, risk_scores))


def spearman_correlation(risk_scores, errors):
    """Spearman rank correlation between risk scores and prediction errors."""
    rho, _ = stats.spearmanr(risk_scores, errors)
    return float(rho)


def abstention_improvement(risk_scores, errors, abstention_rate=0.2):
    """
    Relative MAE improvement when removing the highest-risk fraction
    of predictions.
    """
    n = len(risk_scores)
    n_keep = max(1, int((1 - abstention_rate) * n))
    order = np.argsort(risk_scores)
    kept_errors = errors[order[:n_keep]]
    full_mae = np.mean(errors)
    kept_mae = np.mean(kept_errors)
    if full_mae == 0:
        return 0.0
    return float((1 - kept_mae / full_mae) * 100)
