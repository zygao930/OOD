import os
import json
import warnings
from itertools import combinations

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
os.makedirs("results/hypothesis_analysis", exist_ok=True)

from data_utils import pair_genes


###############################################################################
# Dataset parsing helpers
###############################################################################

def _mean_expr(adata, cond_key, cond):
    mask = adata.obs[cond_key] == cond
    if mask.sum() == 0:
        return None
    X = adata[mask].X
    return np.array(
        X.toarray().mean(0) if hasattr(X, "toarray") else X.mean(0)
    ).flatten()


def parse_norman(adata, cond_key="condition"):
    """Return (singles, eligible) dicts of effect vectors for Norman."""
    ctrl_mean = _mean_expr(adata, cond_key, "ctrl")
    if ctrl_mean is None:
        ctrl_mean = np.zeros(adata.n_vars)
    singles, pairs = {}, {}
    for c in adata.obs[cond_key].unique():
        s = str(c)
        if s == "ctrl" or "+" not in s:
            continue
        genes = [g.strip() for g in s.split("+") if g.strip()]
        if len(genes) != 2:
            continue
        real = [g for g in genes if g != "ctrl"]
        m = _mean_expr(adata, cond_key, c)
        if m is None:
            continue
        eff = m - ctrl_mean
        if len(real) == 1:
            singles[real[0]] = eff
        elif len(real) == 2:
            pairs[tuple(sorted(real))] = eff
    eligible = {k: v for k, v in pairs.items() if k[0] in singles and k[1] in singles}
    return singles, eligible


def parse_joungzhang(adata, cond_key):
    """Return (singles, eligible) dicts of effect vectors for Joung-Zhang 2023.

    GFP is treated as a control/filler: GFP+GENE is a single, GENE1+GENE2
    (no GFP) is a double. The baseline is the pure GFP control if present,
    otherwise the mean of all GFP+GENE cells.
    """
    CTRL = "GFP"
    ctrl_labels = ["ctrl", "control", "non-targeting"]

    ctrl_mean = None
    for lab in [CTRL] + ctrl_labels:
        m = _mean_expr(adata, cond_key, lab)
        if m is not None:
            ctrl_mean = m
            break
    if ctrl_mean is None:
        acc = np.zeros(adata.n_vars); n = 0
        for c in adata.obs[cond_key].unique():
            parts = [p.strip() for p in str(c).split("+") if p.strip()]
            real = [p for p in parts if p != CTRL and p not in ctrl_labels]
            if len(real) == 1 and (CTRL in parts or any(p in ctrl_labels for p in parts)):
                mask = adata.obs[cond_key] == c
                if mask.sum() > 0:
                    X = adata[mask].X
                    acc += np.array(
                        X.toarray().mean(0) if hasattr(X, "toarray") else X.mean(0)
                    ).flatten() * mask.sum()
                    n += mask.sum()
        ctrl_mean = acc / max(n, 1)

    singles, pairs = {}, {}
    for c in adata.obs[cond_key].unique():
        parts = [p.strip() for p in str(c).split("+") if p.strip()]
        real = [p for p in parts if p != CTRL and p not in ctrl_labels]
        m = _mean_expr(adata, cond_key, c)
        if m is None:
            continue
        eff = m - ctrl_mean
        if len(real) == 1 and (CTRL in parts or len(parts) == 1):
            singles[real[0]] = eff
        elif len(real) == 2 and CTRL not in parts:
            pairs[tuple(sorted(real))] = eff
    eligible = {k: v for k, v in pairs.items() if k[0] in singles and k[1] in singles}
    return singles, eligible


###############################################################################
# Hypotheses
###############################################################################

def _model_keys(records, prefix):
    """Collect model names across the UNION of keys in all records (not just
    records[0]), so a model missing from the first record is not dropped."""
    keys = set()
    for r in records:
        for k in r:
            if k.startswith(prefix) and k != "error_additive":
                keys.add(k[len(prefix):])
    return sorted(keys)


def run_h1(singles, eligible, name):
    print(f"\n{'=' * 60}\nH1: Are true interactions small? ({name})\n{'=' * 60}")
    results = {}
    pairs = sorted(eligible.keys())
    i_mags, a_mags, e_mags = [], [], []
    for g1, g2 in pairs:
        add = singles[g1] + singles[g2]
        true = eligible[(g1, g2)]
        i_mags.append(float(np.linalg.norm(true - add, 1)))
        a_mags.append(float(np.linalg.norm(add, 1)))
        e_mags.append(float(np.linalg.norm(true, 1)))
    if not pairs:
        print("  No eligible pairs."); return results
    i_mags = np.array(i_mags); e_mags = np.array(e_mags)
    fracs = i_mags / (e_mags + 1e-10)
    results.update({
        "n": len(pairs),
        "i_true_mean": float(np.mean(i_mags)),
        "i_true_median": float(np.median(i_mags)),
        "additive_mean": float(np.mean(a_mags)),
        "frac_mean": float(np.mean(fracs)),
        "frac_median": float(np.median(fracs)),
        "below_10pct": float(np.mean(fracs < 0.10)),
        "below_25pct": float(np.mean(fracs < 0.25)),
    })
    print(f"  ||I_true||: mean={results['i_true_mean']:.4f} median={results['i_true_median']:.4f}")
    print(f"  Interaction fraction: mean={results['frac_mean']:.1%} median={results['frac_median']:.1%}")
    print(f"  Pairs <10%: {results['below_10pct']:.1%}  <25%: {results['below_25pct']:.1%}")
    return results


def run_h2(singles, eligible, records, name):
    print(f"\n{'=' * 60}\nH2: Excess error vs true interaction ({name})\n{'=' * 60}")
    results = {}
    itrue = {}
    for g1, g2 in eligible:
        itrue[(g1, g2)] = float(np.linalg.norm(eligible[(g1, g2)] - singles[g1] - singles[g2], 1))
    if not records:
        print("  No records available."); return results
    model_keys = _model_keys(records, "error_")
    for m in model_keys:
        ek = f"error_{m}"
        itr, exc = [], []
        for r in records:
            if r.get(ek) is None or r.get("error_additive") is None:
                continue
            genes = tuple(sorted(pair_genes(str(r.get("pair", "")))))
            if genes not in itrue:
                continue
            itr.append(itrue[genes])
            exc.append(r[ek] - r["error_additive"])
        if len(itr) < 5:
            continue
        rho_e, p_e = stats.spearmanr(itr, exc)
        results[m] = {"rho_excess": float(rho_e), "p_excess": float(p_e), "n": len(itr)}
        print(f"  {m} (n={len(itr)}): corr(||I_true||, excess): rho={rho_e:.3f} (p={p_e:.4f})")
    return results


def run_h3(singles, eligible, name):
    print(f"\n{'=' * 60}\nH3: Learnable structure in interactions? ({name})\n{'=' * 60}")
    results = {}
    pairs = sorted(eligible.keys())
    n_p = len(pairs)
    if n_p < 10:
        print("  Too few pairs."); return results
    ivecs, sprofs = [], []
    for g1, g2 in pairs:
        add = singles[g1] + singles[g2]
        ivecs.append(eligible[(g1, g2)] - add)
        sprofs.append(np.concatenate([singles[g1], singles[g2]]))
    ivecs = np.array(ivecs); sprofs = np.array(sprofs)

    cpairs = list(combinations(range(n_p), 2))
    if len(cpairs) > 5000:
        rng = np.random.default_rng(42)
        cpairs = [cpairs[i] for i in rng.choice(len(cpairs), 5000, replace=False)]
    ss, ii, sf = [], [], []
    for i, j in cpairs:
        n1, n2 = np.linalg.norm(sprofs[i]), np.linalg.norm(sprofs[j])
        ss.append(np.dot(sprofs[i], sprofs[j]) / (n1 * n2 + 1e-10))
        n1, n2 = np.linalg.norm(ivecs[i]), np.linalg.norm(ivecs[j])
        ii.append(np.dot(ivecs[i], ivecs[j]) / (n1 * n2 + 1e-10))
        sf.append(len(set(pairs[i]) & set(pairs[j])) > 0)
    rho_s, p_s = stats.spearmanr(ss, ii)
    results["rho_single_vs_interaction_sim"] = float(rho_s)
    print(f"  3a: single-gene sim -> interaction sim: rho={rho_s:.3f} (p={p_s:.4f})")

    sf = np.array(sf, bool); ii = np.array(ii)
    if sf.sum() > 5 and (~sf).sum() > 5:
        results["shared_mean"] = float(np.mean(ii[sf]))
        results["unshared_mean"] = float(np.mean(ii[~sf]))
        _, tp = stats.ttest_ind(ii[sf], ii[~sf])
        results["shared_p"] = float(tp)
        print(f"  3b: shared={results['shared_mean']:.4f} unshared={results['unshared_mean']:.4f} (p={tp:.4f})")

    smags, imags = [], []
    for g1, g2 in pairs:
        smags.append(np.linalg.norm(singles[g1], 1) + np.linalg.norm(singles[g2], 1))
        imags.append(np.linalg.norm(eligible[(g1, g2)] - singles[g1] - singles[g2], 1))
    rho_m, p_m = stats.spearmanr(smags, imags)
    results["rho_mag"] = float(rho_m)
    print(f"  3c: single mag -> interaction mag: rho={rho_m:.3f} (p={p_m:.4f})")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_p, 20)); pca.fit(ivecs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    results["pc1_var"] = float(pca.explained_variance_ratio_[0])
    results["n_50pct"] = int(np.searchsorted(cumvar, 0.50)) + 1
    results["n_90pct"] = int(np.searchsorted(cumvar, 0.90)) + 1
    print(f"  3d: PC1={results['pc1_var']:.1%}, {results['n_50pct']} for 50%, {results['n_90pct']} for 90%")
    return results


def run_h4(adata, singles, eligible, records, name):
    print(f"\n{'=' * 60}\nH4: High-degree genes as shortcuts? ({name})\n{'=' * 60}")
    results = {}
    pairs = sorted(eligible.keys())
    if "gene_name" in adata.var.columns:
        gene2idx = {g: i for i, g in enumerate(adata.var["gene_name"])}
    else:
        gene2idx = {g: i for i, g in enumerate(adata.var_names)}

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    if X.shape[0] > 5000:
        rng = np.random.default_rng(0)
        X = X[rng.choice(X.shape[0], 5000, replace=False)]
    gvar = np.var(X, axis=0)
    valid_cols = np.where(gvar > 0)[0]
    if len(valid_cols) == 0:
        print("  No variable genes."); return results
    rng = np.random.default_rng(0)
    targets = rng.choice(valid_cols, min(500, len(valid_cols)), replace=False)

    print(f"  Computing co-expression degree proxy for {len(singles)} genes...")
    deg = {}
    for gene in singles:
        if gene not in gene2idx:
            continue
        gi = gene2idx[gene]
        if gvar[gi] == 0:
            continue
        corrs = [abs(np.corrcoef(X[:, gi], X[:, t])[0, 1]) for t in targets if t != gi]
        if corrs:
            deg[gene] = float(np.nanmean(corrs))
    if len(deg) < 5:
        print("  Too few genes with degree."); return results

    pd_, pe, pi = [], [], []
    for g1, g2 in pairs:
        if g1 not in deg or g2 not in deg:
            continue
        pd_.append((deg[g1] + deg[g2]) / 2)
        add = singles[g1] + singles[g2]
        pe.append(np.mean(np.abs(add - eligible[(g1, g2)])))
        pi.append(np.linalg.norm(eligible[(g1, g2)] - add, 1))
    if len(pd_) > 5:
        rho_e, p_e = stats.spearmanr(pd_, pe)
        rho_i, p_i = stats.spearmanr(pd_, pi)
        results["rho_degree_add_error"] = float(rho_e)
        results["rho_degree_interaction"] = float(rho_i)
        print(f"  corr(degree, additive error): rho={rho_e:.3f} (p={p_e:.4f})")
        print(f"  corr(degree, true interaction): rho={rho_i:.3f} (p={p_i:.4f})")

    if records:
        model_keys = _model_keys(records, "ad_")
        for m in model_keys:
            md, mc, me = [], [], []
            for r in records:
                genes = tuple(sorted(pair_genes(str(r.get("pair", "")))))
                if len(genes) != 2 or genes[0] not in deg or genes[1] not in deg:
                    continue
                av = r.get(f"ad_{m}")
                ev = r.get(f"error_{m}")
                if av is None or ev is None:
                    continue
                md.append((deg[genes[0]] + deg[genes[1]]) / 2)
                mc.append(av); me.append(ev)
            if len(md) > 5:
                rc, _ = stats.spearmanr(md, mc)
                re_, _ = stats.spearmanr(md, me)
                results[f"{m}_degree_correction"] = float(rc)
                results[f"{m}_degree_error"] = float(re_)
                print(f"  {m}: degree vs correction={rc:.3f}, degree vs error={re_:.3f}")
    return results


def run_h5(singles, eligible, records, name):
    print(f"\n{'=' * 60}\nH5: Metric sensitivity ({name})\n{'=' * 60}")
    results = {}
    if not records:
        print("  No records."); return results
    model_keys = _model_keys(records, "ad_")
    for m in model_keys:
        ek, tk, adk, pmk = f"error_{m}", f"top20_{m}", f"ad_{m}", f"pm_{m}"
        valid = [r for r in records
                 if r.get(ek) is not None and r.get(adk) is not None]
        if len(valid) < 5:
            continue
        ef = np.array([r[ek] for r in valid])
        ad = np.array([r[adk] for r in valid])
        pm = np.array([r.get(pmk, 0) for r in valid])
        entry = {"AD_full": float(stats.spearmanr(ad, ef)[0]),
                 "PM_full": float(stats.spearmanr(pm, ef)[0]), "n": len(valid)}
        t20valid = [r for r in valid if r.get(tk) is not None]
        if len(t20valid) >= 5:
            et = np.array([r[tk] for r in t20valid])
            adt = np.array([r[adk] for r in t20valid])
            pmt = np.array([r.get(pmk, 0) for r in t20valid])
            entry["AD_top20"] = float(stats.spearmanr(adt, et)[0])
            entry["PM_top20"] = float(stats.spearmanr(pmt, et)[0])
        results[m] = entry
        print(f"  {m} (n={len(valid)}): AD_full={entry['AD_full']:.3f}"
              + (f" AD_top20={entry['AD_top20']:.3f}" if 'AD_top20' in entry else ""))
    return results


###############################################################################
# Load records
###############################################################################

def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


norman_records = _load("results/norman/all_records.json")
joungzhang_records = _load("results/joungzhang/records.json")
print(f"Loaded {len(norman_records)} Norman records, {len(joungzhang_records)} Joung-Zhang records")


###############################################################################
# Run on both datasets
###############################################################################

all_results = {}

# ---- Norman ----
print("\n" + "#" * 60 + "\n# NORMAN\n" + "#" * 60)
try:
    from gears import PertData
    pert_data = PertData("./data")
    pert_data.load(data_name="norman")
    adata_n = pert_data.adata
    singles_n, eligible_n = parse_norman(adata_n, "condition")
    print(f"  {len(singles_n)} singles, {len(eligible_n)} eligible pairs")
    all_results["norman"] = {
        "h1": run_h1(singles_n, eligible_n, "norman"),
        "h2": run_h2(singles_n, eligible_n, norman_records, "norman"),
        "h3": run_h3(singles_n, eligible_n, "norman"),
        "h4": run_h4(adata_n, singles_n, eligible_n, norman_records, "norman"),
        "h5": run_h5(singles_n, eligible_n, norman_records, "norman"),
    }
    del adata_n
except Exception as e:
    print(f"  Norman analysis failed: {e}")

# ---- Joung-Zhang 2023 ----
print("\n" + "#" * 60 + "\n# JOUNG-ZHANG 2023\n" + "#" * 60)
try:
    import anndata as ad
    cpath = "data/joungzhang/JoungZhang2023_combinatorial.h5ad"
    if os.path.exists(cpath):
        adata_c = ad.read_h5ad(cpath)
        cond_key_c = "perturbation" if "perturbation" in adata_c.obs.columns else "condition"
        singles_c, eligible_c = parse_joungzhang(adata_c, cond_key_c)
        print(f"  {len(singles_c)} singles, {len(eligible_c)} eligible pairs")
        all_results["joungzhang"] = {
            "h1": run_h1(singles_c, eligible_c, "joungzhang"),
            "h2": run_h2(singles_c, eligible_c, joungzhang_records, "joungzhang"),
            "h3": run_h3(singles_c, eligible_c, "joungzhang"),
            "h4": run_h4(adata_c, singles_c, eligible_c, joungzhang_records, "joungzhang"),
            "h5": run_h5(singles_c, eligible_c, joungzhang_records, "joungzhang"),
        }
        del adata_c
    else:
        print(f"  Joung-Zhang data not found: {cpath}")
except Exception as e:
    print(f"  Joung-Zhang analysis failed: {e}")


###############################################################################
# Save
###############################################################################

with open("results/hypothesis_analysis/results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved results/hypothesis_analysis/results.json "
      f"({len(all_results)} datasets)")
print("DONE")
