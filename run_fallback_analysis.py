import json
import numpy as np

def run_fallback(recs, model_name, dataset_name, error_key, add_key, signals):
    """Sweep routing thresholds and report best MAE for each signal."""
    valid = [r for r in recs if r.get(error_key) is not None and r.get(add_key) is not None]
    if not valid:
        return

    pcts = np.linspace(0.05, 0.95, 19)
    always_model = np.mean([r[error_key] for r in valid])
    always_add = np.mean([r[add_key] for r in valid])
    oracle = np.mean([min(r[error_key], r[add_key]) for r in valid])
    wins = sum(1 for r in valid if r[error_key] < r[add_key])

    print(f"\n{'=' * 70}")
    print(f"{model_name} on {dataset_name} ({len(valid)} evaluations, "
          f"wins: {wins}/{len(valid)})")
    print(f"{'=' * 70}")
    print(f"{'Policy':<40} {'MAE':>8}")
    print(f"{'No routing (always model)':<40} {always_model:>8.4f}")
    print(f"{'No routing (always additive)':<40} {always_add:>8.4f}")

    for sig_name, sig_key in signals:
        vals = [r.get(sig_key) for r in valid]
        if None in vals:
            continue
        vals_arr = np.array(vals)

        best_mae = float("inf")
        for p in pcts:
            thresh = np.percentile(vals_arr, p * 100)
            routed = [
                r[error_key] if r[sig_key] <= thresh else r[add_key]
                for r in valid
            ]
            mae = np.mean(routed)
            if mae < best_mae:
                best_mae = mae

        print(f"  Route by {sig_name:<34} {best_mae:>8.4f}")

    print(f"{'Oracle routing':<40} {oracle:>8.4f}")


###############################################################################
# Norman
###############################################################################

print("\n" + "#" * 70)
print("#  NORMAN (Table 4, top)")
print("#" * 70)

try:
    with open("results/norman/all_records.json") as f:
        recs = json.load(f)
    print(f"Loaded {len(recs)} Norman records")

    for model_name, err_key, ad_key, pm_key, gv_key, knn_key in [
        ("Ridge", "error_ridge", "ad_ridge", "pm_ridge", "gv_ridge", "knn_gears"),
        ("CPA", "error_cpa", "ad_cpa", "pm_cpa", "gv_cpa", "knn_cpa"),
        ("MLP", "error_mlp", "ad_mlp", "pm_mlp", "gv_mlp", "knn_mlp"),
        ("Transformer", "error_transformer", "ad_transformer", "pm_transformer", "gv_transformer", "knn_transformer"),
        ("scGPT", "error_scgpt", "ad_scgpt", "pm_scgpt", "gv_scgpt", "knn_scgpt"),
        ("GEARS+scGPT", "error_gears_scgpt", "ad_gears_scgpt", "pm_gears_scgpt", "gv_gears_scgpt", "knn_gears_scgpt"),
        ("GEARS", "error_gears", "ad_gears", "pm_gears", "gv_gears", "knn_gears"),
    ]:
        run_fallback(recs, model_name, "Norman", err_key, "error_additive", [
            ("additive disagreement", ad_key),
            ("prediction magnitude", pm_key),
            ("gene-wise variance", gv_key),
            ("kNN distance", knn_key),
        ])

except FileNotFoundError:
    print("  Norman results not found. Run run_norman_baselines.py first.")


###############################################################################
# Joung-Zhang 2023
###############################################################################

print("\n\n" + "#" * 70)
print("#  JOUNG-ZHANG 2023 (Table 4, bottom)")
print("#" * 70)

try:
    with open("results/joungzhang/records.json") as f:
        recs = json.load(f)
    print(f"Loaded {len(recs)} Joung-Zhang records")

    for model_name, err_key, ad_key, pm_key, gv_key, knn_key in [
        ("Ridge", "error_ridge", "ad_ridge", "pm_ridge", "gv_ridge", "knn_ridge"),
        ("CPA", "error_cpa", "ad_cpa", "pm_cpa", "gv_cpa", "knn_cpa"),
        ("MLP", "error_mlp", "ad_mlp", "pm_mlp", "gv_mlp", "knn_mlp"),
        ("Transformer", "error_transformer", "ad_transformer", "pm_transformer", "gv_transformer", "knn_transformer"),
        ("scGPT", "error_scgpt", "ad_scgpt", "pm_scgpt", "gv_scgpt", "knn_scgpt"),
        ("GEARS+scGPT", "error_gears_scgpt", "ad_gears_scgpt", "pm_gears_scgpt", "gv_gears_scgpt", "knn_gears_scgpt"),
        ("GEARS", "error_gears", "ad_gears", "pm_gears", "gv_gears", "knn_gears"),
    ]:
        run_fallback(recs, model_name, "Joung-Zhang 2023", err_key, "error_additive", [
            ("additive disagreement", ad_key),
            ("prediction magnitude", pm_key),
            ("gene-wise variance", gv_key),
            ("kNN distance", knn_key),
        ])

except FileNotFoundError:
    print("  Joung-Zhang results not found. Run run_joungzhang.py first.")

print("\nDONE")
