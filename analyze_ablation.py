"""
analyze_ablation.py — Feature ablation study for pruned features.

Adds back each of the 7 SHAP-pruned features one at a time to the 22-feature
baseline and runs walk-forward evaluation to measure per-fold IC impact.
This reveals which pruned features carry regime-specific signal.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python analyze_ablation.py
"""

import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# The 7 features pruned by SHAP analysis
PRUNED_FEATURES = [
    "macd_cross",
    "macd_above_zero",
    "avg_volume_20d",
    "rel_volume_ratio",
    "vol_ratio_10_60",
    "volume_trend",
    "volume_price_div",
]

BASELINE_FEATURES = list(cfg.FEATURES)  # current 22 features
ALL_29_FEATURES = BASELINE_FEATURES + PRUNED_FEATURES


def _make_config_with_features(feature_list):
    """Create a config-like namespace with a custom FEATURES list."""
    mock = SimpleNamespace()
    # Copy all config attributes
    for attr in dir(cfg):
        if not attr.startswith("_"):
            setattr(mock, attr, getattr(cfg, attr))
    mock.FEATURES = list(feature_list)
    mock.N_FEATURES = len(feature_list)
    return mock


def _run_walk_forward(X_all, y_all, all_dates, feature_names, gbm_params=None,
                      n_folds=5, test_days=60, forward_days=5):
    """Inline walk-forward to avoid coupling with train_gbm.py."""
    from model.gbm_scorer import GBMScorer
    from scipy.stats import pearsonr

    unique_dates = sorted(set(all_dates))
    n_dates = len(unique_dates)
    total_test_dates = test_days * n_folds

    if total_test_dates + forward_days >= n_dates:
        log.warning("Not enough dates for walk-forward")
        return [0.0] * n_folds

    test_start_date_idx = n_dates - total_test_dates
    fold_ics = []

    for fold in range(n_folds):
        test_date_start = test_start_date_idx + fold * test_days
        test_date_end = test_date_start + test_days
        test_date_set = set(unique_dates[test_date_start:test_date_end])

        train_end_date_idx = test_date_start - forward_days
        if train_end_date_idx <= 0:
            fold_ics.append(0.0)
            continue
        train_date_set = set(unique_dates[:train_end_date_idx])

        train_mask = np.array([d in train_date_set for d in all_dates])
        test_mask = np.array([d in test_date_set for d in all_dates])

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask], y_all[test_mask]

        if len(y_tr) < 1000 or len(y_te) < 100:
            fold_ics.append(0.0)
            continue

        # Sub-split for early stopping
        train_dates_sorted = sorted(train_date_set)
        sub_val_start = int(len(train_dates_sorted) * 0.85)
        sub_val_dates = set(train_dates_sorted[sub_val_start:])
        sub_train_mask = np.array([d not in sub_val_dates
                                   for d, m in zip(all_dates, train_mask) if m])
        sub_val_mask = ~sub_train_mask

        scorer = GBMScorer(params=gbm_params, n_estimators=2000, early_stopping_rounds=50)
        scorer.fit(X_tr[sub_train_mask], y_tr[sub_train_mask],
                   X_tr[sub_val_mask], y_tr[sub_val_mask],
                   feature_names=feature_names)

        preds = scorer.predict(X_te)
        ic, _ = pearsonr(preds, y_te)
        fold_ics.append(float(ic))

    return fold_ics


def main():
    from data.dataset import build_regression_arrays

    # Load best params from v1.5 tuning
    params_path = Path("checkpoints/gbm_best_params.json")
    if params_path.exists():
        gbm_params = json.loads(params_path.read_text())["best_params"]
        log.info("Using tuned params from %s", params_path)
    else:
        gbm_params = None
        log.info("No tuned params found, using defaults")

    # Build full dataset with all 29 features using a mock config
    log.info("Loading dataset with all 29 features...")
    mock_cfg = _make_config_with_features(ALL_29_FEATURES)
    X_all_29, y_all, all_dates = build_regression_arrays("data/cache", mock_cfg)
    log.info("Dataset: %d samples × %d features, %d unique dates",
             X_all_29.shape[0], X_all_29.shape[1], len(set(all_dates)))

    # Build feature index map for the 29-feature matrix
    feat_to_idx = {f: i for i, f in enumerate(ALL_29_FEATURES)}
    baseline_col_indices = [feat_to_idx[f] for f in BASELINE_FEATURES]

    # ── Baseline: 22 features ─────────────────────────────────────────────────
    log.info("Running baseline walk-forward (22 features)...")
    X_baseline = X_all_29[:, baseline_col_indices]
    baseline_ics = _run_walk_forward(
        X_baseline, y_all, all_dates, BASELINE_FEATURES, gbm_params)
    log.info("Baseline fold ICs: %s  mean=%.4f",
             [f"{x:.4f}" for x in baseline_ics], np.mean(baseline_ics))

    # ── Add-back ablation: +1 pruned feature at a time ────────────────────────
    results = {
        "baseline_22": {
            "features": BASELINE_FEATURES,
            "fold_ics": [round(x, 6) for x in baseline_ics],
            "mean_ic": round(float(np.mean(baseline_ics)), 6),
        }
    }

    for feat in PRUNED_FEATURES:
        feature_set = BASELINE_FEATURES + [feat]
        col_indices = baseline_col_indices + [feat_to_idx[feat]]
        X_test = X_all_29[:, col_indices]

        log.info("Testing +%s (23 features)...", feat)
        fold_ics = _run_walk_forward(X_test, y_all, all_dates, feature_set, gbm_params)

        delta_ics = [fold_ics[i] - baseline_ics[i] for i in range(len(fold_ics))]
        mean_delta = float(np.mean(delta_ics))

        results[feat] = {
            "fold_ics": [round(x, 6) for x in fold_ics],
            "mean_ic": round(float(np.mean(fold_ics)), 6),
            "delta_vs_baseline": [round(d, 6) for d in delta_ics],
            "mean_delta": round(mean_delta, 6),
        }
        log.info("  +%s: fold ICs=%s  mean=%.4f  delta=%+.4f",
                 feat, [f"{x:.4f}" for x in fold_ics], np.mean(fold_ics), mean_delta)

    # ── Also test all 29 together ─────────────────────────────────────────────
    log.info("Testing all 29 features together...")
    all29_ics = _run_walk_forward(X_all_29, y_all, all_dates, ALL_29_FEATURES, gbm_params)
    results["all_29"] = {
        "features": ALL_29_FEATURES,
        "fold_ics": [round(x, 6) for x in all29_ics],
        "mean_ic": round(float(np.mean(all29_ics)), 6),
        "delta_vs_baseline": [round(all29_ics[i] - baseline_ics[i], 6)
                              for i in range(len(all29_ics))],
    }
    log.info("All-29 fold ICs: %s  mean=%.4f",
             [f"{x:.4f}" for x in all29_ics], np.mean(all29_ics))

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  FEATURE ABLATION STUDY — Walk-Forward IC per Fold")
    print("=" * 100)
    header = f"  {'Config':<22} {'Fold1':>8} {'Fold2':>8} {'Fold3':>8} {'Fold4':>8} {'Fold5':>8} {'Mean':>8} {'Delta':>8}"
    print(header)
    print("  " + "-" * 96)

    # Baseline row
    b = baseline_ics
    print(f"  {'Baseline (22)':<22} {b[0]:>+8.4f} {b[1]:>+8.4f} {b[2]:>+8.4f} {b[3]:>+8.4f} {b[4]:>+8.4f} {np.mean(b):>+8.4f} {'---':>8}")

    # Per-feature rows
    for feat in PRUNED_FEATURES:
        r = results[feat]
        ics = r["fold_ics"]
        delta = r["mean_delta"]
        marker = " ★" if delta > 0.003 else " ✗" if delta < -0.003 else ""
        print(f"  +{feat:<21} {ics[0]:>+8.4f} {ics[1]:>+8.4f} {ics[2]:>+8.4f} {ics[3]:>+8.4f} {ics[4]:>+8.4f} {np.mean(ics):>+8.4f} {delta:>+8.4f}{marker}")

    # All-29 row
    a = all29_ics
    all_delta = float(np.mean(a)) - float(np.mean(b))
    print("  " + "-" * 96)
    print(f"  {'All 29 together':<22} {a[0]:>+8.4f} {a[1]:>+8.4f} {a[2]:>+8.4f} {a[3]:>+8.4f} {a[4]:>+8.4f} {np.mean(a):>+8.4f} {all_delta:>+8.4f}")
    print("=" * 100)
    print("  ★ = adds >0.003 mean IC   ✗ = hurts >0.003 mean IC")
    print()

    # ── Per-fold delta heatmap ────────────────────────────────────────────────
    print("  PER-FOLD DELTA vs BASELINE (green = helps, red = hurts):")
    print(f"  {'Feature':<22} {'Fold1':>8} {'Fold2':>8} {'Fold3':>8} {'Fold4':>8} {'Fold5':>8}")
    print("  " + "-" * 62)
    for feat in PRUNED_FEATURES:
        deltas = results[feat]["delta_vs_baseline"]
        cells = []
        for d in deltas:
            if d > 0.005:
                cells.append(f"\033[92m{d:>+8.4f}\033[0m")  # green
            elif d < -0.005:
                cells.append(f"\033[91m{d:>+8.4f}\033[0m")  # red
            else:
                cells.append(f"{d:>+8.4f}")
        print(f"  {feat:<22} {''.join(cells)}")
    print()

    # Save report
    report_path = Path("checkpoints/ablation_report.json")
    report_path.write_text(json.dumps(results, indent=2))
    log.info("Ablation report saved to %s", report_path)


if __name__ == "__main__":
    main()
