"""
analyze_shap.py — SHAP feature importance analysis for the GBM model.

Loads a trained GBM booster, runs TreeExplainer on validation data,
and outputs:
  1. SHAP summary bar plot  (mean |SHAP| per feature)
  2. SHAP beeswarm plot     (feature value vs SHAP impact)
  3. Structured JSON report  (ranked features + noise candidates)
  4. Feature correlation matrix for top features

Usage:
    python analyze_shap.py                                          # defaults
    python analyze_shap.py --model checkpoints/gbm_best.txt        # custom model
    python analyze_shap.py --data-dir data/cache --output shap_out  # custom paths
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _extract_arrays(loader) -> tuple[np.ndarray, np.ndarray]:
    """Pull (X, y) from a DataLoader into numpy arrays."""
    import torch
    Xs, ys = [], []
    for X_batch, y_batch in loader:
        Xs.append(X_batch.numpy())
        ys.append(y_batch.numpy())
    return np.vstack(Xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP feature importance analysis for GBM alpha scorer."
    )
    parser.add_argument("--model", default="checkpoints/gbm_best.txt",
                        help="Path to trained GBMScorer booster file.")
    parser.add_argument("--data-dir", default="data/cache",
                        help="Directory containing per-ticker parquet files.")
    parser.add_argument("--output", default="checkpoints/shap_report",
                        help="Output directory for SHAP plots and report.")
    parser.add_argument("--max-samples", type=int, default=50_000,
                        help="Max validation samples for SHAP (downsample if larger).")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    from model.gbm_scorer import GBMScorer
    model_path = Path(args.model)
    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        sys.exit(1)

    scorer = GBMScorer.load(str(model_path))
    booster = scorer._booster
    feature_names = scorer._feature_names
    log.info("Loaded GBM model: %d features, best_iter=%d, val_ic=%.4f",
             len(feature_names), scorer._best_iteration, scorer._val_ic)

    # ── Build dataset and extract val set ──────────────────────────────────────
    import config as cfg
    from data.dataset import build_regression_datasets

    log.info("Building regression datasets from %s...", args.data_dir)
    try:
        _, val_loader, _, _ = build_regression_datasets(
            data_dir=args.data_dir, config_module=cfg,
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    X_val, y_val = _extract_arrays(val_loader)
    log.info("Validation set: %d samples × %d features", X_val.shape[0], X_val.shape[1])

    # Downsample if too large (SHAP is O(n × trees))
    if X_val.shape[0] > args.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_val.shape[0], args.max_samples, replace=False)
        idx.sort()
        X_val = X_val[idx]
        y_val = y_val[idx]
        log.info("Downsampled to %d samples for SHAP", len(y_val))

    # ── Run SHAP ──────────────────────────────────────────────────────────────
    try:
        import shap
    except ImportError:
        log.error("shap not installed — run: pip install shap>=0.42.0")
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("matplotlib not installed — run: pip install matplotlib>=3.7.0")
        sys.exit(1)

    log.info("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_val)
    log.info("SHAP computation complete — shape: %s", shap_values.shape)

    # ── Feature ranking ───────────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_ranking = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: -x[1],
    )

    top_shap = feature_ranking[0][1]
    noise_threshold = 0.001 * top_shap  # per design doc §11.1.3

    print()
    print("=" * 65)
    print("  SHAP FEATURE IMPORTANCE RANKING")
    print("=" * 65)
    print(f"  {'Rank':<5} {'Feature':<22} {'Mean |SHAP|':>12}  {'Rel %':>7}  {'Status'}")
    print(f"  {'-'*5} {'-'*22} {'-'*12}  {'-'*7}  {'-'*10}")
    noise_features = []
    for rank, (feat, shap_val) in enumerate(feature_ranking, 1):
        rel_pct = 100.0 * shap_val / top_shap if top_shap > 0 else 0
        status = "✓" if shap_val >= noise_threshold else "⚠ NOISE"
        if shap_val < noise_threshold:
            noise_features.append(feat)
        print(f"  {rank:<5} {feat:<22} {shap_val:>12.6f}  {rel_pct:>6.1f}%  {status}")
    print("=" * 65)

    if noise_features:
        print(f"\n  ⚠ Noise feature candidates (mean|SHAP| < 0.1% of top): {noise_features}")
        print("  Consider removing these before Optuna tuning.\n")
    else:
        print("\n  ✓ No noise features detected — all features contribute meaningfully.\n")

    # ── Feature correlation matrix (top 10) ────────────────────────────────────
    from scipy.stats import pearsonr
    top_n = min(10, len(feature_names))
    top_features = [f for f, _ in feature_ranking[:top_n]]
    top_indices = [feature_names.index(f) for f in top_features]

    corr_matrix = np.corrcoef(X_val[:, top_indices].T)

    high_corr_pairs = []
    for i in range(top_n):
        for j in range(i + 1, top_n):
            rho = corr_matrix[i, j]
            if abs(rho) >= 0.7:
                high_corr_pairs.append((top_features[i], top_features[j], round(float(rho), 3)))

    if high_corr_pairs:
        print("  High-correlation pairs (|ρ| ≥ 0.7) among top-10 features:")
        for f1, f2, rho in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            print(f"    {f1:<22} ↔ {f2:<22}  ρ = {rho:+.3f}")
        print()

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=len(feature_names))
    plt.tight_layout()
    bar_path = out_dir / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("SHAP bar plot saved to %s", bar_path)

    # 2. Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, feature_names=feature_names,
                      show=False, max_display=len(feature_names))
    plt.tight_layout()
    bee_path = out_dir / "shap_beeswarm.png"
    plt.savefig(bee_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("SHAP beeswarm plot saved to %s", bee_path)

    # 3. JSON report
    report = {
        "model_path": str(model_path),
        "n_val_samples": int(X_val.shape[0]),
        "n_features": len(feature_names),
        "noise_threshold": float(noise_threshold),
        "feature_ranking": [
            {
                "rank": rank,
                "feature": feat,
                "mean_abs_shap": round(float(shap_val), 8),
                "relative_pct": round(100.0 * shap_val / top_shap, 2) if top_shap > 0 else 0,
                "is_noise": bool(shap_val < noise_threshold),
            }
            for rank, (feat, shap_val) in enumerate(feature_ranking, 1)
        ],
        "noise_features": noise_features,
        "high_correlation_pairs": [
            {"feature_1": f1, "feature_2": f2, "correlation": rho}
            for f1, f2, rho in high_corr_pairs
        ],
    }
    report_path = out_dir / "shap_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("SHAP report saved to %s", report_path)

    # 4. Gain-based importance comparison
    gain_importance = scorer.feature_importance(importance_type="gain")
    gain_ranking = sorted(gain_importance.items(), key=lambda x: -x[1])

    print("  SHAP vs Gain importance comparison (top 10):")
    print(f"  {'SHAP Rank':<12} {'Feature':<22} {'Gain Rank':<12}")
    print(f"  {'-'*12} {'-'*22} {'-'*12}")
    shap_rank_map = {feat: rank for rank, (feat, _) in enumerate(feature_ranking, 1)}
    gain_rank_map = {feat: rank for rank, (feat, _) in enumerate(gain_ranking, 1)}
    for rank, (feat, _) in enumerate(feature_ranking[:10], 1):
        gain_rank = gain_rank_map.get(feat, "—")
        print(f"  #{rank:<11} {feat:<22} #{gain_rank}")
    print()

    log.info("SHAP analysis complete. Outputs in %s/", out_dir)


if __name__ == "__main__":
    main()
