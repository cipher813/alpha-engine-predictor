"""
config.py — Central configuration for alpha-engine-predictor.

All S3 paths, model hyperparameters, feature definitions, and production
gates live here. Import this module everywhere rather than hard-coding values.

Tunable parameters are loaded from config/predictor.yaml (gitignored).
Copy config/predictor.sample.yaml to get started.
"""

import os
from pathlib import Path

import yaml

# ── Load predictor config YAML ────────────────────────────────────────────────
_CONFIG_DIR = Path(__file__).parent / "config"

_CONFIG_SEARCH = [
    Path.home() / "alpha-engine-config" / "predictor" / "predictor.yaml",
    Path(__file__).parent.parent / "alpha-engine-config" / "predictor" / "predictor.yaml",
    _CONFIG_DIR / "predictor.yaml",
]
_CONFIG_PATH = next((p for p in _CONFIG_SEARCH if p.exists()), _CONFIG_DIR / "predictor.yaml")


def _load() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Predictor config not found. Searched:\n"
            + "\n".join(f"  {p}" for p in _CONFIG_SEARCH)
        )
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load()

# ── S3 paths ─────────────────────────────────────────────────────────────────
S3_BUCKET = "alpha-engine-research"

MODEL_WEIGHTS_KEY = "predictor/weights/latest.pt"
MODEL_WEIGHTS_DATED_KEY = "predictor/weights/{date}.pt"

# GBM_WEIGHTS_META_KEY remains live — ``inference/stages/write_output.py::
# _load_gbm_meta`` reads it best-effort to populate ``last_trained`` in the
# inference metrics payload. The v2 ``GBM_*_WEIGHTS_KEY`` siblings were
# deleted in the v2 cleanup PR (2026-04-27) along with the v2 inference
# path that consumed them.
GBM_WEIGHTS_META_KEY = "predictor/weights/gbm_latest.txt.meta.json"

# Calibrator weights (isotonic regression on v3 meta-model output).
# Path is contractual with alpha-engine-backtester retrain_alert grace
# gate — it reads isotonic_calibrator.meta.json for deployed_at. The v2
# GBM-era path (predictor/weights/calibrator_latest.pkl) was retired
# 2026-04-15 with the binary UP/DOWN migration.
CALIBRATOR_WEIGHTS_KEY      = "predictor/weights/meta/isotonic_calibrator.pkl"
CALIBRATOR_WEIGHTS_META_KEY = "predictor/weights/meta/isotonic_calibrator.pkl.meta.json"

PREDICTIONS_KEY = "predictor/predictions/{date}.json"
PREDICTIONS_LATEST_KEY = "predictor/predictions/latest.json"

METRICS_KEY = "predictor/metrics/latest.json"

PRICE_CACHE_KEY = "predictor/price_cache/{ticker}.parquet"

# ── Features ──────────────────────────────────────────────────────────────────
# Must stay in sync with data/feature_engineer.py::compute_features().
FEATURES = [
    "rsi_14",
    "macd_cross",
    "macd_above_zero",
    "macd_line_last",
    "price_vs_ma50",
    "price_vs_ma200",
    "momentum_20d",
    "avg_volume_20d",
    # v1.1 additions
    "dist_from_52w_high",
    "momentum_5d",
    "rel_volume_ratio",
    "return_vs_spy_5d",
    # v1.2 additions — market context features
    "vix_level",
    "dist_from_52w_low",
    "vol_ratio_10_60",
    "bollinger_pct",
    "sector_vs_spy_5d",
    "sector_vs_spy_10d",
    "sector_vs_spy_20d",
    # v1.3 additions — macro regime features
    "yield_10y",
    "yield_curve_slope",
    "gold_mom_5d",
    "oil_mom_5d",
    # v1.6 additions — investigation upgrades (A2)
    "vix_term_slope",
    "xsect_dispersion",
    # v1.4 additions — design doc Appendix A feature completions
    "price_accel",
    "ema_cross_8_21",
    "atr_14_pct",
    "realized_vol_20d",
    "volume_trend",
    "obv_slope_10d",
    "rsi_slope_5d",
    "volume_price_div",
    # v1.5 additions — regime interaction terms (cross-sectional: macro × ticker signal)
    "mom5d_x_vix",
    "rsi_x_vix",
    "sector_x_trend",
    "atr_x_vix",
    "vol_trend_x_vix",
    # v2.0 additions — alternative data signals (O10-O12)
    "earnings_surprise_pct",    # O10: PEAD — magnitude of most recent quarterly surprise
    "days_since_earnings",      # O10: PEAD — recency of last earnings (0-1, capped at 90d)
    "eps_revision_4w",          # O11: 4-week cumulative EPS revision percentage
    "revision_streak",          # O11: consecutive weeks of same-direction revisions
    "put_call_ratio",           # O12: log-transformed put/call OI ratio
    "iv_rank",                  # O12: IV percentile rank (0-1)
    "iv_vs_rv",                 # O12: implied vol / realized vol ratio
    # v3.0 additions — fundamental ratios (quarterly, from FMP)
    "pe_ratio",                  # P/E / 30, normalized
    "pb_ratio",                  # P/B / 5, normalized
    "debt_to_equity",            # D/E / 2, normalized
    "revenue_growth_yoy",        # YoY revenue growth (decimal)
    "fcf_yield",                 # Free cash flow / market cap
    "gross_margin",              # Gross profit / revenue (0-1)
    "roe",                       # Return on equity (decimal)
    "current_ratio",             # Current ratio / 3, normalized
]
N_FEATURES = 53

# Macro features — identical across all tickers on a given day, cannot predict
# cross-sectional alpha.  Excluded from GBM training/inference but kept in
# FEATURES for other callers (Research module, backtester technical scoring).
MACRO_FEATURES = {
    "vix_level", "yield_10y", "yield_curve_slope", "gold_mom_5d", "oil_mom_5d",
    "vix_term_slope",      # market-wide: identical across tickers → constant after rank norm
    "xsect_dispersion",    # market-wide: identical across tickers → constant after rank norm
}

# Fundamental features — computed and stored in feature store but excluded from
# GBM training/inference until backtester A/B validates their contribution.
# Move to GBM_FEATURES once validated with sufficient history.
_FUNDAMENTAL_EXCLUDE = {
    "pe_ratio", "pb_ratio", "debt_to_equity", "revenue_growth_yoy",
    "fcf_yield", "gross_margin", "roe", "current_ratio",
}

GBM_FEATURES = [f for f in FEATURES if f not in MACRO_FEATURES and f not in _FUNDAMENTAL_EXCLUDE]
N_GBM_FEATURES = len(GBM_FEATURES)  # 38 (31 technical + 7 alternative)

# ── Model architecture hyperparameters ───────────────────────────────────────
_model_cfg = _cfg["model"]
HIDDEN_1 = _model_cfg["hidden_1"]
HIDDEN_2 = _model_cfg["hidden_2"]
DROPOUT_1 = _model_cfg["dropout_1"]
DROPOUT_2 = _model_cfg["dropout_2"]

# ── Training hyperparameters ─────────────────────────────────────────────────
_train_cfg = _cfg["training"]
BATCH_SIZE = _train_cfg["batch_size"]
LEARNING_RATE = float(_train_cfg["learning_rate"])
WEIGHT_DECAY = float(_train_cfg["weight_decay"])
MAX_EPOCHS = _train_cfg["max_epochs"]
EARLY_STOPPING_PATIENCE = _train_cfg["early_stopping_patience"]
SCHEDULER_FACTOR = _train_cfg["scheduler_factor"]
SCHEDULER_PATIENCE = _train_cfg["scheduler_patience"]
MIN_LR = float(_train_cfg["min_lr"])
GRAD_CLIP_NORM = _train_cfg["grad_clip_norm"]
# ROADMAP L1581 — optional cutoff for the meta-trainer's data load.
# None (default) keeps all available history; an ISO date string
# ("YYYY-MM-DD") drops rows dated before that boundary. Diagnostic-only:
# set in alpha-engine-config for a single Sat retrain to compare meta IC
# against the no-clamp run, then revert. See ROADMAP entry for the
# 2020-03-31 vs ArcticDB-full-history hypothesis.
TRAIN_START_DATE: str | None = _train_cfg.get("train_start_date")

# ── Label thresholds ─────────────────────────────────────────────────────────
_label_cfg = _cfg["labeling"]
FORWARD_DAYS = _label_cfg["forward_days"]
UP_THRESHOLD = _label_cfg["up_threshold"]
DOWN_THRESHOLD = _label_cfg["down_threshold"]
LABEL_CLIP = _label_cfg["label_clip"]
ADAPTIVE_THRESHOLDS = _label_cfg.get("adaptive_thresholds", False)
ADAPTIVE_WINDOW = _label_cfg.get("adaptive_window", 63)
ADAPTIVE_UP_PCT = _label_cfg.get("adaptive_up_pct", 65)
ADAPTIVE_DOWN_PCT = _label_cfg.get("adaptive_down_pct", 35)

# ── Triple-barrier alpha labels (Stage 3, parallel-observe) ─────────────────
# LdP Ch. 3.4 — vol-scaled barriers. PR 1 of the Stage 3 arc ships the label
# generator + config knobs only; PR 2 wires parallel L1 training; PR 5 flips
# `enforce_cutover` to consume triple-barrier labels as the canonical L1
# alpha target. Plan: alpha-engine-docs/private/triple-barrier-260510.md.
_tb_cfg = _cfg.get("triple_barrier", {})
TRIPLE_BARRIER_FORWARD_WINDOW = _tb_cfg.get("forward_window", 21)
TRIPLE_BARRIER_VOL_WINDOW = _tb_cfg.get("vol_window", 20)
TRIPLE_BARRIER_VOL_MULTIPLIER = _tb_cfg.get("vol_multiplier", 2.0)
TRIPLE_BARRIER_MIN_PERIODS = _tb_cfg.get("min_periods", 10)
TRIPLE_BARRIER_ENFORCE_CUTOVER = _tb_cfg.get("enforce_cutover", False)

# ── GBM tuned hyperparameters ────────────────────────────────────────────────
_gbm_cfg = _cfg["gbm"]
GBM_N_ESTIMATORS = _gbm_cfg["n_estimators"]
GBM_EARLY_STOPPING_ROUNDS = _gbm_cfg["early_stopping_rounds"]
GBM_IC_IR_GATE = _gbm_cfg["ic_ir_gate"]
GBM_TUNED_PARAMS = _gbm_cfg["tuned_params"]

# Momentum base model — the shared GBM params above are tuned for the strong
# volatility signal. On the weak momentum target they overfit; the momentum
# base needs its own capacity config. Real values live in the private
# alpha-engine-config/predictor/predictor.yaml under gbm.momentum. Backtester
# can further override via s3://{bucket}/config/predictor_momentum_params.json.
# Missing section hard-fails — we never silently fall back to shared params.
_gbm_mom_cfg = _gbm_cfg.get("momentum")
if not _gbm_mom_cfg or "tuned_params" not in _gbm_mom_cfg:
    raise RuntimeError(
        "predictor.yaml is missing the required gbm.momentum section. "
        "Production values live in alpha-engine-config/predictor/predictor.yaml. "
        "See config/predictor.sample.yaml for the required schema."
    )
MOMENTUM_GBM_N_ESTIMATORS = _gbm_mom_cfg["n_estimators"]
MOMENTUM_GBM_EARLY_STOPPING_ROUNDS = _gbm_mom_cfg["early_stopping_rounds"]
MOMENTUM_GBM_TUNED_PARAMS = {**GBM_TUNED_PARAMS, **_gbm_mom_cfg["tuned_params"]}

# CatBoost ensemble config retired 2026-04-27 alongside the v2 inference
# path — its only consumer was ``_load_gbm`` (deleted) and the LGB-Cat
# blend mode that ``_load_meta_models`` does not implement.

# ── Production gates ─────────────────────────────────────────────────────────
_gates_cfg = _cfg["gates"]
MIN_HIT_RATE = _gates_cfg["min_hit_rate"]
MIN_IC = _gates_cfg["min_ic"]
MIN_CONFIDENCE = _gates_cfg["min_confidence"]
# Momentum veto threshold for the per-ticker ``momentum_veto`` flag on
# predictions.json. A ticker whose 20-day raw return is below this
# threshold gets ``momentum_veto=True``, signaling the executor to skip
# the ENTER (falling-knife guard). Moved from executor's inline
# momentum_gate (2026-05-11 refactor) to consolidate signal-side rules
# on the predictor side and free the executor to focus on risk rules.
# Default -0.05 (-5% over 20 days) matches the prior executor default.
# ``.get(..., -0.05)`` fallback lets this ship before predictor.yaml is
# updated in alpha-engine-config.
MOMENTUM_VETO_THRESHOLD = _gates_cfg.get("momentum_veto_threshold", -0.05)

# ── Training split (by time, no lookahead) ───────────────────────────────────
_split_cfg = _cfg["split"]
TRAIN_FRAC = _split_cfg["train_frac"]
VAL_FRAC = _split_cfg["val_frac"]

# ── Data fetching ────────────────────────────────────────────────────────────
_data_cfg = _cfg["data"]
REFRESH_BATCH_SIZE = _data_cfg["refresh_batch_size"]
BOOTSTRAP_PERIOD = _data_cfg["bootstrap_period"]
INFERENCE_PERIOD = _data_cfg["inference_period"]
DAILY_CLOSES_PERIOD = _data_cfg["daily_closes_period"]
STALENESS_THRESHOLD_DAYS = _data_cfg["staleness_threshold_days"]
SLIM_CACHE_LOOKBACK_DAYS = _data_cfg["slim_cache_lookback_days"]
INFERENCE_BATCH_SIZE = _data_cfg["inference_batch_size"]
MIN_ROWS_FOR_FEATURES = _data_cfg["min_rows_for_features"]
SPLIT_RETURN_THRESHOLD = _data_cfg["split_return_threshold"]

# ── Walk-forward validation ────────────────────────────────────────────────
_wf_cfg = _cfg.get("walk_forward", {})
WF_ENABLED = _wf_cfg.get("enabled", False)
# Renamed from test_window_days → test_window_trading_days; fallback to old key
WF_TEST_WINDOW_DAYS = _wf_cfg.get(
    "test_window_trading_days", _wf_cfg.get("test_window_days", 126)
)
WF_MIN_TRAIN_DAYS = _wf_cfg.get("min_train_days", 504)
WF_PURGE_DAYS = _wf_cfg.get("purge_days", 5)
WF_MIN_FOLDS_POSITIVE = _wf_cfg.get("min_folds_positive", 0.60)
WF_MEDIAN_IC_GATE = _wf_cfg.get("median_ic_gate", 0.02)

# Audit Phase 2a output-distribution gate (2026-05-07). When True, a
# calibrator-output-shape check is added to the promotion gate alongside
# the existing meta-IC and per-component subsample gates. Default False
# for one Saturday-SF observation cycle so we measure baseline pass rate
# before promoting the gate to blocking. Flip to True via predictor.yaml's
# walk_forward.output_distribution_gate_blocking once a baseline run
# confirms the gate doesn't false-block on healthy models.
OUTPUT_DISTRIBUTION_GATE_BLOCKING = _wf_cfg.get(
    "output_distribution_gate_blocking", False
)

# Audit Phase 2a-INFER inference-time variant of the output-distribution
# gate (2026-05-07). When True, write_predictions raises before any S3
# write if the live batch's p_up distribution fails the same four
# invariants that the promotion gate checks on synthetic input.
# Fail-closed semantics: failure → exception → SF Catch → flow-doctor
# alert → predictions.json NOT written → executor reads prior-day
# fallback. Default False for one observation cycle; flip via
# predictor.yaml's walk_forward.output_distribution_gate_inference_blocking
# after the metrics_out forensic trail confirms no false-blocks on
# healthy live batches.
OUTPUT_DISTRIBUTION_GATE_INFERENCE_BLOCKING = _wf_cfg.get(
    "output_distribution_gate_inference_blocking", False
)

WF_N_ESTIMATORS = _wf_cfg.get("wf_n_estimators", None)  # None → use GBM_N_ESTIMATORS
WF_EARLY_STOPPING = _wf_cfg.get("wf_early_stopping", None)  # None → use GBM_EARLY_STOPPING_ROUNDS

# Regime classifier promotion gate. 3-class balanced-random baseline is 0.333;
# 0.40 requires the model to clear random by a meaningful margin on walk-forward
# OOS. Macro-F1 guards against a degenerate majority-class classifier that scores
# high on accuracy while having zero recall on bear or bull. Thresholds are
# deliberately modest while the model is unvalidated — raise after we see real
# OOS numbers from the first walk-forward run.
REGIME_OOS_ACCURACY_GATE = _wf_cfg.get("regime_oos_accuracy_gate", 0.40)
REGIME_OOS_MACRO_F1_GATE = _wf_cfg.get("regime_oos_macro_f1_gate", 0.30)
# Per-class recall floor. Bear/neutral/bull must each recall at least this
# fraction of their true instances on walk-forward OOS. Catches the failure
# mode where average metrics look OK but the classifier ignores a class
# (e.g. bear recall = 0 under L2 regularization + class imbalance, smoke
# test 2026-04-16). Set conservatively — a model that can't identify 15%
# of bear periods is useless as a regime signal regardless of macro-F1.
REGIME_OOS_PER_CLASS_RECALL_FLOOR = _wf_cfg.get("regime_oos_per_class_recall_floor", 0.15)

# ── Feature selection / noise detection ─────────────────────────────────────
_fs_cfg = _cfg.get("feature_selection", {})
SHAP_NOISE_THRESHOLD_PCT = _fs_cfg.get("shap_noise_threshold_pct", 1.0)
IC_NOISE_THRESHOLD = _fs_cfg.get("ic_noise_threshold", 0.005)
AUTO_PRUNE_NOISE_FEATURES = _fs_cfg.get("auto_prune", False)

# ── Calibration ────────────────────────────────────────────────────────────
_cal_cfg = _cfg.get("calibration", {})
CALIBRATION_METHOD = _cal_cfg.get("method", "platt")  # "platt" or "isotonic"
CALIBRATION_ENABLED = _cal_cfg.get("enabled", True)

# Multi-horizon prediction config retired 2026-04-27 alongside the v2
# inference path — the v3 meta-model trains one horizon (cfg.FORWARD_DAYS).

# ── Meta-model architecture (v3.0) ─────────────────────────────────────────
_meta_cfg = _cfg.get("meta_model", {})
META_MODEL_ENABLED = _meta_cfg.get("enabled", False)

# Per-model feature sets (subsets of FEATURES computed by alpha-engine-data).
#
# v3.1 iteration (2026-04-15, predictor ROADMAP P2):
#
#   First pass added 6 features (return_60d, return_120d,
#   overnight_return_5d, intraday_return_5d, dist_from_5d_high,
#   dist_from_20d_high). Post-training Layer-1 feature importance
#   analysis (2026-04-15 full training run) showed:
#     return_60d          23.3% gain  ← #1 feature in momentum model
#     return_120d         14.0% gain  ← #4 feature
#     intraday_return_5d   8.7% gain  ← modest but non-zero
#     overnight_return_5d  0.0% gain  ← DEAD, removed
#     dist_from_20d_high   0.7% gain  ← near-dead, removed
#     dist_from_5d_high    0.3% gain  ← DEAD, removed
#
#   Dropped the 3 dead features from the subscription lists. They
#   remain in alpha-engine-data's feature store and ArcticDB rows
#   (compute cost is trivial, storage is trivial, and they may be
#   useful in a different model architecture or different horizon
#   later). This PR only updates which columns the Layer-1 GBMs
#   subscribe to.
MOMENTUM_FEATURES = [
    "momentum_5d", "momentum_20d", "price_vs_ma50", "price_vs_ma200",
    "rsi_14", "macd_cross",
    # v3.1: longer-horizon returns — pulled real weight in Layer-1
    "return_60d", "return_120d",
    # v3.1: intraday component of 5d return — marginal but non-zero
    "intraday_return_5d",
]
VOLATILITY_FEATURES = [
    "atr_14_pct", "realized_vol_20d", "vol_ratio_10_60",
    "iv_rank", "dist_from_52w_high", "dist_from_52w_low",
]

# Stage 2b of regime-conditioning rebuild (plan: regime-conditioning-260510.md)
# — per-ticker risk features written to ArcticDB by alpha-engine-data Stage
# 2a (#202) and Stage 2a-extended (realized_vol_63d). Each varies cross-
# sectionally on a given date, so the existing rank-norm pipeline applies
# directly. Feeds the parallel ``prod_vol_risk_aug`` GBM (Stage 2b parallel
# observation) alongside the plain volatility GBM. Captures distinct risk
# dimensions the existing 6 vol features cannot split on:
#   - beta_60d:        systematic market exposure
#   - idio_vol_60d:    residual risk (after removing market-beta exposure)
#   - vol_of_vol_30d:  vol-regime stability
#   - max_drawdown_60d: recent left-tail risk
#   - realized_vol_63d: 3-month vol regime (pairs with 20d for term structure)
RISK_AUG_FEATURES = [
    "beta_60d",
    "idio_vol_60d",
    "vol_of_vol_30d",
    "max_drawdown_60d",
    "realized_vol_63d",
]

# Stage 1 of regime-conditioning rebuild (plan: regime-conditioning-260510.md)
# — macros consumed by L1 GBMs as time-series-z-scored features. Cross-
# sectional rank-norm degenerates these to 0.5 for every ticker because
# they're constant across tickers on a given date; time-series z-score
# over a rolling window preserves the regime signal (high VIX vs low VIX).
# Trees discover interactions with rank-normed per-ticker features
# through sequential splits — institutional SOTA per-feature normalization.
#
# Source: ``regime_features_df`` columns (built by
# ``model.regime_predictor.RegimePredictor.build_features()`` — kept as
# feature-engineering utility after the Tier-0 classifier was retired).
MACRO_NORM_FEATURES = [
    "spy_20d_return",
    "spy_20d_vol",
    "vix_level",
    "vix_term_slope",
    "yield_curve_slope",
    "market_breadth",
    # Stage 2c-partial additions (2026-05-10):
    # - vix_vix3m_ratio: institutional-canonical VIX term-structure
    #   normalization (ratio > 1 = backwardation/stress; < 1 = contango).
    #   Distinct from vix_term_slope which is bps-style normalized diff.
    # - market_breadth_200d: secular bull-vs-bear regime indicator.
    #   Distinct from market_breadth (50d cyclical breadth).
    # Stage 2c-full (gated on Stage 2.5 + 2.5b data ingestion + backfill):
    # yield_curve_10y_2y, hy_oas_level, hy_oas_change_21d.
    "vix_vix3m_ratio",
    "market_breadth_200d",
    # Stage 2c-full additions (2026-05-10):
    # - yield_curve_10y_2y: recession-canonical curve, distinct from
    #   yield_curve_slope (10Y-3M cyclical). Inverted values precede
    #   recessions historically.
    # - hy_oas_level / hy_oas_change_21d: HY-specific credit stress
    #   regime. License-gated to 2023+ on FRED (forward signal).
    # - baa10y_level / baa10y_change_21d: BBB-rated corporate spread,
    #   full 40y FRED history. Provides credit-regime signal across
    #   the full predictor training corpus.
    "yield_curve_10y_2y",
    "hy_oas_level",
    "hy_oas_change_21d",
    "baa10y_level",
    "baa10y_change_21d",
]
# Rolling window for time-series z-score. 252 trading days = 1 year, the
# institutional default for macro normalization (captures one full regime
# cycle without being overly slow to adapt). ``min_periods`` is set in
# ``data.dataset.time_series_zscore_normalize`` (default 60 = 3 months).
MACRO_NORM_WINDOW = 252

# Regime predictor uses macro series directly (not GBM features)
# Research calibrator uses signals.json fields (not price features)

# S3 keys for meta-model weights
META_WEIGHTS_PREFIX = "predictor/weights/meta/"
META_MANIFEST_KEY = "predictor/weights/meta/manifest.json"
META_FEATURE_LIST_KEY = "predictor/weights/meta/feature_list.json"

# ── Feature engineering parameters ───────────────────────────────────────────
FEATURE_CFG: dict = _cfg["features"]

# ── Feature Store (read-only — writes handled by alpha-engine-data) ──────────
FEATURE_STORE_PREFIX = "features/"

# Fundamental features (v3.0 — quarterly FMP data)
FUNDAMENTAL_FEATURES = [
    "pe_ratio",
    "pb_ratio",
    "debt_to_equity",
    "revenue_growth_yoy",
    "fcf_yield",
    "gross_margin",
    "roe",
    "current_ratio",
]

# ── AWS / Email ──────────────────────────────────────────────────────────────
AWS_REGION       = os.environ.get("AWS_REGION", "us-east-1")
EMAIL_SENDER     = os.environ.get("EMAIL_SENDER", "")
EMAIL_RECIPIENTS = [
    r.strip()
    for r in os.environ.get("EMAIL_RECIPIENTS", "").split(",")
    if r.strip()
]
