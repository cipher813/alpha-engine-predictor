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
_CONFIG_PATH = _CONFIG_DIR / "predictor.yaml"


def _load() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Predictor config not found: {_CONFIG_PATH}\n"
            "Copy config/predictor.sample.yaml to config/predictor.yaml and customise."
        )
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load()

# ── S3 paths ─────────────────────────────────────────────────────────────────
S3_BUCKET = "alpha-engine-research"

MODEL_WEIGHTS_KEY = "predictor/weights/latest.pt"
MODEL_WEIGHTS_DATED_KEY = "predictor/weights/{date}.pt"

GBM_WEIGHTS_KEY      = "predictor/weights/gbm_latest.txt"
GBM_WEIGHTS_META_KEY = "predictor/weights/gbm_latest.txt.meta.json"

# Ensemble model weights (MSE + lambdarank)
GBM_MSE_WEIGHTS_KEY       = "predictor/weights/gbm_mse_latest.txt"
GBM_MSE_WEIGHTS_META_KEY  = "predictor/weights/gbm_mse_latest.txt.meta.json"
GBM_RANK_WEIGHTS_KEY      = "predictor/weights/gbm_rank_latest.txt"
GBM_RANK_WEIGHTS_META_KEY = "predictor/weights/gbm_rank_latest.txt.meta.json"
GBM_MODE_KEY              = "predictor/weights/gbm_mode.json"

# Calibrator weights (Platt scaling / isotonic regression)
CALIBRATOR_WEIGHTS_KEY      = "predictor/weights/calibrator_latest.pkl"
CALIBRATOR_WEIGHTS_META_KEY = "predictor/weights/calibrator_latest.pkl.meta.json"

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
N_CLASSES = 3  # UP, FLAT, DOWN

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

# Class labels — index matches model output neuron order
CLASS_LABELS = ["DOWN", "FLAT", "UP"]  # index 0, 1, 2

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

# ── GBM tuned hyperparameters ────────────────────────────────────────────────
_gbm_cfg = _cfg["gbm"]
GBM_N_ESTIMATORS = _gbm_cfg["n_estimators"]
GBM_EARLY_STOPPING_ROUNDS = _gbm_cfg["early_stopping_rounds"]
GBM_IC_IR_GATE = _gbm_cfg["ic_ir_gate"]
GBM_TUNED_PARAMS = _gbm_cfg["tuned_params"]
GBM_ENSEMBLE_LAMBDARANK = _gbm_cfg.get("ensemble_lambdarank", True)

# ── CatBoost ensemble ──────────────────────────────────────────────────────
_cat_cfg = _cfg.get("catboost", {})
CATBOOST_ENABLED = _cat_cfg.get("enabled", False)
CATBOOST_PARAMS = _cat_cfg.get("params", None)
CATBOOST_WEIGHTS_KEY = "predictor/weights/catboost_latest.cbm"
CATBOOST_WEIGHTS_META_KEY = "predictor/weights/catboost_latest.cbm.meta.json"

# ── Production gates ─────────────────────────────────────────────────────────
_gates_cfg = _cfg["gates"]
MIN_HIT_RATE = _gates_cfg["min_hit_rate"]
MIN_IC = _gates_cfg["min_ic"]
MIN_CONFIDENCE = _gates_cfg["min_confidence"]

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
WF_N_ESTIMATORS = _wf_cfg.get("wf_n_estimators", None)  # None → use GBM_N_ESTIMATORS
WF_EARLY_STOPPING = _wf_cfg.get("wf_early_stopping", None)  # None → use GBM_EARLY_STOPPING_ROUNDS

# ── Feature selection / noise detection ─────────────────────────────────────
_fs_cfg = _cfg.get("feature_selection", {})
SHAP_NOISE_THRESHOLD_PCT = _fs_cfg.get("shap_noise_threshold_pct", 1.0)
IC_NOISE_THRESHOLD = _fs_cfg.get("ic_noise_threshold", 0.005)
AUTO_PRUNE_NOISE_FEATURES = _fs_cfg.get("auto_prune", False)

# ── Calibration ────────────────────────────────────────────────────────────
_cal_cfg = _cfg.get("calibration", {})
CALIBRATION_METHOD = _cal_cfg.get("method", "platt")  # "platt" or "isotonic"
CALIBRATION_ENABLED = _cal_cfg.get("enabled", True)

# ── Multi-horizon prediction ───────────────────────────────────────────────
_mh_cfg = _cfg.get("multi_horizon", {})
MULTI_HORIZON_ENABLED = _mh_cfg.get("enabled", False)
MULTI_HORIZON_LIST = _mh_cfg.get("horizons", [1, 5, 10, 20])

# ── Meta-model architecture (v3.0) ─────────────────────────────────────────
_meta_cfg = _cfg.get("meta_model", {})
META_MODEL_ENABLED = _meta_cfg.get("enabled", False)

# Per-model feature sets (subsets of FEATURES computed by feature_engineer)
MOMENTUM_FEATURES = [
    "momentum_5d", "momentum_20d", "price_vs_ma50", "price_vs_ma200",
    "rsi_14", "macd_cross",
]
VOLATILITY_FEATURES = [
    "atr_14_pct", "realized_vol_20d", "vol_ratio_10_60",
    "iv_rank", "dist_from_52w_high", "dist_from_52w_low",
]
# Regime predictor uses macro series directly (not GBM features)
# Research calibrator uses signals.json fields (not price features)

# S3 keys for meta-model weights
META_WEIGHTS_PREFIX = "predictor/weights/meta/"
META_MANIFEST_KEY = "predictor/weights/meta/manifest.json"

# ── Feature engineering parameters ───────────────────────────────────────────
FEATURE_CFG: dict = _cfg["features"]

# ── Feature Store ────────────────────────────────────────────────────────────
_fs_store_cfg = _cfg.get("feature_store", {})
FEATURE_STORE_ENABLED = _fs_store_cfg.get("enabled", False)
FEATURE_STORE_PREFIX = _fs_store_cfg.get("prefix", "features/")
FEATURE_STORE_WRITE_ON_INFERENCE = False  # alpha-engine-data/features/compute.py handles writes
FEATURE_STORE_WRITE_ON_TRAINING = False   # alpha-engine-data/features/compute.py handles writes
FEATURE_STORE_FULL_UNIVERSE = _fs_store_cfg.get("full_universe", True)
FEATURE_STORE_READ_ON_INFERENCE = _fs_store_cfg.get("read_on_inference", True)

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
