"""
config.py — Central configuration for alpha-engine-predictor.

All S3 paths, model hyperparameters, feature definitions, and production
gates live here. Import this module everywhere rather than hard-coding values.
"""

# ── S3 paths ─────────────────────────────────────────────────────────────────
S3_BUCKET = "alpha-engine-research"

MODEL_WEIGHTS_KEY = "predictor/weights/latest.pt"
MODEL_WEIGHTS_DATED_KEY = "predictor/weights/{date}.pt"

PREDICTIONS_KEY = "predictor/predictions/{date}.json"
PREDICTIONS_LATEST_KEY = "predictor/predictions/latest.json"

METRICS_KEY = "predictor/metrics/latest.json"

PRICE_CACHE_KEY = "predictor/price_cache/{ticker}.parquet"

# ── Features ──────────────────────────────────────────────────────────────────
# Must stay in sync with data/feature_engineer.py::compute_features().
# First 8 mirror compute_technical_indicators() in alpha-engine-research.
# v1.1 added 4 alpha features; v1.2 adds 5 market-context features.
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
    "dist_from_52w_high",   # (close - 252d rolling max) / 252d rolling max
    "momentum_5d",          # (close / close.shift(5)) - 1
    "rel_volume_ratio",     # today's volume / 20d rolling mean volume
    "return_vs_spy_5d",     # 5d momentum of stock minus 5d momentum of SPY
    # v1.2 additions — market context features
    "vix_level",            # VIX Close / 20.0 (fear regime indicator)
    "dist_from_52w_low",    # (close - 252d rolling min) / 252d rolling min
    "vol_ratio_10_60",      # 10d realized vol / 60d realized vol
    "bollinger_pct",        # (close - lower_bb20) / (upper_bb20 - lower_bb20)
    "sector_vs_spy_5d",     # sector ETF 5d return - SPY 5d return
]
N_FEATURES = 17
N_CLASSES = 3  # UP, FLAT, DOWN

# Class labels — index matches model output neuron order
CLASS_LABELS = ["DOWN", "FLAT", "UP"]  # index 0, 1, 2

# ── Model architecture hyperparameters ───────────────────────────────────────
HIDDEN_1 = 64
HIDDEN_2 = 32
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2

# ── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# ── Label thresholds ─────────────────────────────────────────────────────────
# 5-day forward return bins that define UP / DOWN / FLAT
UP_THRESHOLD = 0.01     # > +1% forward return → UP
DOWN_THRESHOLD = -0.01  # < -1% forward return → DOWN
# everything in between → FLAT

# ── Production gates ─────────────────────────────────────────────────────────
# Model must meet these thresholds before being applied in production.
MIN_HIT_RATE = 0.55       # >55% directional accuracy required
MIN_IC = 0.05             # >0.05 Pearson IC (predicted vs actual return)
MIN_CONFIDENCE = 0.65     # predictions below this gate are not applied as score modifiers

# ── Training split (by time, no lookahead) ───────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (remainder — always the most recent data)

# ── Prediction horizon ───────────────────────────────────────────────────────
FORWARD_DAYS = 5  # predict 5-trading-day forward return
