# alpha-engine-predictor

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-135_passing-brightgreen.svg)]()

> Meta-model (4 specialized GBMs + ridge) predicts 5-day market-relative returns. Produces directional predictions (UP/FLAT/DOWN) with confidence scores, and provides a veto gate that blocks entry into declining positions.

**Part of the [Nous Ergon](https://nousergon.ai) autonomous trading system.**
See the [system overview](https://github.com/cipher813/alpha-engine#readme) for how all modules connect, or the [full documentation index](https://github.com/cipher813/alpha-engine-docs#readme).

## Table of Contents

- [Role in the System](#role-in-the-system)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [How Training Works](#how-training-works)
- [How Inference Works](#how-inference-works)
- [Direction Labels and the Veto Gate](#direction-labels-and-the-veto-gate)
- [Configuration Reference](#configuration-reference)
- [Key Files](#key-files)
- [Deployment](#deployment)
- [S3 Contract](#s3-contract)
- [Testing](#testing)
- [Related Modules](#related-modules)

---

## Role in the System

The Predictor adds an ML layer on top of Research signals. It reads `signals.json` from S3 (written by Research), runs inference on each ticker, and writes `predictions.json` — which the Executor reads before placing trades. High-confidence DOWN predictions trigger a veto gate that overrides BUY signals.

```
Research → signals.json → Predictor → predictions.json → Executor
```

---

## Quick Start

### Prerequisites

- Python 3.12 (Lambda runtime); tests also pass on 3.9+
- AWS credentials with S3 read/write access
- For training: multi-year OHLCV data (fetched automatically by bootstrap script)

### Setup

```bash
git clone https://github.com/cipher813/alpha-engine-predictor.git
cd alpha-engine-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config/predictor.sample.yaml config/predictor.yaml
# Edit config/predictor.yaml — set S3 paths, hyperparameters, confidence gates
```

### Bootstrap Training Data (One-Time)

Downloads multi-year daily OHLCV for S&P 500 + S&P 400 constituents:

```bash
python data/bootstrap_fetcher.py --output-dir data/cache

# Test with a small subset first:
python data/bootstrap_fetcher.py --output-dir data/cache --limit 20
```

### Train

```bash
python train_gbm.py --data-dir data/cache
# Output: checkpoints/gbm_best.txt, checkpoints/gbm_eval_report.json
```

### Run Inference (Dry Run)

```bash
python inference/daily_predict.py --local --model-type gbm --dry-run
```

### Run Tests

```bash
pytest tests/ -v
```

All tests run without network access or AWS credentials.

---

## Architecture

```
┌────────────────────────────────────────────────────┐
│     Weekly Retraining (Saturday, Step Function)     │
│  ArcticDB prices → feature_engineer → dataset      │
│  → 4 specialized GBMs (momentum, mean-rev,         │
│    volatility, regime) + ridge meta-learner         │
│  → walk-forward validation (5 folds, IC gate)      │
│  → promote weights to S3 if IC > 0.05              │
│  → drift detection + retrain alerts                │
└────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────┐
│      Daily Inference — Lambda (6:07 AM PT)          │
│  → load meta-model weights from S3                  │
│  → read watchlist from signals.json                 │
│  → load prices (ArcticDB slim + daily_closes delta) │
│  → compute features (from feature store or inline)  │
│  → meta-model predict → direction + confidence      │
│  → write predictions/{date}.json + latest.json      │
│  → daily health check Lambda (drift, clustering)    │
└────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────┐
│      Downstream Integration                         │
│  Executor reads predictions (best-effort)           │
│  Veto gate: DOWN ≥ confidence threshold → HOLD      │
│  Backtester evaluates: confusion matrix, P/R/F1    │
└────────────────────────────────────────────────────┘
```

**v3.0 Meta-Model** (deployed 2026-04-01): 4 specialized GBMs (momentum, mean-reversion, volatility, regime) combined via ridge meta-learner. Walk-forward validated across 5 folds. IC ~4x single GBM baseline.

**Feedback Loop** (added 2026-04-07): Drift detection (feature z-scores, prediction clustering), ensemble mode evaluation, feature pruning, and retrain alerts (5 trigger conditions → email notification).

**Daily Health Check Lambda**: Verifies prediction freshness, detects feature drift, checks for degenerate clustering. SNS alert on anomaly.

---

## How Training Works

Training runs **weekly on Monday** via Lambda. The GBM learns to predict 5-day sector-relative returns from historical price data.

1. **Data**: Multi-year OHLCV parquets (one per ticker, plus SPY, VIX, sector ETFs, treasuries, gold, oil)
2. **Features**: 36 rolling technical, macro-interaction, and alternative data indicators computed per row (252-row warmup for 52-week rolling windows)
3. **Labels (sector-neutral)**: `alpha = stock_5d_return - sector_ETF_5d_return`, isolating stock-specific alpha from sector/market moves
4. **Objective**: LightGBM regression (MSE) on continuous alpha, evaluated on Pearson IC. An ensemble mode also trains a lambdarank model and promotes whichever achieves higher IC.
5. **Walk-forward validation**: Expanding window with 5-day purge gap to prevent label leakage; model promoted only if median IC > 0.01 across 60%+ of folds
6. **Production gate**: Minimum IC required before uploading weights to S3

The trained model weights are frozen until the next Monday retrain. Inference between retrains uses the same weights but with fresh daily price data (see below).

An MLP (PyTorch) model is also available but not active. Activate by setting `model_type="mlp"` in `handler.py`.

---

## How Inference Works

Inference runs **daily on weekdays** at 6:15 AM PT. Although the GBM weights are frozen (weekly retrain), predictions change every day because the input features are recomputed from fresh price data.

1. EventBridge fires daily at 6:15 AM PT on weekdays
2. GBM weights loaded from S3 (same weights all week until Monday retrain)
3. Watchlist read from latest population or `signals.json` — only Research-tracked tickers are predicted
4. **Fresh OHLCV fetched via yfinance** (2-year window per ticker) — this is the new data each day
5. **Features recomputed from scratch** — all 36 indicators (RSI, MACD, momentum, Bollinger bands, relative volume, etc.) shift daily as new price bars are incorporated
6. GBM predicts continuous alpha → thresholded to UP/FLAT/DOWN + confidence
7. Results written to `predictor/predictions/{date}.json` and `latest.json`

Tickers with insufficient history for long-window features are skipped. Per-ticker failures do not abort the run.

### Why predictions change daily without retraining

The GBM model is a function: `features_in → alpha_out`. Training freezes the function (the tree structure and leaf values). But the features themselves are rolling calculations over price data, so they produce different values every day. For example:

| Feature | What changes daily |
|---------|-------------------|
| `momentum_5d` | Last 5 trading days shift by one day |
| `rsi_14` | New close changes the 14-day RSI value |
| `return_vs_spy_5d` | Stock's relative performance vs SPY over the latest 5 days |
| `bollinger_pct` | Price position within the 20-day Bollinger band |
| `volume_trend` | 5-day vs 20-day average volume ratio |
| `obv_slope_10d` | On-Balance Volume trend shifts with each new bar |

A stock predicted UP on Monday could flip to DOWN on Tuesday if it drops sharply — RSI falls, momentum turns negative, it breaks below the Bollinger band. The model sees an entirely new feature vector each day.

Each day's prediction asks: *"will this stock beat its sector ETF over the next 5 trading days starting today?"* — a rolling window that advances by one day per run. These are genuinely daily predictions, not weekly predictions reused.

---

## Direction Labels and the Veto Gate

The GBM outputs a **continuous alpha score** (predicted 5-day sector-relative return). This raw score is the primary output. The UP/FLAT/DOWN labels are derived from it using configurable thresholds:

- **UP**: alpha score > `up_threshold` (default: +0.001)
- **DOWN**: alpha score < `down_threshold` (default: -0.001)
- **FLAT**: alpha score between the two thresholds

### How the Executor uses predictions

The Executor treats UP and FLAT identically — both are eligible for entry. Only DOWN triggers the veto gate:

| Direction | Executor behavior |
|-----------|------------------|
| **UP** | Eligible for ENTER (no restriction from predictor) |
| **FLAT** | Eligible for ENTER (no restriction from predictor) |
| **DOWN** + confidence >= veto threshold | **Vetoed** — ENTER demoted to HOLD |
| **DOWN** + confidence < veto threshold | Eligible for ENTER (low-confidence DOWN ignored) |

The veto threshold is regime-adaptive (lowered in bear/caution markets, raised in bull markets) and auto-tuned by the backtester.

### Alpha score calibration

The system trains two GBM models weekly and promotes whichever achieves the highest IC (information coefficient):

- **MSE model**: trained with `objective: regression` on actual sector-relative 5-day returns. Its output is a calibrated return prediction (e.g., `+0.015` = "I predict this stock will beat its sector ETF by 1.5% over the next 5 days").
- **Lambdarank model**: trained with `objective: lambdarank` optimizing NDCG. Produces better cross-sectional rankings but its output scale is arbitrary (not calibrated to returns).

At inference, the MSE model's output is always used as `predicted_alpha` — the reported alpha score in predictions and emails. If the lambdarank or ensemble model wins the IC competition, it's used for ranking/direction decisions, but the alpha magnitude still comes from the MSE model. This ensures `predicted_alpha` always represents a genuine return estimate. Scores are clipped to `±LABEL_CLIP` (±25%) as a bound on extrapolation.

### Direction thresholds

The UP/FLAT/DOWN thresholds define what constitutes a meaningful predicted edge over a 5-day horizon:

- **UP**: predicted sector-relative alpha > +2% (stock expected to meaningfully outperform)
- **FLAT**: predicted alpha within ±2% (no clear directional edge)
- **DOWN**: predicted sector-relative alpha < -2% (stock expected to meaningfully underperform)

The ±2% band reflects a practical noise floor — within that range, the model's prediction is not distinguishable from zero with meaningful confidence. The Executor currently treats UP and FLAT identically (both eligible for entry), with only high-confidence DOWN predictions triggering the veto gate.

### Rolling 5-day horizon

Each daily prediction asks: *"will this stock beat its sector ETF over the next 5 trading days starting today?"* The 5-day window is the forecast horizon; the daily cadence is the refresh rate. Monday's prediction covers Mon→Fri. Tuesday's prediction covers Tue→next Mon. Each overlaps the previous by 4 days but incorporates the latest price action into its feature vector. This is analogous to a 5-day weather forecast issued every morning — the horizon stays fixed but the starting point advances daily with updated information.

---

## Configuration Reference

All hyperparameters are in `config/predictor.yaml` (gitignored — copy from `config/predictor.sample.yaml`). The backtester auto-tunes the veto confidence threshold via `config/predictor_params.json` in S3.

Key configuration dimensions:
- **S3 paths**: Bucket, weights key, predictions prefix
- **Prediction horizon**: Forward days for return calculation
- **Label thresholds**: Sector-neutral return cutoffs for UP/DOWN classification
- **Confidence gates**: Minimum confidence for score modifier and veto gate
- **Production gates**: Minimum IC and hit rate required for weight promotion
- **GBM hyperparameters**: Tree complexity, ensemble size, regularization

---

## Key Files

```
alpha-engine-predictor/
├── data/
│   ├── bootstrap_fetcher.py     # One-time OHLCV fetch + sector_map.json
│   ├── feature_engineer.py      # Rolling feature computation
│   ├── label_generator.py       # Sector-neutral forward return labels
│   └── dataset.py               # Dataset builder for GBM and MLP
├── model/
│   ├── gbm_scorer.py            # LightGBM wrapper (production)
│   ├── predictor.py             # MLP model + checkpoint save/load
│   ├── trainer.py               # MLP training loop
│   └── evaluator.py             # IC, hit rate, confusion matrix
├── inference/
│   ├── daily_predict.py         # Daily prediction pipeline
│   └── handler.py               # AWS Lambda entry point
├── config/
│   ├── predictor.sample.yaml    # Template — copy to predictor.yaml
│   └── predictor.yaml           # GITIGNORED — your tuned config
├── train_gbm.py                 # GBM training entry point (production)
├── train.py                     # MLP training entry point
├── config.py                    # S3 paths, feature list, defaults
└── tests/                       # Unit tests (no network/AWS required)
```

---

## Deployment

### Lambda (Zip — No Docker Required)

```bash
# See docs/deployment.md for the full build procedure
pip install --platform manylinux_2_28_x86_64 --target /tmp/lambda-pkg \
  --implementation cp --python-version 3.12 --only-binary=:all: \
  -r requirements-lambda.txt
cd /tmp/lambda-pkg && zip -r /tmp/predictor.zip . -q
aws s3 cp /tmp/predictor.zip s3://alpha-engine-research/lambda/alpha-engine-predictor.zip
aws lambda update-function-code --function-name alpha-engine-predictor-inference \
  --s3-bucket alpha-engine-research --s3-key lambda/alpha-engine-predictor.zip
```

### Weight Update (No Redeploy Needed)

```bash
python train_gbm.py --data-dir data/cache
aws s3 cp checkpoints/gbm_best.txt \
  s3://alpha-engine-research/predictor/weights/gbm_latest.txt
```

---

## Testing

```bash
pytest tests/ -v
```

Tests cover feature engineering, model forward pass, checkpoint loading, and prediction pipeline. All tests run without network access or AWS credentials.

### Local Inference Runs

```bash
# Offline mode: synthetic prices, dummy GBM scorers, no S3/API calls
# Tests full feature computation, ranking, combined rank, veto logic
python inference/daily_predict.py --offline

# Dry run: real S3 data, real models, but skip S3 writes
python inference/daily_predict.py --model-type gbm --watchlist auto --dry-run

# Full local run: reads from and writes to S3, sends email
python inference/daily_predict.py --model-type gbm --watchlist auto
```

**Preprod workflow:**
1. `--offline` — verify code changes don't crash (no API calls needed)
2. `--dry-run` — verify with real models and prices
3. Deploy to Lambda: `./infrastructure/deploy.sh main`

---

## S3 Contract

### Reads
| Path | Source | Content |
|------|--------|---------|
| `signals/{date}/signals.json` | Research | Watchlist (population tickers) |
| ArcticDB `universe_slim` | Data | 2y OHLCV for feature computation |
| `predictor/daily_closes/{date}.parquet` | Data | Daily delta for slim cache |
| `predictor/feature_store/{date}/` | Data | Pre-computed 54 features (primary) |
| `config/predictor_params.json` | Backtester | Auto-tuned veto threshold |

### Writes
| Path | Content |
|------|---------|
| `predictor/predictions/{date}.json` | Per-ticker direction, confidence, predicted alpha |
| `predictor/predictions/latest.json` | Pointer to most recent predictions |
| `predictor/weights/` | GBM + meta-model weights (weekly) |
| `predictor/metrics/latest.json` | IC, hit rate, model performance |
| `predictor/metrics/drift_{date}.json` | Feature drift alerts |
| `health/predictor_inference.json` | Module health marker |

---

## Related Modules

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor + system overview
- [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) — Autonomous LLM research pipeline
- [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) — Evaluation framework and parameter optimization
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard
- [`alpha-engine-data`](https://github.com/cipher813/alpha-engine-data) — Centralized data collection and ArcticDB
- [`alpha-engine-docs`](https://github.com/cipher813/alpha-engine-docs) — Documentation index

---

## License

MIT — see [LICENSE](LICENSE).
