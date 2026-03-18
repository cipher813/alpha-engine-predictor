# Alpha Engine Predictor

LightGBM model that predicts 5-day market-relative returns for each ticker. Produces directional predictions (UP/FLAT/DOWN) with confidence scores, and provides a veto gate that blocks entry into declining positions.

> Part of [Nous Ergon: Alpha Engine](https://github.com/cipher813/alpha-engine).

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
│          Weekly Retraining (Monday)                 │
│  bootstrap_fetcher → feature_engineer → dataset    │
│  → train_gbm.py → checkpoints/gbm_best.txt → S3   │
└────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────┐
│      Daily Inference — Lambda (6:15 AM PT)          │
│  → load GBM weights from S3                         │
│  → read watchlist from signals.json                 │
│  → fetch OHLCV per ticker (yfinance)                │
│  → compute features (latest row only)               │
│  → GBM predict → direction + confidence             │
│  → write predictions/{date}.json + latest.json      │
└────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────┐
│      Downstream Integration                         │
│  Executor reads predictions (best-effort)           │
│  Veto gate: DOWN ≥ confidence threshold → HOLD      │
└────────────────────────────────────────────────────┘
```

---

## How Training Works

1. **Data**: Multi-year OHLCV parquets (one per ticker, plus SPY, VIX, sector ETFs)
2. **Features**: Rolling technical + macro indicators per row (warmup period required for long-window features)
3. **Labels (sector-neutral)**: Forward return relative to the ticker's sector ETF, isolating stock-specific alpha as the training target
4. **Objective**: LightGBM regression (MSE), evaluated on Pearson IC
5. **Split**: Time-based (configurable ratios); test set = most recent dates
6. **Early stopping**: Validation IC monitored; feature importance logged
7. **Production gate**: Minimum IC required before uploading weights to S3

An MLP (PyTorch) model is also available but not running in production. Activate by setting `model_type="mlp"` in `handler.py`.

---

## How Inference Works

1. EventBridge fires daily at 6:15 AM PT on weekdays
2. GBM weights loaded from S3
3. Watchlist read from latest `signals.json` — only Research-tracked tickers are predicted
4. OHLCV fetched via yfinance (1-year window per ticker)
5. Features computed; only the last row (today's snapshot) is used
6. GBM predicts continuous alpha → thresholded to UP/FLAT/DOWN + confidence
7. Results written to `predictor/predictions/{date}.json` and `latest.json`

Tickers with insufficient history for long-window features are skipped. Per-ticker failures do not abort the run.

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

---

## Opportunities for Improvement

### Model Architecture

1. **3-model stacking ensemble** — add XGBoost and CatBoost alongside LightGBM, combined with a ridge regression meta-learner optimized on IC. MSE + lambdarank ensemble already captures objective-level diversity, so marginal gain from additional tree libraries is uncertain. Training 4 models + SHAP may approach Lambda's 15-minute timeout. **Deferred** — revisit if the temporal ensemble (item 2) doesn't close the accuracy gap, or if training moves to EC2 where compute budget is less constrained.

2. **Temporal ensemble** — currently only predicts 5-day returns. Training on 3, 5, 7, and 10-day horizons and blending predictions captures multi-horizon alpha and reduces label noise sensitivity. Implementation: separate GBM per horizon + learned blend weights. **Highest-value deferred item** — different horizons capture genuinely different market dynamics. Likely adds more value than model-library diversity (item 1).

3. ~~**Ranking loss (lambdarank)**~~ — **Implemented 2026-03-17.** Both MSE and lambdarank LightGBM models train weekly. The system evaluates MSE IC, lambdarank IC, and ensemble IC (rank-normalized average), then promotes whichever achieves the highest IC. Inference reads `gbm_mode.json` from S3 to determine which mode to load. Config: `ensemble_lambdarank: true` (default).

4. **Inactive MLP alternative** — `model/predictor.py` contains a DirectionPredictor MLP (v1.2.0) that could be activated as an ensemble member. Lowest priority — temporal ensemble (item 2) likely adds more value. Interface already exists.

### Feature Engineering Gaps

| Feature Category | Current | Missing |
|-----------------|---------|---------|
| Momentum | 7+ variants (RSI, MACD, momentum_5d/20d, price_accel) | VWAP divergence, buying/selling pressure |
| Volume | rel_volume_ratio, volume_trend, obv_slope | Volume profile, VWAP distance |
| Volatility | atr_14_pct, realized_vol_20d, vol_ratio_10_60 | Options-implied vol, IV rank |
| Cross-asset | sector_vs_spy_5d | Beta vs sector, correlation vs peers, breadth |
| Macro | VIX, yields, gold/oil momentum | Credit spreads, PMI, real rates |
| Regime interactions | 5 interaction terms (v1.5) | Yield curve x sector sensitivity |
| Fundamental | None | Earnings surprise, analyst revisions |

Key cross-asset opportunities:
- **Beta vs sector**: per-ticker rolling beta against sector ETF captures sensitivity to sector moves
- **Correlation vs peers**: rolling correlation with top-5 holdings in the same sector detects regime shifts
- **Breadth**: advance/decline ratio, % above 50d MA — confirms or contradicts aggregate index moves

### Training Pipeline Gaps

1. ~~**Walk-forward fold generation mixes calendar and trading days**~~ — **Resolved 2026-03-18.** Config key renamed to `test_window_trading_days` (with backward compat fallback) to clarify that fold windows are in trading days. Walk-forward fast mode added (`wf_n_estimators: 500`, `wf_early_stopping: 20`) to reduce per-fold training time ~4x. Per-fold timing logged.

2. **No automated feature selection** — features are manually curated across 5 generations. SHAP-based noise feature detection now identifies candidates (features with SHAP < 1% of top feature AND |IC| < 0.005), logged as informational warnings in the training email. Auto-pruning (which requires syncing feature lists between training and inference) is deferred.

3. ~~**SHAP feature importance monitoring**~~ — **Implemented 2026-03-17.** Weekly training computes SHAP TreeExplainer values (capped at 500 test rows), writes `shap_{date}.json` to S3, and includes a gain-vs-SHAP rank comparison table in the training email. Week-over-week Spearman rank correlation detects feature drift (warning threshold: 0.80).

4. ~~**Per-feature IC tracking**~~ — **Implemented 2026-03-18.** Pearson IC computed for each of the 36 GBM features against forward returns on the test set. Top 5 and bottom 5 features by |IC| logged. Results included in S3 metadata and training email under "Feature Health".

5. **Macro features excluded from GBM** — VIX, yields, gold/oil momentum are cross-sectional constants (same for all tickers on a given day) and cannot predict cross-sectional alpha. They are excluded from GBM training/inference. Regime interaction terms (e.g., `mom5d_x_vix`) partially address this by capturing macro × ticker signal interactions. Per-feature IC monitoring now tracks their effectiveness.

### Label Construction

1. ~~**Label winsorization at ±15% clips earnings gap alpha**~~ — **Resolved 2026-03-18.** Label clip raised from ±15% to ±25%. Sector-neutral 5d returns >25% are extremely rare, but 15-25% earnings gaps are informative and no longer clipped.

2. **Adaptive thresholds** — now supported via `adaptive_thresholds: true` in config (implemented 2026-03-17). Rolling percentile-based UP/DOWN classification adapts to volatility regime. Needs validation on historical data before enabling in production.

---

## Related Modules

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor (trade execution + system overview)
- [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) — Autonomous LLM research pipeline
- [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) — Signal quality analysis and parameter optimization
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard

---

## License

MIT — see [LICENSE](LICENSE).
