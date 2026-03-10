# Training — alpha-engine-predictor

## Data pipeline overview

```
data/bootstrap_fetcher.py
    └── Downloads 5y OHLCV for ~900 tickers (S&P 500 + S&P 400)
    └── Saves data/cache/{ticker}.parquet

data/feature_engineer.py
    └── compute_features(df) → rolling 8-feature vectors per row
    └── Warmup: 200 rows (for MA200)

data/label_generator.py
    └── compute_labels(df) → forward_return_5d, direction, direction_int
    └── Drops last 5 rows (no forward return available)

data/dataset.py
    └── build_datasets() → train/val/test DataLoaders
    └── Time-based split: 70/15/15
    └── Z-score normalization (fit on train only)
    └── Saves data/norm_stats.json
```

---

## Bootstrap fetch

The bootstrap fetch is a one-time operation that populates the local parquet
cache. It downloads 5 years of daily OHLCV history (~1.1M ticker-days total).

```bash
python data/bootstrap_fetcher.py --output-dir data/cache
```

Options:
- `--upload` — upload parquets to S3 after saving locally (for SageMaker)
- `--period 3y` — reduce to 3 years if 5 years is too slow
- `--limit 50` — test with 50 tickers first
- `--tickers AAPL MSFT NVDA` — override universe

Failed tickers (delisted, no data) are logged to `data/cache/failed_tickers.txt`.

### Survivorship bias note

The bootstrap fetches current S&P 500/400 constituents only (survivors).
This introduces upward bias because stocks that performed poorly enough to be
dropped from the index are excluded from training. Mitigation options:
- Use historical constituent lists (CRSP, Compustat, or Bloomberg)
- Limit training labels to rolling 12-month windows per ticker to reduce
  the effective look-ahead horizon
- Track this bias in the evaluation IC/hit rate vs. a universe-neutral baseline

---

## Feature computation

Features are computed by `data/feature_engineer.py::compute_features()`,
which mirrors `compute_technical_indicators()` from alpha-engine-research
exactly — same EWM parameters, same rolling windows, same normalization logic.

### RSI(14)

Uses Wilder's exponential smoothing: `ewm(com=13, adjust=False)`. This is
equivalent to a 14-period smoothed RSI but with EWM's stability properties.

```
delta = close.diff()
avg_gain = delta.clip(lower=0).ewm(com=13).mean()
avg_loss = (-delta.clip(upper=0)).ewm(com=13).mean()
rsi = 100 - (100 / (1 + avg_gain / avg_loss))
```

Range: [0, 100]. Values near 0 = oversold; values near 100 = overbought.

### MACD (12/26/9)

```
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
```

`macd_cross`: looks back 3 rows for a cross event (+1 bullish, -1 bearish, 0 none).
`macd_above_zero`: binary — whether MACD line is above zero.
`macd_line_last`: raw continuous MACD value (captures magnitude).

### Moving averages

`price_vs_ma50 = (close - ma50) / ma50` — ratio, not percent.
`price_vs_ma200 = (close - ma200) / ma200` — ratio, not percent.

Both use `rolling(window=200, min_periods=200).mean()` so rows before the
200th index produce NaN and are dropped. This is the primary warmup constraint.

### Momentum and volume

`momentum_20d = (close / close.shift(20)) - 1` — pure 20-day price return.

`avg_volume_20d`: 20-day rolling mean volume, divided by the global mean
volume of the entire series. This normalizes for large differences in average
daily volume across tickers (a small-cap with 1M ADV and a large-cap with
100M ADV both produce values near 1.0 in normal conditions).

---

## Split strategy

**Time-based split — no shuffling across the split boundary.**

All samples across all tickers are merged into a single array and sorted by
date. The split boundaries are then applied by index:

```
Oldest ────────────────────────────────────────── Newest
│       70% train       │   15% val   │  15% test  │
└───────────────────────┴─────────────┴────────────┘
```

**Why time-based?** Randomly shuffling before splitting would leak future
information into the training set — a sample from 2025 appearing in the train
split when a nearby 2020 sample is in the test split. In financial ML this
manifests as inflated test metrics that don't hold up in production. The test
set is always the most recent 15% of the historical data, which is the closest
proxy for live performance.

Intra-split shuffling (within the training set) is fine and improves gradient
stability — only the split boundaries are time-ordered.

---

## Normalization

Z-score normalization is applied per feature:

```python
x_norm = (x - mean_train) / std_train
```

- `mean_train` and `std_train` are computed on the **training set only**.
- The same statistics are applied to val and test sets (no fitting on val/test).
- Statistics are saved to `data/norm_stats.json` and embedded in every
  model checkpoint so inference-time normalization is always consistent.

If a feature has zero variance in the training set (rare, e.g., `macd_above_zero`
could be constant for a small dataset), `std` is clamped to 1.0 to avoid division
by zero.

---

## Hyperparameters

All hyperparameters are in `config.py`. Key values and tuning guidance:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `HIDDEN_1` | 64 | First hidden layer width. Try 128 if underfitting. |
| `HIDDEN_2` | 32 | Second hidden layer width. Bottleneck to prevent overfitting. |
| `DROPOUT_1` | 0.3 | Higher = more regularization. Don't exceed 0.5. |
| `DROPOUT_2` | 0.2 | Lower than DROPOUT_1 — closer to output, less dropout. |
| `BATCH_SIZE` | 512 | Large batches work well with BatchNorm. Reduce if OOM. |
| `LEARNING_RATE` | 1e-3 | AdamW default. ReduceLROnPlateau will halve automatically. |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization. Increase if val_loss diverges from train_loss. |
| `MAX_EPOCHS` | 100 | Early stopping typically fires before this. |
| `EARLY_STOPPING_PATIENCE` | 10 | Epochs without val_loss improvement before stopping. |

### Tuning guidance

- **Underfitting** (train_loss stays high): increase `HIDDEN_1`, `HIDDEN_2`, or reduce dropout.
- **Overfitting** (val_loss diverges from train_loss): increase `DROPOUT_1`, `DROPOUT_2`, or `WEIGHT_DECAY`.
- **Slow convergence**: `LR` too low; try 3e-3 or 5e-3.
- **Unstable training**: gradient explosion — gradient clipping is already applied (`max_norm=1.0`).
- **Class imbalance**: if FLAT dominates, consider class-weighted CrossEntropyLoss.

---

## Evaluation metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Overall hit rate | > 55% | % correct direction predictions |
| UP hit rate | > 55% | % correct among UP predictions |
| DOWN hit rate | > 55% | % correct among DOWN predictions |
| Pearson IC | > 0.05 | Correlation of (p_up - p_down) vs actual return |
| IC IR | > 0.3 | IC / std(IC) over 20-day rolling windows |
| Direction Sharpe | > 0.5 | Annualized Sharpe of long-UP / short-DOWN strategy |

Random baseline for a 3-class problem is 33% — hit rate > 55% is meaningful
signal. IC > 0.05 is considered minimum investable signal in quantitative
equity research.

### Gate before production use

The model must pass all four gates before being enabled in `config/universe.yaml`:
1. Test set hit rate > 55%
2. Test set IC > 0.05
3. 30 days of live predictions with rolling hit rate > 55%
4. Backtester shows positive contribution when predictor modifier is applied

---

## Retraining schedule

**Frequency**: Monthly, first Sunday of each month.

**Trigger**: `infrastructure/retrain.sh` (manual or scheduled via EventBridge).

**Why monthly?** The 5-day forward return signal from technical indicators is
relatively stable — market regime changes (bull → bear) happen on monthly
timescales, not daily. Weekly retraining would add operational overhead without
meaningfully improving signal freshness. Annual retraining would allow the model
to drift significantly as market conditions change.

**What triggers an unscheduled retrain?**
- Rolling 30-day hit rate drops below 0.50 (near random)
- IC drops below 0.02 for two consecutive weeks
- Major market regime change (e.g., VIX spike above 40)
- New features added to the feature set

**Retraining data**: re-run bootstrap fetch to include the most recent ~30 days
of price history that was excluded from the initial bootstrap run. Old parquets
can be reused; only recent data needs refreshing.
