# Training — alpha-engine-predictor

## Data pipeline overview

```
data/bootstrap_fetcher.py
    └── Downloads 5y OHLCV for ~900 tickers (S&P 500 + S&P 400)
    └── Saves data/cache/{ticker}.parquet

data/feature_engineer.py
    └── compute_features(df) → rolling 17-feature vectors per row
    └── Warmup: 252 rows (for 52-week high/low)

data/label_generator.py
    └── compute_labels(df) → forward_return_5d, direction, direction_int
    └── Labels are RELATIVE to SPY when spy_series is provided
    └── Drops last 5 rows (no forward return available)

data/dataset.py
    └── build_regression_datasets() → train/val/test DataLoaders
    └── Time-based split: 70/15/15 with date-boundary alignment
    └── DateGroupedSampler: each batch = all stocks on one trading date
    └── Z-score normalization (fit on train only)
    └── Saves data/norm_stats.json
```

---

---

## Dataloader architecture — theory and construction

### What a "sample" is

One sample in the dataset represents **one ticker on one specific trading date**.
It is a tuple of `(feature_vector, forward_return_5d)`:

- **feature_vector**: a 17-element float32 array of rolling technical indicators
  computed from all price history available up to and including date T. The model
  never sees any price data after T when computing features.
- **forward_return_5d**: the 5-trading-day return starting from date T, computed
  as `(close[T+5] / close[T]) - 1`, then made relative to SPY when SPY data is
  available. This is the label — what the model is trained to predict.

Crucially, **the feature date (T) and the label date (T+5) are different**. The
feature vector is a snapshot of market state at T. The label is what happens
_after_ T. The model never sees the label during a forward pass — it only sees
the feature vector.

### The per-day grouping: DateGroupedSampler

After features and labels are computed for all ~900 tickers across 5 years, the
dataset contains roughly **1.1 million samples** (900 tickers × ~1250 trading
days × ~70% warmup survival rate).

The `DateGroupedSampler` (in `data/dataset.py`) arranges these samples so that
**each mini-batch contains all tickers observed on one specific trading date** —
typically ~800–900 stocks per batch.

```
Mini-batch 1: AAPL/2021-03-15, MSFT/2021-03-15, NVDA/2021-03-15 ... (~850 stocks)
Mini-batch 2: AAPL/2021-07-22, MSFT/2021-07-22, NVDA/2021-07-22 ... (~830 stocks)
Mini-batch 3: AAPL/2022-11-08, MSFT/2022-11-08, NVDA/2022-11-08 ... (~870 stocks)
```

The order of dates across epochs is shuffled (training) or sequential (val/test).
Each epoch sees every trading date exactly once.

### Why per-day batching is the correct structure for a quant model

The loss function is `ICLoss` — the negative Pearson correlation between
predictions and actual 5-day returns within a batch:

```
IC = Pearson(model_predictions, actual_returns)
   = corr([pred_AAPL, pred_MSFT, pred_NVDA, ...], [ret_AAPL, ret_MSFT, ret_NVDA, ...])
```

This directly optimizes for **cross-sectional ranking ability**: the capacity to
correctly order stocks by expected return on a single trading day.

**Why per-day batching is required for IC loss to be meaningful:**

If stocks from different dates were mixed in a batch, the correlation would
conflate time-series variation with cross-sectional variation. For example, a
batch containing AAPL on a 2021 bull-market day and MSFT on a 2023 rate-hike
day would produce a meaningless IC — the returns differ for macro reasons
unrelated to the model's cross-sectional ranking signal.

With per-day batches, every gradient step answers the correct question:
**"On this specific day, did the model rank the cross-section of stocks
correctly?"** This aligns the training objective with how alpha signals are
used in production: the model score is rank-transformed within each day's
universe before portfolio construction.

The IC loss is also scale-invariant by design. A model that outputs
`[0.01, 0.02, 0.03]` and one that outputs `[1.0, 2.0, 3.0]` have identical
IC — both rank the three stocks the same way. This is the correct property
for a ranking-based alpha signal, unlike MSE or Huber loss which penalize
prediction magnitude.

### How this differs from the train/test split

The per-day grouping is a property of **how samples are batched during
training**. It is orthogonal to which samples are in train vs val vs test.

The train/test split is purely temporal:

```
All 1.1M samples sorted by date (T, the feature date):

Oldest ───────────────────────────────────────────── Newest
│        70% train         │   15% val   │  15% test  │
│  (e.g. Jan 2019–Oct 2022)│(Oct 22–May 23)│(May 23–Dec 24)│
└──────────────────────────┴─────────────┴────────────┘
```

Each of these three sets contains samples from ~900 tickers. The test set is
the most recent ~9 months of data across all tickers — the closest proxy for
live performance.

**Does the test set cover the 5 days after the training cutoff?** No — and this
is a common source of confusion. The split is based on the **feature date T**,
not the label date T+5. A sample with feature date T=2022-10-28 (last day of
train) has a label computed from close[2022-11-04]. The first val sample has
feature date T=2022-10-31. There is a 5-day overlap in the price data used
to compute labels at the train/val boundary, but the model never sees those
forward prices as inputs. This minor boundary overlap is unavoidable with any
forward-return labeling scheme and is standard in quantitative ML.

The test set spans a continuous later period in calendar time. It is not
"the 5 days immediately after training ends" — it is a full 15% slice of
the total date range, evaluated sample-by-sample across all tickers.

### Date-boundary alignment in the split

The split indices in `build_regression_datasets()` are advanced to the nearest
date boundary to prevent any single trading date from being split across two
sets. If the 70% index lands mid-day (e.g., between the 400th and 401st ticker
on 2022-10-28), all remaining tickers from that date are moved to the train set.
This ensures that `DateGroupedSampler` always sees complete cross-sections in
each set — no partial-date batches.

### Summary: the full flow for one training step

```
1. DateGroupedSampler yields all sample indices for trading date T
2. DataLoader assembles those ~850 (feature_vector, forward_return) pairs
3. Model forward pass: feature_vector → scalar return prediction (regression)
4. ICLoss: Pearson(predictions, actual_returns) computed across the ~850 stocks
5. Backprop: gradient signal = "did you rank the cross-section correctly on date T?"
6. Optimizer step: weights updated to improve cross-sectional ranking
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

### Label winsorization

Before training, forward returns are clipped at `±LABEL_CLIP` (default ±15%).
This eliminates the gradient noise introduced by extreme events — earnings gaps,
M&A announcements, biotech FDA outcomes — that produce ±20–50% 5-day moves.
These outliers are real, but a simple MLP has no way to predict them from
technical indicators. Including them unclipped causes `ICLoss` to spend gradient
budget on noise instead of the typical ±2–4% signal range.

```python
# config.py
LABEL_CLIP = 0.15  # set to None to disable
```

The dataset logger reports how many samples were clipped and the label
percentile distribution after clipping so you can verify the clip level
is appropriate for the current universe.

---

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

### v1.3 macro regime features

Four market-wide features added in v1.3 capture interest rate regime and
commodity cycle signals that technical price indicators cannot express.

**yield_10y** — `TNX / 10.0`
The 10-Year Treasury yield normalized to 0–1 scale. High rates increase the
discount rate for equity cash flows; the level also signals whether the Fed is
in a tightening or easing cycle. `^TNX` from yfinance provides this in percent
(e.g. 4.5), forward-filled across equity trading days.

**yield_curve_slope** — `(TNX - IRX) / 10.0`
The spread between 10Y and 3M Treasury yields. Positive = normal/steep curve
(growth expected); negative = inverted (historical recession precursor).
`^IRX` (13-week T-bill) is the short leg. Forward-filled like `^TNX`.

**gold_mom_5d** — `(GLD[T] / GLD[T-5]) - 1`
5-day momentum of the GLD gold ETF. Rising gold signals risk-off positioning
or inflation expectations — both environments that historically suppress equity
alpha from technical momentum strategies. Falling gold implies risk-on.

**oil_mom_5d** — `(USO[T] / USO[T-5]) - 1`
5-day momentum of the USO oil ETF (WTI crude proxy). Directly relevant for
energy sector stocks; indirectly affects consumer discretionary, airlines, and
industrials through input cost sensitivity. Also an inflation leading indicator.

These four features are market-wide (identical across all tickers on a given
date) so they follow the same loading pattern as VIX: downloaded once as
parquets by `bootstrap_fetcher.py`, loaded at the dataset level, and aligned
to each ticker's date index with forward-fill.

---

### On FRED integration

If you have FRED API access (which the alpha-engine-research stack already
uses), the `DFF` (Daily Effective Federal Funds Rate) series is a more precise
source for the short rate leg than `^IRX` (which is the 3M T-bill market rate,
a close but not identical proxy). To use it, add a `data/macro_fetcher.py` that
fetches `DFF` and `GS10` from FRED and saves them as `FEDFUNDS.parquet` and
`TNX_FRED.parquet`, then swap the `irx_series` loader in `dataset.py` to read
`FEDFUNDS.parquet`. The feature computation in `feature_engineer.py` is
unchanged — it only cares about the aligned pandas Series.

---

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

## Hyperparameter tuning

`tune.py` runs an Optuna search before full training. It samples hyperparameter
combinations, trains each for a reduced number of epochs, and saves the best
configuration to `checkpoints/best_params.json`. `train.py` loads this file
automatically on the next run.

### Running the tuner

```bash
# Default: 50 trials, 30 epochs each (~15 min on CPU with 900 tickers)
python tune.py --data-dir data/cache

# Faster search for iteration:
python tune.py --data-dir data/cache --trials 20 --epochs 15

# Resume a previous study (requires --storage):
python tune.py --data-dir data/cache --storage sqlite:///tune.db

# Then train with the best params:
python train.py --data-dir data/cache
```

The retrain shell script runs tuning automatically before `train.py`:
```bash
./infrastructure/retrain.sh --local               # tune (50 trials) then train
./infrastructure/retrain.sh --local --skip-tune   # skip tune, use existing best_params.json
TUNE_TRIALS=20 ./infrastructure/retrain.sh --local  # faster search
```

### Search space

| Parameter | Type | Range | Notes |
|-----------|------|-------|-------|
| `HIDDEN_1` | categorical | 32, 64, 128, 256 | First hidden layer width |
| `HIDDEN_2` | categorical | 16, 32, 64, 128 | Must be ≤ HIDDEN_1 in practice |
| `DROPOUT_1` | float | 0.1 – 0.5 | Step 0.1 |
| `DROPOUT_2` | float | 0.0 – 0.4 | Step 0.1; typically < DROPOUT_1 |
| `LEARNING_RATE` | log-float | 1e-4 – 5e-3 | Log scale; TPE explores this well |
| `WEIGHT_DECAY` | log-float | 1e-5 – 1e-3 | L2 regularization |
| `UP_THRESHOLD` | categorical | 0.5%, 1%, 1.5%, 2% | Controls class balance; DOWN = symmetric |

**Why `UP_THRESHOLD` is the most impactful knob:** it determines the FLAT class
width. A ±1% threshold on a 5-day return produces roughly 30/40/30 DOWN/FLAT/UP
class balance at historical equity volatility. Widening to ±2% shifts more samples
into FLAT (~55%), making UP/DOWN harder to learn but cleaner signal when predicted.
Narrowing to ±0.5% increases class frequency but adds label noise.

### Sampler and pruner

- **TPE sampler** (default): Tree-structured Parzen Estimator. Builds a probabilistic
  model of the objective function and samples from the more promising regions.
  Significantly more efficient than random search after ~10 trials.
- **MedianPruner**: kills trials whose intermediate val_loss is worse than the median
  of completed trials at the same epoch. Skips `n_startup_trials=10` to let the
  sampler warm up first.
- Both are seeded at 42 for reproducibility.

### Per-trial early stopping

Each trial uses a patience of 5 (shorter than the full training patience of 10) to
keep per-trial wall time under ~30 seconds on CPU. The objective value is the best
validation loss achieved across all epochs in the trial.

### best_params.json schema

```json
{
  "best_params": {
    "HIDDEN_1": 128,
    "HIDDEN_2": 64,
    "DROPOUT_1": 0.2,
    "DROPOUT_2": 0.1,
    "LEARNING_RATE": 0.0008,
    "WEIGHT_DECAY": 0.00005,
    "UP_THRESHOLD": 0.01,
    "DOWN_THRESHOLD": -0.01
  },
  "best_val_loss": 1.0842,
  "best_trial_number": 31,
  "n_trials": 50,
  "n_pruned": 18
}
```

`train.py` reads `best_params` and applies each key as an override to the `config`
module before building datasets and instantiating the model. All other config values
(S3 paths, feature list, production gates) are unchanged.

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
