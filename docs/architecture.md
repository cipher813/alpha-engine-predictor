# Architecture — alpha-engine-predictor

## Model versions

| Version | Name | Status | IC (test) | Lambda `model_type` |
|---------|------|--------|-----------|---------------------|
| v1 | MLP — Direction Predictor | Available, not in production | ~0.033 | `mlp` |
| v1.5 | GBM — LightGBM alpha scorer | **Production** (2026-03-12) | ~0.046 | `gbm` |
| v2 | TFT — Temporal Fusion Transformer | Planned | — | — |

The Lambda handler (`inference/handler.py`) currently runs `model_type="gbm"`.

---

## GBM v1.5 — LightGBM Alpha Scorer (Production)

### Overview

A LightGBM gradient-boosted tree model that takes a 21-feature snapshot per
ticker and outputs a continuous predicted alpha (5-day return minus sector ETF
return). The continuous score is thresholded at inference time to produce a
3-class direction prediction identical in schema to the MLP output.

### Why GBM over MLP for tabular financial data

| Property | MLP | GBM |
|---|---|---|
| Feature normalization required | Yes (z-score) | No |
| Handles non-linear thresholds | Moderate | Excellent (e.g. RSI > 70 AND dist_52w_high < −0.1) |
| Feature importance (interpretable) | SHAP-only | Gain-based + SHAP |
| Cold-start Lambda | Slow (PyTorch load) | Fast (~100ms) |
| Lambda deployment | Container image (PyTorch 350MB) | Zip (lightgbm 11MB) |
| Test IC | ~0.033 | ~0.046 |

### Training objective

```
Objective  : regression (MSE) on sector-neutral 5-day return
             y = (close[T+5]/close[T] - 1) - (sector_etf[T+5]/sector_etf[T] - 1)
Eval metric: Pearson IC (cross-sectional, same as MLP)
Library    : lightgbm >= 4.0.0
```

Sector-neutral labels (introduced 2026-03-12) replace SPY-relative labels.
Using the sector ETF as benchmark removes industry momentum that the model
cannot reliably time, leaving only stock-specific alpha for the GBM to learn.

### Feature set (21 features)

| Feature | Description |
|---|---|
| `rsi_14` | 14-day RSI |
| `macd_cross` | MACD signal-line crossover {−1, 0, +1} |
| `macd_above_zero` | MACD histogram sign {0, 1} |
| `macd_line_last` | MACD line value |
| `price_vs_ma50` | % distance from 50-day MA |
| `price_vs_ma200` | % distance from 200-day MA |
| `momentum_20d` | 20-day price return |
| `avg_volume_20d` | Normalised 20-day avg volume |
| `dist_52w_high` | % below 52-week high |
| `dist_52w_low` | % above 52-week low |
| `atr_14_pct` | ATR(14) as % of price (volatility) |
| `bb_pct_b` | Bollinger Band %B |
| `volume_trend` | 5d vs 20d volume ratio |
| `price_accel` | 5d momentum minus 20d momentum |
| `rsi_slope` | 5-day RSI slope |
| `macd_histogram` | MACD histogram value |
| `ema_cross_8_21` | EMA(8)/EMA(21) − 1 |
| `sector_vs_spy_5d` | Sector ETF 5d return − SPY 5d return |
| Macro features | SPY 5d return, VIX level, TNX yield (3 features) |

GBM is scale-invariant; features are passed raw (no z-score normalization).
`norm_stats.json` is still generated but only used by the MLP path.

### S3 artifacts

```
s3://alpha-engine-research/
  predictor/weights/gbm_latest.txt        ← LightGBM booster text file (production)
  predictor/weights/gbm_YYYYMMDD.txt      ← dated snapshots (on retrain)
  predictor/predictions/YYYY-MM-DD.json   ← per-ticker predictions
  predictor/predictions/latest.json       ← always points to today
  predictor/metrics/latest.json           ← model health (IC, n_predictions, etc.)
```

### Output schema (per ticker in `predictions/latest.json`)

```json
{
  "ticker": "AAPL",
  "predicted_direction": "UP",
  "prediction_confidence": 0.72,
  "predicted_alpha": 0.0083,
  "p_up": 0.72,
  "p_flat": 0.0,
  "p_down": 0.28,
  "watchlist_source": "tracked"
}
```

`predicted_direction` is derived from the continuous `predicted_alpha` score:
`UP` if alpha > threshold, `DOWN` if alpha < −threshold, `FLAT` otherwise.
The `prediction_confidence` maps directly to the argmax probability for
compatibility with the MLP schema consumed by alpha-engine-research.

### Option A — Veto gate (active)

The GBM output feeds an ENTER-veto gate in the research consolidator:
- `predicted_direction == "DOWN"` **and** `confidence ≥ 0.60` → ENTER signal
  downgraded to HOLD
- Surfaced in the research email as `↓✗` in the GBM column of the thesis table
- Does not affect EXIT, REDUCE, or HOLD signals

---

## MLP v1 — Direction Predictor

The v1 model is a feedforward Multi-Layer Perceptron (MLP) that takes a single
8-feature technical indicator snapshot for a ticker and outputs a 3-class
softmax distribution over predicted 5-day return direction.

### Network diagram

```
Input layer
┌─────────────────────────────────────────────────┐
│  rsi_14  macd_cross  macd_above_zero  macd_line  │   8 features
│  price_vs_ma50  price_vs_ma200  momentum  vol    │   (z-score normalized)
└─────────────────────────────────────────────────┘
                         │
                    Linear(8→64)
                    BatchNorm1d(64)
                    ReLU
                    Dropout(0.3)
                         │
                    Linear(64→32)
                    BatchNorm1d(32)
                    ReLU
                    Dropout(0.2)
                         │
                    Linear(32→3)
                         │
           ┌─────────────┼─────────────┐
         P(DOWN)       P(FLAT)       P(UP)
          [0]           [1]           [2]
```

Output: raw logits during training (CrossEntropyLoss applies log-softmax
internally). At inference time, `F.softmax(logits, dim=-1)` converts to
probabilities.

**Predicted direction** = `argmax([P(DOWN), P(FLAT), P(UP)])`
**Confidence** = the maximum probability value.

### Why MLP first

The current 8 features are all point-in-time snapshots (no sequence dependency).
An MLP can capture the full information content of these features without the
added complexity of an RNN or Transformer. If the IC from an MLP is near zero,
the features don't carry signal — changing the architecture won't fix that.

Once MLP achieves IC > 0.05 and hit rate > 55%, the signal is confirmed and
upgrading to TFT (v2) is justified on the basis of extracting more signal from
the same features by modeling their temporal evolution.

### BatchNorm and Dropout

BatchNorm is applied before the activation in each hidden layer. This:
- Stabilizes training by reducing internal covariate shift
- Acts as a regularizer, reducing the need for high dropout rates
- Enables higher learning rates

Dropout rates are moderate (0.3 / 0.2) — financial tabular data benefits from
some regularization but overfit risk is lower than text/image domains.

### Class label mapping

| Index | Class | Forward return condition |
|-------|-------|--------------------------|
| 0     | DOWN  | return < −1%             |
| 1     | FLAT  | −1% ≤ return ≤ +1%       |
| 2     | UP    | return > +1%             |

The ±1% threshold was chosen to exclude near-zero returns that are effectively
noise. Tightening (e.g., ±2%) reduces the FLAT class and creates a harder
classification problem with more signal; widening reduces class imbalance.

---

## Feature rationale

| Feature | Why included | Range after z-score |
|---------|-------------|---------------------|
| `rsi_14` | Mean-reversion signal; oversold/overbought | ~[-3, +3] |
| `macd_cross` | Momentum direction change event | {-1, 0, +1} |
| `macd_above_zero` | Trend regime (MACD above/below zero) | {0.0, 1.0} |
| `macd_line_last` | Continuous momentum signal | ~[-3, +3] |
| `price_vs_ma50` | Short-term trend position | ~[-3, +3] |
| `price_vs_ma200` | Long-term trend position | ~[-3, +3] |
| `momentum_20d` | 20-day price return | ~[-3, +3] |
| `avg_volume_20d` | Volume trend (relative to series mean) | ~[0, 10+] |

All features are computed with the same EWM/rolling parameters as
`compute_technical_indicators()` in alpha-engine-research, ensuring that
training and inference use identical feature semantics.

---

## TFT v2 — Upgrade path

The Temporal Fusion Transformer (TFT) is the designated v2 architecture.
It extends the MLP by treating each feature as a time series over a 30-day
lookback window rather than a single snapshot. This allows the model to learn
temporal patterns like "RSI trending up for 5 days" that a point-in-time MLP
cannot capture.

### TFT architecture (v2 reference)

```
Static covariates (sector one-hot)
    │
    ▼
Variable Selection Network  ←── selects relevant input features per timestep
    │
    ▼
Encoder LSTM (historical window, length=30)
    │
    ▼
Multi-head Self-Attention   ←── learns which past timesteps matter
    │
    ▼
Feed-forward layers
    │
    ▼
Linear(3) → Softmax → [P(DOWN), P(FLAT), P(UP)]
```

Library: `pytorch-forecasting.TemporalFusionTransformer`

### What changes in the codebase for TFT

| File | Change |
|------|--------|
| `data/dataset.py` | Replace `(features, label)` pairs with `(sequence_tensor, static_covariates, label)` |
| `model/predictor.py` | Replace `DirectionPredictor` MLP with TFT model |
| `model/trainer.py` | Switch to `pytorch-forecasting` Trainer |
| `inference/daily_predict.py` | Feed last 30 days of features, not just today's snapshot |

### What does NOT change

Everything else is unchanged: bootstrap fetcher, feature engineer, label
generator, config, S3 output schema, Lambda handler, research integration.

### Validation gate before upgrading

Do not migrate to TFT until MLP achieves all three on held-out test data:
- Hit rate > 55%
- IC > 0.05
- IC IR > 0.3 over rolling 20-day windows

If MLP plateaus below these thresholds after feature iteration, investigate
whether the features need expansion before upgrading the architecture.
