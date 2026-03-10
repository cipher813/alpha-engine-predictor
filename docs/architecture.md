# Architecture — alpha-engine-predictor

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
