# Inference — alpha-engine-predictor

## Daily inference workflow

The predictor Lambda runs 30 minutes after the research pipeline completes
(6:15am PT on trading days). This ensures the research pipeline's price
data is available in S3 before the predictor needs to fetch prices.

```
EventBridge (6:15am PT, trading days)
    │
    ▼
Lambda: alpha-engine-predictor-inference
    │
    ├── 1. Load model weights
    │       s3://alpha-engine-research/predictor/weights/latest.pt → /tmp/
    │       Extract norm_stats (mean/std) from checkpoint
    │
    ├── 2. Get ticker universe
    │       Read signals/signals_YYYYMMDD.json from S3
    │       Falls back to hardcoded 15-ticker list on failure
    │
    ├── 3. Fetch prices (yfinance)
    │       1-year OHLCV for each ticker (enough history for MA200)
    │       Batch download in groups of 100
    │
    ├── 4. Compute features + inference (per ticker)
    │       compute_features(df) → 8-feature vector (latest row)
    │       z-score normalize using checkpoint's norm_stats
    │       model(x) → logits → softmax → [p_down, p_flat, p_up]
    │       Skip tickers with < 205 rows of history
    │
    └── 5. Write output to S3
            predictions/YYYY-MM-DD.json  (dated archive)
            predictions/latest.json      (overwritten daily)
            metrics/latest.json          (model health summary)
```

Total Lambda runtime: typically 60–120 seconds for ~500 tickers on a 1GB
Lambda with PyTorch CPU-only.

---

## Output schema

### predictions/latest.json

Written daily at both `predictor/predictions/YYYY-MM-DD.json` (archive) and
`predictor/predictions/latest.json` (latest pointer).

```json
{
  "date": "2026-03-11",
  "model_version": "v1.0.0",
  "model_hit_rate_30d": 0.581,
  "n_predictions": 487,
  "n_high_confidence": 94,
  "predictions": [
    {
      "ticker": "LLY",
      "predicted_direction": "UP",
      "prediction_confidence": 0.74,
      "p_up": 0.74,
      "p_flat": 0.18,
      "p_down": 0.08
    },
    {
      "ticker": "COST",
      "predicted_direction": "FLAT",
      "prediction_confidence": 0.61,
      "p_up": 0.28,
      "p_flat": 0.61,
      "p_down": 0.11
    }
  ]
}
```

Predictions are sorted by `p_up - p_down` descending (most bullish first).

**Field descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | string | Ticker symbol |
| `predicted_direction` | string | "UP", "FLAT", or "DOWN" |
| `prediction_confidence` | float | Max softmax probability (confidence in the prediction) |
| `p_up` | float | Probability of >+1% 5-day return |
| `p_flat` | float | Probability of return in [-1%, +1%] |
| `p_down` | float | Probability of <-1% 5-day return |

Note: `p_up + p_flat + p_down = 1.0` by construction (softmax output).

### metrics/latest.json

Model health summary written after each inference run and updated after
monthly retraining.

```json
{
  "model_version": "v1.0.0",
  "last_trained": "2026-02-01",
  "training_samples": 185420,
  "test_hit_rate": 0.581,
  "hit_rate_30d_rolling": 0.562,
  "ic_30d": 0.067,
  "ic_ir_30d": 0.41,
  "n_predictions_today": 487,
  "n_high_confidence": 94,
  "last_run_utc": "2026-03-11T14:15:00Z",
  "status": "ok"
}
```

`status` is "ok" when the Lambda completed successfully, "error" if it failed.

---

## Confidence gate

`prediction_confidence` is the max softmax probability, i.e., the model's
self-reported certainty for its predicted direction.

The production gate is `MIN_CONFIDENCE = 0.65` (set in `config.py`).

- Predictions with confidence ≥ 0.65 are considered actionable and will be
  used as score modifiers in the research pipeline.
- Predictions below this threshold are still written to `predictions.json`
  and visible in the dashboard, but are not applied as score modifiers.

**Why 0.65?** For a 3-class problem, a random classifier produces max
probabilities near 0.40–0.50 (uniform distribution). Requiring ≥ 0.65
filters out predictions where the model is uncertain and focuses the modifier
on high-conviction calls. This threshold can be adjusted in `config.py`.

A well-calibrated model should show higher hit rates for predictions above
the confidence gate. The dashboard's "Hit Rate by Confidence Bucket" chart
validates this calibration.

---

## Integration with alpha-engine-research

Predictions are consumed by the research pipeline via the following pattern
(see design doc §7.1 for full detail):

1. `fetch_data()` in `graph/research_graph.py` reads `predictions/latest.json`
   from S3 at the start of each research run (best-effort, `{}` on failure).

2. `run_universe_agents()` merges predictor values into the `indicators` dict:
   ```python
   pred = state["predictions"].get(ticker, {})
   if pred:
       indicators.update({
           "p_up": pred["p_up"],
           "p_flat": pred["p_flat"],
           "p_down": pred["p_down"],
           "prediction_confidence": pred["prediction_confidence"],
           "predicted_direction": pred["predicted_direction"],
       })
   ```

3. `compute_technical_score()` applies a confidence-gated modifier:
   ```python
   if p_up is not None and confidence >= 0.65:
       direction_signal = (p_up - p_down) * 10.0 * confidence
       composite = composite + direction_signal
   ```

If the predictor Lambda hasn't run (e.g., network failure, first day of
deployment), `predictions` defaults to `{}` and the research pipeline
proceeds with existing logic unchanged. This fallback is critical for
operational resilience.

---

## Running inference locally

For testing and development, use the `--local` flag to load from
`checkpoints/best.pt` instead of S3:

```bash
# Dry run: compute predictions and print to stdout without S3 writes
python inference/daily_predict.py --local --dry-run

# Override date (useful for backtesting)
python inference/daily_predict.py --local --date 2026-03-01 --dry-run

# Full local run (writes to S3)
python inference/daily_predict.py --local
```

Requires a trained model at `checkpoints/best.pt`. Run `python train.py`
first if no checkpoint exists.
