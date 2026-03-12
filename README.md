# alpha-engine-predictor

A standalone ML module that produces per-ticker 5-day return direction predictions
from 21 technical and macro features. It runs daily as an AWS Lambda function
(Python 3.12, zip deployment), reads price data via yfinance, and writes a
`predictions.json` to S3.

**Current production model**: LightGBM GBM (test IC ≈ 0.046, sector-neutral labels).
The MLP (PyTorch) is available for reactivation but is not running in the Lambda.

The research pipeline reads these predictions and uses them as:
1. A confidence-gated score modifier (+/− 10pt on technical score)
2. An ENTER-veto gate: predicted DOWN with ≥ 60% confidence blocks new position entries

The predictor is deliberately separate from the research and executor modules —
integration happens through S3 artifacts, not shared code.

---

## Architecture

```
  ┌────────────────────────────────────────────────────┐
  │          Weekly/monthly retraining (GBM)           │
  │  bootstrap_fetcher → feature_engineer → dataset    │
  │  → train_gbm.py → checkpoints/gbm_best.txt → S3   │
  │  s3://alpha-engine-research/predictor/weights/     │
  │         gbm_latest.txt                             │
  └────────────────────────────────────────────────────┘
                            │
  ┌─────────────────────────▼──────────────────────────┐
  │      Daily inference — Lambda (6:15am PT)          │
  │  EventBridge ae-predictor-run (13:15 UTC weekdays) │
  │  → load GBM booster from S3                        │
  │  → read watchlist from signals.json (S3)           │
  │  → fetch 1y OHLCV per ticker (yfinance)            │
  │  → compute 21 features (latest row)                │
  │  → GBM predict → direction + confidence            │
  │  → write predictions/YYYY-MM-DD.json + latest.json │
  └────────────────────────────────────────────────────┘
                            │
             s3://alpha-engine-research/predictor/predictions/latest.json
                            │
  ┌─────────────────────────▼──────────────────────────┐
  │          alpha-engine-research integration         │
  │  consolidator reads predictions (best-effort)      │
  │  Option A veto: DOWN ≥ 60% conf → ENTER → HOLD     │
  │  ±10pt confidence-gated modifier on tech score     │
  └────────────────────────────────────────────────────┘
```

**GBM (production):** LightGBM regression on sector-neutral 5-day alpha.
21 features, no normalization required, test IC ≈ 0.046.

**MLP (available, not running):** PyTorch feedforward, 8 features, z-score
normalized. Activate by setting `model_type="mlp"` in `handler.py` and
deploying as a container image (PyTorch requires Docker build).

---

## Quick start

### Prerequisites

- Python 3.12 (Lambda runtime); tests also pass on 3.9+
- pip, virtualenv or conda
- AWS CLI configured (for S3 reads/writes and deployment)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/alpha-engine-predictor.git
cd alpha-engine-predictor
pip install -r requirements.txt
```

### Bootstrap training data (one-time)

Downloads 5-year daily OHLCV for all S&P 500 + S&P 400 constituents (~900
tickers). Takes approximately 30–60 minutes depending on network speed.

```bash
python data/bootstrap_fetcher.py --output-dir data/cache

# Test with a small subset first:
python data/bootstrap_fetcher.py --output-dir data/cache --limit 20
```

### Train

```bash
python train.py --data-dir data/cache --device cpu --output checkpoints/
```

Training output:
- `checkpoints/best.pt` — best model by validation loss
- `checkpoints/eval_report.json` — test set metrics
- `data/norm_stats.json` — feature normalization statistics

### Evaluate

Evaluation runs automatically at the end of `train.py`. To evaluate an
existing checkpoint separately:

```python
from model.predictor import load_checkpoint
from model.evaluator import evaluate, compute_hit_rate
from data.dataset import build_datasets
import config as cfg

train_loader, val_loader, test_loader = build_datasets("data/cache", cfg)
model, checkpoint = load_checkpoint("checkpoints/best.pt")
results = evaluate(model, test_loader)
print(f"Test accuracy: {results['accuracy']:.1%}")
```

### Train the GBM (production model)

```bash
python train_gbm.py --data-dir data/cache
# Output: checkpoints/gbm_best.txt, checkpoints/gbm_eval_report.json
```

### Run inference locally (dry run)

```bash
# GBM (production model) — requires checkpoints/gbm_best.txt
python inference/daily_predict.py --local --model-type gbm --dry-run

# Watchlist mode (mirrors Lambda behaviour)
python inference/daily_predict.py --local --model-type gbm --watchlist auto --dry-run

# MLP — requires checkpoints/best.pt
python inference/daily_predict.py --local --model-type mlp --dry-run
```

Prints the predictions JSON to stdout without writing to S3.

### Run tests

```bash
pytest tests/ -v
```

All tests run without network access or AWS credentials.

---

## Integration with alpha-engine-research

The predictor enriches the technical scoring pipeline via a single dict merge.
No function signatures change; the modifier is skipped automatically if the
predictor hasn't run or confidence is below the gate.

**Required changes to alpha-engine-research** (see design doc §7.1):

1. `graph/research_graph.py` — `fetch_data()` reads `predictions/latest.json` from S3
2. `graph/research_graph.py` — `run_universe_agents()` merges predictor values into `indicators`
3. `scoring/technical.py` — `compute_technical_score()` applies confidence-gated modifier
4. `archive/manager.py` — add `load_predictions_json()` helper
5. `config/universe.yaml` — add `predictor:` section with `enabled: false` gate

Enabling the modifier:
```yaml
# config/universe.yaml
predictor:
  enabled: true           # flip once model passes production gates
  min_confidence: 0.65
```

---

## Deployment

See [docs/deployment.md](docs/deployment.md) for full walkthrough.

### Build and deploy Lambda (zip — no Docker required)

```bash
# See docs/deployment.md for the full build procedure.
# Short version:
pip install --platform manylinux_2_28_x86_64 --target /tmp/lambda-pkg \
  --implementation cp --python-version 3.12 --only-binary=:all: \
  -r requirements-lambda.txt
# (add libgomp.so.1 stub + scipy stub + source files)
cd /tmp/lambda-pkg && zip -r /tmp/predictor.zip . -q
aws s3 cp /tmp/predictor.zip s3://alpha-engine-research/lambda/alpha-engine-predictor.zip
aws lambda update-function-code --function-name alpha-engine-predictor-inference \
  --s3-bucket alpha-engine-research --s3-key lambda/alpha-engine-predictor.zip
```

### GBM retraining (weights update — no Lambda redeploy)

```bash
python train_gbm.py --data-dir data/cache
aws s3 cp checkpoints/gbm_best.txt \
  s3://alpha-engine-research/predictor/weights/gbm_latest.txt
```

---

## Directory structure

```
alpha-engine-predictor/
├── data/
│   ├── bootstrap_fetcher.py   # one-time 5y OHLCV fetch + sector_map.json
│   ├── feature_engineer.py    # rolling 21-feature computation (mirrors research)
│   ├── label_generator.py     # sector-neutral 5-day forward return labels
│   ├── dataset.py             # Dataset + DataLoader for MLP and GBM
│   └── norm_stats.json        # z-score stats (MLP only; bundled in Lambda zip)
├── model/
│   ├── predictor.py           # DirectionPredictor MLP + checkpoint save/load
│   ├── gbm_scorer.py          # GBMScorer (LightGBM) — production model
│   ├── trainer.py             # MLP training loop (early stopping, LR scheduler)
│   └── evaluator.py           # IC, hit rate, confusion matrix
├── inference/
│   ├── daily_predict.py       # daily prediction pipeline; --model-type gbm|mlp
│   └── handler.py             # AWS Lambda entry point (model_type="gbm")
├── infrastructure/
│   ├── deploy.sh              # container image build + ECR push (MLP path)
│   └── retrain.sh             # local or SageMaker retrain
├── tests/
│   ├── test_features.py       # feature engineering unit tests
│   ├── test_model.py          # model forward pass + checkpoint tests
│   └── test_inference.py      # predict_ticker() unit tests
├── docs/
│   ├── architecture.md        # GBM v1.5 + MLP v1 + TFT v2 upgrade path
│   ├── training.md            # data pipeline, split strategy, hyperparameters
│   ├── inference.md           # daily workflow, output schema, confidence gate
│   └── deployment.md          # AWS resources, scheduling, zip build procedure
├── config.py                  # S3 paths, hyperparameters, feature list
├── train.py                   # MLP training entry point
├── train_gbm.py               # GBM training entry point (production)
├── tune.py                    # Optuna hyperparameter search (MLP)
├── buildspec.yml              # AWS CodeBuild spec (container image path)
├── requirements.txt           # full dev dependencies
├── requirements-lambda.txt    # Lambda-only deps (no torch, no scipy, no pyarrow)
├── Dockerfile                 # container image (MLP/PyTorch path)
└── alpha-engine-predictor-design-260310.md
```

---

## Key configuration

All constants are in `config.py`. Notable values:

| Constant | Default | Description |
|----------|---------|-------------|
| `S3_BUCKET` | `alpha-engine-research` | S3 bucket for weights and predictions |
| `GBM_WEIGHTS_KEY` | `predictor/weights/gbm_latest.txt` | S3 key for GBM booster |
| `FEATURES` | 21 features | Must stay in sync with research pipeline feature list |
| `FORWARD_DAYS` | 5 | Prediction horizon (trading days) |
| `UP_THRESHOLD` | 0.01 | > +1% sector-neutral return = UP class |
| `DOWN_THRESHOLD` | -0.01 | < -1% sector-neutral return = DOWN class |
| `MIN_CONFIDENCE` | 0.65 | Below this, ±10pt modifier not applied in research |
| `GBM_VETO_CONFIDENCE` | 0.60 | DOWN prediction at or above this blocks ENTER signals |
| `MIN_HIT_RATE` | 0.55 | Required for production deployment |
| `MIN_IC` | 0.05 | Required for production deployment |
| `TRAIN_FRAC` | 0.70 | Time-based training split |
| `HIDDEN_1` | 64 | MLP first hidden layer width |
| `HIDDEN_2` | 32 | MLP second hidden layer width |
| `GBM_NUM_LEAVES` | 63 | LightGBM num_leaves |
| `GBM_N_ESTIMATORS` | 500 | LightGBM trees (with early stopping) |

---

## How training works

### GBM (production — `train_gbm.py`)

1. **Data**: 5-year OHLCV parquets in `data/cache/` (one per ticker, plus SPY,
   VIX, sector ETFs, and `sector_map.json` generated by `bootstrap_fetcher.py`).
2. **Features**: `compute_features()` produces 21 rolling technical + macro
   indicators per row. 252-row warmup required for 52-week high/low features.
3. **Labels (sector-neutral)**: `compute_labels()` computes the 5-day forward
   return relative to the ticker's **sector ETF** (not SPY):
   `(close[T+5]/close[T] - 1) - (sector_etf[T+5]/sector_etf[T] - 1)`.
   Falls back to SPY-relative if sector ETF data is unavailable.
   Sector-neutral labels remove industry momentum the model cannot time,
   leaving only stock-specific alpha as the training target.
4. **Objective**: LightGBM regression (MSE). Evaluated on Pearson IC.
5. **Split**: 70/15/15 time-based split. Test set = most recent 15% of dates.
6. **Early stopping**: Validation IC monitored; training stops when IC stops
   improving. Feature importance logged to `checkpoints/gbm_feature_importance.csv`.
7. **Checkpoint**: `checkpoints/gbm_best.txt` (LightGBM booster text format).
8. **Production gate**: Test IC ≥ 0.04 required before uploading to S3.

### MLP (available — `train.py`)

Same data pipeline, but uses PyTorch `DateGroupedSampler` + `ICLoss` (negative
Pearson IC) with z-score normalization. See [docs/training.md](docs/training.md)
for full MLP theory and hyperparameter details.

---

## How inference works

1. **EventBridge** `ae-predictor-run` fires at 13:15 UTC (6:15am PDT / 5:15am PST)
   on weekdays, invoking `alpha-engine-predictor-inference`.
2. **GBM booster** loaded from
   `s3://alpha-engine-research/predictor/weights/gbm_latest.txt`.
3. **Watchlist** (`watchlist_path="auto"`) read from
   `s3://alpha-engine-research/signals/YYYY-MM-DD/signals.json` — only the
   research module's tracked tickers + buy candidates (~10–30 names) are
   predicted. Falls back to a hardcoded 15-ticker list if S3 read fails.
4. **OHLCV** fetched via yfinance (1-year window per ticker).
5. `compute_features()` produces 21 features per row; only the **last row**
   (today's snapshot) is used.
6. **GBM predict** → continuous alpha score → thresholded to
   {UP / FLAT / DOWN} + confidence.
7. **S3 write**: results to `predictor/predictions/YYYY-MM-DD.json` and
   `predictor/predictions/latest.json`. Metrics to `predictor/metrics/latest.json`.

Tickers with fewer than 205 rows (insufficient for MA200 feature) are skipped
and logged. Per-ticker failures do not abort the run.

---

## Evaluating model quality

Run evaluation on the held-out test set after training:

```bash
python train.py  # evaluation runs automatically at end
cat checkpoints/eval_report.json
```

**Production gates** (all must pass before enabling in research):

| Gate | Threshold |
|------|-----------|
| Test set hit rate | > 55% |
| Test set IC (Pearson) | > 0.05 |
| 30 days live hit rate | > 55% |

**IC IR gate** (before TFT upgrade):

| Gate | Threshold |
|------|-----------|
| Rolling 20-day IC IR | > 0.3 |

The predictor is shipped with `enabled: false` in `config/universe.yaml`. Flip
to `true` only after all production gates pass.

---

## Contributing / development setup

```bash
# Install with dev dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_features.py -v

# Type check (optional, requires mypy)
mypy config.py model/ data/ inference/

# Local inference dry run
python inference/daily_predict.py --local --dry-run --limit 10
```

### Adding new features

1. Add the feature name to `config.FEATURES` and increment `config.N_FEATURES`.
2. Implement the rolling computation in `data/feature_engineer.py::compute_features()`.
3. Mirror the snapshot computation in `alpha-engine-research/data/fetchers/price_fetcher.py::compute_technical_indicators()`.
4. Update `data/norm_stats.json` by re-running `train.py` (new stats are generated automatically).
5. Retrain the model — previous checkpoint is incompatible due to changed input dimension.

### Feature parity with research

The 8 features in this repo must always match the snapshot values produced by
`compute_technical_indicators()` in alpha-engine-research. Any divergence causes
training-inference skew (the model is trained on one feature scale and scored
on another). When modifying features, update both repos simultaneously.

---

## License

See [LICENSE](LICENSE).
