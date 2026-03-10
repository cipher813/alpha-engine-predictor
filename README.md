# alpha-engine-predictor

A standalone PyTorch ML module that produces per-ticker 5-day return direction
predictions from technical indicators. It runs daily as an AWS Lambda function,
reads price data via yfinance, computes the same 8 features used by the
alpha-engine-research pipeline, and writes a `predictions.json` to S3. The
research pipeline optionally loads these predictions and uses them as a
confidence-gated modifier to per-ticker technical scores. The predictor is
deliberately separate from the research and executor modules — integration
happens through S3 artifacts, not shared code.

---

## Architecture

```
  ┌────────────────────────────────────────────────────┐
  │                 Monthly retraining                 │
  │  bootstrap_fetcher → feature_engineer → dataset    │
  │  → train.py → checkpoints/best.pt → S3             │
  └────────────────────────────────────────────────────┘
                            │
             s3://alpha-engine-research/predictor/weights/latest.pt
                            │
  ┌─────────────────────────▼──────────────────────────┐
  │              Daily inference (Lambda)              │
  │  EventBridge 6:15am PT                             │
  │  → load model from S3                              │
  │  → get universe from signals.json                  │
  │  → fetch 1y OHLCV (yfinance)                       │
  │  → compute 8 features (latest row)                 │
  │  → z-score normalize (checkpoint norm_stats)       │
  │  → MLP forward pass → softmax                      │
  │  → write predictions/latest.json to S3             │
  └────────────────────────────────────────────────────┘
                            │
             s3://alpha-engine-research/predictor/predictions/latest.json
                            │
  ┌─────────────────────────▼──────────────────────────┐
  │          alpha-engine-research integration         │
  │  fetch_data() reads latest.json (best-effort)      │
  │  run_universe_agents() merges predictor values      │
  │  into indicators dict → compute_technical_score()  │
  │  applies ±10pt confidence-gated modifier           │
  └────────────────────────────────────────────────────┘
```

**MLP v1 architecture:**
```
Input(8) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
         → Linear(32) → BatchNorm → ReLU → Dropout(0.2)
         → Linear(3)  → [P(DOWN), P(FLAT), P(UP)]
```

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

### Run inference locally (dry run)

```bash
python inference/daily_predict.py --local --dry-run
```

Prints the predictions JSON to stdout without writing to S3. Requires a
trained checkpoint at `checkpoints/best.pt`.

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

### Build and deploy Lambda (container image)

```bash
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1
./infrastructure/deploy.sh

# Dry run (build only, no AWS push):
./infrastructure/deploy.sh --dry-run
```

### Monthly retraining

```bash
# Local (development)
./infrastructure/retrain.sh --local

# SageMaker (production)
export SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole
./infrastructure/retrain.sh
```

---

## Directory structure

```
alpha-engine-predictor/
├── data/
│   ├── bootstrap_fetcher.py   # one-time 5y OHLCV fetch for ~900 tickers
│   ├── feature_engineer.py    # rolling 8-feature computation (mirrors research)
│   ├── label_generator.py     # 5-day forward return labels
│   └── dataset.py             # PyTorch Dataset + DataLoader construction
├── model/
│   ├── predictor.py           # DirectionPredictor MLP + checkpoint save/load
│   ├── trainer.py             # training loop, early stopping, LR scheduler
│   └── evaluator.py           # IC, hit rate, direction Sharpe, confusion matrix
├── inference/
│   ├── daily_predict.py       # daily prediction pipeline (CLI + programmatic)
│   └── handler.py             # AWS Lambda entry point
├── infrastructure/
│   ├── deploy.sh              # ECR build + push + Lambda update
│   └── retrain.sh             # SageMaker Training Job or local retrain
├── tests/
│   ├── test_features.py       # feature engineering unit tests
│   ├── test_model.py          # model forward pass + checkpoint tests
│   └── test_inference.py      # predict_ticker() unit tests
├── docs/
│   ├── architecture.md        # MLP architecture, TFT upgrade path
│   ├── training.md            # data pipeline, split strategy, hyperparameters
│   ├── inference.md           # daily workflow, output schema, confidence gate
│   └── deployment.md          # AWS setup, ECR, EventBridge, SageMaker
├── config.py                  # S3 paths, hyperparameters, feature list
├── train.py                   # root-level training entry point
├── requirements.txt
├── Dockerfile
└── alpha-engine-predictor-design-260310.md
```

---

## Key configuration

All constants are in `config.py`. Notable values:

| Constant | Default | Description |
|----------|---------|-------------|
| `S3_BUCKET` | `alpha-engine-research` | S3 bucket for weights and predictions |
| `FEATURES` | 8 features | Must stay in sync with research pipeline |
| `FORWARD_DAYS` | 5 | Prediction horizon (trading days) |
| `UP_THRESHOLD` | 0.01 | > +1% forward return = UP class |
| `DOWN_THRESHOLD` | -0.01 | < -1% forward return = DOWN class |
| `MIN_CONFIDENCE` | 0.65 | Below this, modifier not applied in research |
| `MIN_HIT_RATE` | 0.55 | Required for production deployment |
| `MIN_IC` | 0.05 | Required for production deployment |
| `TRAIN_FRAC` | 0.70 | Time-based training split |
| `HIDDEN_1` | 64 | First hidden layer width |
| `HIDDEN_2` | 32 | Second hidden layer width |

---

## How training works

1. **Data**: 5-year OHLCV parquets in `data/cache/` (one per ticker).
2. **Features**: `compute_features()` computes 8 rolling technical indicators per
   row. Rows lacking sufficient history (MA200 requires 200+ rows) are dropped.
3. **Labels**: `compute_labels()` computes `(close[t+5] / close[t]) - 1` and
   bins into DOWN / FLAT / UP classes.
4. **Split**: All samples sorted by date, split 70/15/15 at index boundaries
   (no shuffling across boundaries — the test set is always the most recent data).
5. **Normalization**: Z-score per feature, fit on training set only. Statistics
   saved to `data/norm_stats.json` and embedded in every checkpoint.
6. **Training**: AdamW optimizer, ReduceLROnPlateau scheduler, early stopping
   with patience=10 epochs, gradient clipping at max_norm=1.0.
7. **Checkpoint**: Best model by validation loss saved to `checkpoints/best.pt`.

---

## How inference works

1. Lambda is triggered by EventBridge at 6:15am PT on trading days.
2. Model weights loaded from `s3://alpha-engine-research/predictor/weights/latest.pt`.
3. Today's ticker universe read from `signals/signals_YYYYMMDD.json` (falls back
   to a hardcoded 15-ticker list if unavailable).
4. 1-year OHLCV fetched via yfinance for each ticker.
5. `compute_features()` runs on each ticker's DataFrame; only the **last row**
   (today's feature vector) is used.
6. Feature vector z-score normalized using `norm_stats` from the checkpoint.
7. `model(x) → softmax → [p_down, p_flat, p_up]`; argmax gives predicted direction.
8. Predictions written to S3 at dated path and `latest.json`.

Tickers with fewer than 205 rows of price history (insufficient for MA200) are
skipped and logged. Missing/failed tickers do not cause the Lambda to fail.

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
