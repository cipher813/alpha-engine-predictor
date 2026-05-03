# alpha-engine-predictor — Code Index

> Index of entry points, key files, and data contracts. Companion to [README.md](README.md). System overview lives in [`alpha-engine-docs`](https://github.com/cipher813/alpha-engine-docs).

## Module purpose

Stacked meta-ensemble that predicts 5-day market-relative alpha — three Layer-1 specialized models feeding a Layer-2 Ridge meta-learner — with a veto gate consumed by the Executor.

## Entry points

| File | What it does |
|---|---|
| [`inference/handler.py`](inference/handler.py) | Daily inference Lambda handler — Saturday/weekday SF invokes this |
| [`inference/daily_predict.py`](inference/daily_predict.py) | Full prediction pipeline + morning briefing email |
| [`training/train_handler.py`](training/train_handler.py) | Weekly training entry — EC2 spot from Saturday SF |
| [`training/meta_trainer.py`](training/meta_trainer.py) | Trains the 3 L1 components + L2 Ridge with walk-forward validation |
| [`scripts/dry_run_meta_training.py`](scripts/dry_run_meta_training.py) | Local dry-run harness for training |

## Where things live

| Concept | File |
|---|---|
| Layer-2 Ridge meta-learner (META_FEATURES list of 12) | [`model/meta_model.py`](model/meta_model.py) |
| Layer-1 LightGBM wrapper (momentum + volatility) | [`model/gbm_scorer.py`](model/gbm_scorer.py) |
| Layer-1 research-score calibrator (lookup → GBM v1 on roadmap) | [`model/research_calibrator.py`](model/research_calibrator.py) |
| Per-component subsample-IC validator (promotion gate) | [`model/subsample_validator.py`](model/subsample_validator.py) |
| Calibration helpers | [`model/calibrator.py`](model/calibrator.py) |
| Research-feature extraction for L2 | [`model/research_features.py`](model/research_features.py) |
| Pruned regime predictor (kept for reference, not in active stack) | [`model/regime_predictor.py`](model/regime_predictor.py) |
| Inference pipeline orchestration | [`inference/pipeline.py`](inference/pipeline.py) |
| Inference preflight (S3 + ArcticDB freshness) | [`inference/preflight.py`](inference/preflight.py) |
| Inference S3 read/write helpers | [`inference/s3_io.py`](inference/s3_io.py) |
| Daily-prediction stages (load model · prices · alt data · run · write) | [`inference/stages/`](inference/stages/) |
| Coverage/clustering check (deploy-time drift gate) | [`inference/deploy_drift.py`](inference/deploy_drift.py) |
| Coverage check (per-ticker feature availability) | [`inference/coverage_check.py`](inference/coverage_check.py) |
| Training preflight | [`training/preflight.py`](training/preflight.py) |
| Dataset assembly (training arrays) | [`data/dataset.py`](data/dataset.py) |
| Bootstrap historical fetcher | [`data/bootstrap_fetcher.py`](data/bootstrap_fetcher.py) |
| Label generator (sector-neutral 5d alpha) | [`data/label_generator.py`](data/label_generator.py) |
| Earnings + options fetchers (alt data) | [`data/earnings_fetcher.py`](data/earnings_fetcher.py), [`data/options_fetcher.py`](data/options_fetcher.py) |
| Mode-comparison utilities | [`analysis/compare_modes.py`](analysis/compare_modes.py) |
| Health-status writer | [`health_status.py`](health_status.py) |
| SSM secret loader | [`ssm_secrets.py`](ssm_secrets.py) |

## Inputs / outputs

### Reads
| Source | Path |
|---|---|
| Watchlist (Research-tracked tickers) | `s3://alpha-engine-research/signals/{date}/signals.json` |
| Price universe (training) | `s3://alpha-engine-research/arcticdb/universe/` |
| Price slim cache (inference) | `s3://alpha-engine-research/arcticdb/universe_slim/` |
| Daily closes delta | `s3://alpha-engine-research/staging/daily_closes/{date}.parquet` |
| Engineered feature store | `s3://alpha-engine-research/predictor/feature_store/{date}/` |
| Veto threshold + auto-tuned params | `s3://alpha-engine-research/config/predictor_params.json` |

### Writes
| Destination | Path |
|---|---|
| Daily predictions | `s3://alpha-engine-research/predictor/predictions/{date}.json` + `latest.json` |
| Active ensemble weights | `s3://alpha-engine-research/predictor/weights/` |
| Training metrics + per-L1 IC + L2 IC | `s3://alpha-engine-research/predictor/metrics/latest.json` |
| Per-call LLM cost JSONLs (training/inference) | `s3://alpha-engine-research/decision_artifacts/_cost_raw/{date}/{run_id}/` |

## Run modes

| Mode | Where | Command |
|---|---|---|
| Production training | EC2 spot (c5.large) | Saturday Step Function |
| Production inference | Lambda (daily 6:07 AM PT) | Weekday Step Function |
| Local dry-run training | venv | `python scripts/dry_run_meta_training.py` |
| Local dry-run inference | venv | `python inference/daily_predict.py --local --dry-run` |
| Smoke test (model load + predict) | venv | `python scripts/smoke_meta_model_load.py` |

Deploy: `./infrastructure/deploy.sh main` builds Docker → ECR → updates Lambda alias `live`. Spot training image built and pushed alongside.

## Tests

`pytest tests/` covers meta-model schema invariants, NaN feature handling, subsample validator, meta-trainer streaming, bad-data handling, and inference pipeline shape. ~135 tests passing.
