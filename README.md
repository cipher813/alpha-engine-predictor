# alpha-engine-predictor

> Part of [**Nous Ergon**](https://nousergon.ai) — Autonomous Multi-Agent Trading System. Repo and S3 names use the underlying project name `alpha-engine`.

[![Part of Nous Ergon](https://img.shields.io/badge/Part_of-Nous_Ergon-1a73e8?style=flat-square)](https://nousergon.ai)
[![Python](https://img.shields.io/badge/python-3.12+-blue?style=flat-square)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-1a73e8?style=flat-square)](https://lightgbm.readthedocs.io/)
[![ArcticDB](https://img.shields.io/badge/ArcticDB-0e1117?style=flat-square)](https://docs.arcticdb.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Phase 2 · Reliability](https://img.shields.io/badge/Phase_2-Reliability-e9c46a?style=flat-square)](https://github.com/cipher813/alpha-engine-docs#phase-trajectory)

Stacked meta-ensemble that predicts 5-day market-relative alpha for each ticker. Three Layer-1 specialized models — LightGBM momentum, LightGBM volatility, and a research-score calibrator — feed a Layer-2 Ridge meta-learner alongside research-context and raw macro features. Outputs UP/FLAT/DOWN with a confidence score and a veto gate that blocks high-confidence DOWN entries.

> System overview, Step Function orchestration, and module relationships live in [`alpha-engine-docs`](https://github.com/cipher813/alpha-engine-docs). Code index lives in [`OVERVIEW.md`](OVERVIEW.md).

## What this does

- **Stacked meta-ensemble** — three Layer-1 specialized models (LightGBM momentum + LightGBM volatility + research-score calibrator) plus research-context and raw macro features feed a Layer-2 Ridge meta-learner. v3.0 architecture deployed 2026-04-01.
- **Sector-neutral training** — labels are `alpha = stock_5d_return − sector_etf_5d_return` with cross-sectional rank normalization and 5-fold walk-forward validation. Promotion gate requires both ensemble IC and per-component subsample IC to pass.
- **Daily inference** — loads ensemble weights from S3, fetches prices (slim cache + daily_closes delta), recomputes features from scratch each morning so predictions shift even when the model is frozen between weekly retrains.
- **Veto gate** — high-confidence DOWN predictions override BUY signals from Research, gating the Executor against entering declining positions. Threshold is auto-tuned weekly by the Backtester.
- **Named-baseline discipline** — every Layer-1 component must clear an explicit named baseline before contributing to L2; the prior 4th L1 (regime classifier) was pruned 2026-04-16 after walk-forward validation showed sub-majority-class accuracy.

## Phase 2 measurement contribution

Predictor outputs are the quantitative layer on top of Research signals. Phase 2 contribution: every prediction is captured with its full feature vector, per-component L1 votes, L2 meta-learner score, confidence, and veto-gate decision. Per-component IC tracking lets the system tell whether a regression in ensemble IC is driven by one underperforming layer or by a regime shift across all of them. The substrate that lets Phase 3 retire or upgrade individual L1 components without destabilizing the stack.

## Architecture

```mermaid
flowchart LR
    Prices[Prices · features<br/>ArcticDB universe] --> L1
    Research[Research signals<br/>composite + sub-scores] --> L1
    Macro[Macro context<br/>VIX · yields · breadth] --> L1

    subgraph L1[Layer 1 — specialized models]
        Mom[LightGBM<br/>momentum]
        Vol[LightGBM<br/>volatility]
        Cal[research-score<br/>calibrator]
    end

    L1 --> L2[Layer 2 — Ridge meta-learner<br/>combines L1 + research-context + raw macro]
    L2 --> Out[predictions/{date}.json<br/>UP / FLAT / DOWN<br/>+ confidence + veto]
```

Veto gate fires when DOWN confidence exceeds the auto-tuned threshold; Executor honors the veto on its next planning cycle.

## Configuration

This repo is **public**. Hyperparameters, confidence thresholds, and tuned ensemble weights live in the private [`alpha-engine-config`](https://github.com/cipher813/alpha-engine-config) repo and are auto-applied weekly by the Backtester via `s3://alpha-engine-research/config/predictor_params.json`. Architecture and approach are public; specific values are private.

## Sister repos

| Module | Repo |
|---|---|
| Executor | [`alpha-engine`](https://github.com/cipher813/alpha-engine) |
| Data | [`alpha-engine-data`](https://github.com/cipher813/alpha-engine-data) |
| Research | [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) |
| Backtester | [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) |
| Dashboard | [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) |
| Library | [`alpha-engine-lib`](https://github.com/cipher813/alpha-engine-lib) |
| Docs | [`alpha-engine-docs`](https://github.com/cipher813/alpha-engine-docs) |

## License

MIT — see [LICENSE](LICENSE).
