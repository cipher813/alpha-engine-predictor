"""
inference/daily_predict.py — Daily prediction run (slim entry point).

Called by the Lambda handler (inference/handler.py) and optionally from the
CLI for local testing. The actual pipeline logic lives in inference/stages/;
this file provides:
  - main() — builds PipelineContext and delegates to run_pipeline()
  - CLI arg parsing (--date, --dry-run, --local, --model-type, --watchlist, --offline)
  - Backward-compatible re-exports so existing tests and handler.py continue to work

Usage:
    # Full universe (~900 tickers, uses S3 signals.json):
    python inference/daily_predict.py [--date DATE] [--dry-run] [--local]

    # Focused watchlist mode:
    python inference/daily_predict.py --watchlist auto --model-type gbm [--dry-run]

    # Offline mode (synthetic data, no S3/API calls):
    python inference/daily_predict.py --offline --date 2026-04-02
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure the project root (parent of inference/) is on sys.path so that
# root-level modules (config, model/, data/) are importable when this script
# is invoked directly as `python inference/daily_predict.py` from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config as cfg

log = logging.getLogger(__name__)


# ── Backward-compatible re-exports ──────────────────────────────────────────
# These functions have moved to their natural stage modules. Re-exported here
# so that tests, handler.py, and offline mode monkey-patching continue to work.

from inference.stages.load_model import (  # noqa: E402, F401
    load_model, load_model_local, load_gbm_local, load_gbm_s3,
)
from inference.stages.load_universe import (  # noqa: E402, F401
    get_universe_tickers, load_watchlist,
)
from inference.stages.load_prices import (  # noqa: E402, F401
    _safe_last_date, fetch_today_prices, fetch_macro_series,
    save_daily_closes, load_price_data_from_cache,
)
from inference.stages.run_inference import predict_ticker  # noqa: E402, F401
from inference.stages.write_output import (  # noqa: E402, F401
    _load_predictor_params_from_s3, get_veto_threshold,
    write_predictions, _build_predictor_email, send_predictor_email,
)
from inference.s3_io import _s3_put_json, _s3_put_bytes  # noqa: E402, F401


# ── Orchestration ─────────────────────────────────────────────────────────────

def main(
    date_str: Optional[str] = None,
    dry_run: bool = False,
    local: bool = False,
    s3_bucket: Optional[str] = None,
    model_type: str = "mlp",
    watchlist_path: Optional[str] = None,
) -> None:
    """
    Run the full daily prediction pipeline.

    Parameters
    ----------
    date_str :       Override prediction date YYYY-MM-DD. Default: today.
    dry_run :        Skip S3 writes; print output to stdout.
    local :          Load model from local checkpoints/ instead of S3.
    s3_bucket :      Override S3 bucket. Falls back to S3_BUCKET env var or config default.
    model_type :     Which model to run: 'mlp' (default) or 'gbm'.
    watchlist_path : Path to watchlist.json. When provided, predictions are
                     restricted to the tickers in the universe.
                     When None, the full signals.json universe is used.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Soft timeout: flush partial predictions if nearing Lambda hard limit
    import time as _time
    _start_ts = _time.monotonic()
    _SOFT_TIMEOUT_S = int(os.environ.get("PREDICTOR_SOFT_TIMEOUT_S", "780"))  # 13 min default

    fd = None

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    bucket = s3_bucket or os.environ.get("S3_BUCKET", cfg.S3_BUCKET)

    log.info(
        "Daily predictor run: date=%s  bucket=%s  dry_run=%s  local=%s  "
        "model_type=%s  watchlist=%s",
        date_str, bucket, dry_run, local, model_type,
        watchlist_path or "full universe",
    )

    # Validate email config (warn early so operator knows before pipeline finishes)
    if not cfg.EMAIL_SENDER or not cfg.EMAIL_RECIPIENTS:
        log.warning(
            "Email not configured (EMAIL_SENDER=%r, EMAIL_RECIPIENTS has %d entries) "
            "— morning briefing will be skipped",
            cfg.EMAIL_SENDER or "(empty)", len(cfg.EMAIL_RECIPIENTS),
        )

    # ── Delegate to staged pipeline ──────────────────────────────────────────
    from inference.pipeline import PipelineContext, run_pipeline

    ctx = PipelineContext(
        date_str=date_str,
        bucket=bucket,
        dry_run=dry_run,
        local=local,
        model_type=model_type,
        watchlist_path=watchlist_path,
        fd=fd,
        start_ts=_start_ts,
        soft_timeout_s=_SOFT_TIMEOUT_S,
    )
    run_pipeline(ctx)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run daily direction predictions and write to S3."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override prediction date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip S3 writes; print predictions JSON to stdout.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Load model from local checkpoints/best.pt instead of S3.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help=f"Override S3 bucket. Default: {cfg.S3_BUCKET}",
    )
    parser.add_argument(
        "--model-type",
        default="gbm",
        choices=["mlp", "gbm"],
        help=(
            "Which model to load for inference: "
            "'gbm' loads GBM weights (or meta-model when META_MODEL_ENABLED=true), "
            "'mlp' loads legacy MLP checkpoints/best.pt. "
            "Default: gbm"
        ),
    )
    parser.add_argument(
        "--watchlist",
        default=None,
        metavar="auto|PATH",
        help=(
            "Restrict predictions to the research module's tracked + buy-candidate "
            "tickers instead of the full ~900-stock universe. "
            "'auto' reads today's signals/{date}/signals.json from S3. "
            "Any other value is treated as a local file path to a signals.json "
            "produced by alpha-engine-research (useful for offline / dry-run testing). "
            "Each prediction result gets a 'watchlist_source' annotation: "
            "'tracked' (universe tickers), 'buy_candidate' (scanner picks), or 'both'. "
            "Omit this flag to predict the full universe. "
            "Example: --watchlist auto   or   --watchlist /tmp/signals.json"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Full offline mode: synthetic prices, dummy GBM models, no S3/API calls. "
             "Tests feature computation, ranking, veto logic, and email formatting.",
    )
    args = parser.parse_args()

    if args.offline:
        # Import the actual stage modules so monkey-patches propagate correctly
        import inference.stages.load_model as _lm
        import inference.stages.load_universe as _lu
        import inference.stages.load_prices as _lp
        import inference.stages.write_output as _wo

        # Dummy GBM scorer: returns small random alpha scores
        class _DummyScorer:
            _best_iteration = 0
            _val_ic = 0.05
            _booster = None  # skip feature validation
            _feature_names = None
            def predict(self, X):
                rng = np.random.RandomState(42)
                return rng.uniform(-0.02, 0.02, size=X.shape[0])

        # Stub model loaders — patch stage module where they're actually called
        _lm.load_gbm_s3 = lambda *a, **k: _DummyScorer()
        _lm.load_gbm_local = lambda *a, **k: _DummyScorer()

        # Stub watchlist: 10 sample tickers
        _OFFLINE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                            "JPM", "JNJ", "XOM", "PFE", "HD"]
        _lu.get_universe_tickers = lambda *a, **k: (_OFFLINE_TICKERS, {})
        _lu.load_watchlist = lambda *a, **k: (
            _OFFLINE_TICKERS,
            {t: "population" for t in _OFFLINE_TICKERS},
            {"market_regime": "neutral"},
        )

        # Stub price data: synthetic 252-day OHLCV DataFrames
        _MACRO_SYMBOLS = ["SPY", "VIX", "TNX", "IRX", "GLD", "USO",
                          "XLK", "XLV", "XLF", "XLE", "XLY", "XLC"]
        def _make_synthetic_df(ticker, base, n=300):
            dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n)
            rng = np.random.RandomState(hash(ticker) % 2**31)
            returns = rng.normal(0.0005, 0.015, n)
            close = base * np.cumprod(1 + returns)
            return pd.DataFrame({
                "Open": close * 0.999, "High": close * 1.01,
                "Low": close * 0.99, "Close": close,
                "Volume": rng.randint(1_000_000, 10_000_000, n),
            }, index=dates)

        def _make_synthetic_prices(tickers, *a, **k):
            price_data = {}
            for i, t in enumerate(tickers):
                price_data[t] = _make_synthetic_df(t, 50 + i * 20)
            # Include macro symbols so feature computation has SPY, VIX, etc.
            macro = {}
            for sym in _MACRO_SYMBOLS:
                df = _make_synthetic_df(sym, 100)
                macro[sym] = df["Close"]
                if sym not in price_data:
                    price_data[sym] = df
            return price_data, macro

        _lp.load_price_data_from_cache = lambda tickers, *a, **k: _make_synthetic_prices(tickers)
        _lp.fetch_today_prices = lambda tickers, **k: _make_synthetic_prices(tickers)[0]
        _lp.save_daily_closes = lambda *a, **k: None
        _lp.fetch_macro_series = lambda *a, **k: {}

        # Stub output
        _wo.write_predictions = lambda *a, **k: log.info("[OFFLINE] Skipped S3 write")
        _wo.send_predictor_email = lambda *a, **k: log.info("[OFFLINE] Skipped email")
        _wo.get_veto_threshold = lambda *a, **k: 0.65

        # Stub data fetchers (imported inside stages from sub-modules)
        import data.earnings_fetcher as _ef
        _ef.fetch_earnings_data = lambda *a, **k: {t: {} for t in _OFFLINE_TICKERS}
        _ef.cache_earnings_to_s3 = lambda *a, **k: None
        _ef.fetch_revision_history = lambda *a, **k: {t: {} for t in _OFFLINE_TICKERS}

        import data.options_fetcher as _of
        _of.fetch_options_features = lambda *a, **k: {t: {} for t in _OFFLINE_TICKERS}
        _of.load_historical_options = lambda *a, **k: {}

        # Stub health write
        try:
            import health_status
            health_status.write_health = lambda *a, **k: None
        except Exception:
            pass

        log.info("OFFLINE MODE: synthetic data, dummy GBM, no S3/API calls")
        main(
            date_str=args.date or datetime.now().strftime("%Y-%m-%d"),
            dry_run=True,
            local=False,  # False so it doesn't try to load local checkpoints
            model_type="gbm",
        )
    else:
        main(
            date_str=args.date,
            dry_run=args.dry_run,
            local=args.local,
            s3_bucket=args.s3_bucket,
            model_type=args.model_type,
            watchlist_path=args.watchlist,
        )
