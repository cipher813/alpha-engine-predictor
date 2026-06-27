"""
config.py — Central configuration for alpha-engine-predictor.

All S3 paths, model hyperparameters, feature definitions, and production
gates live here. Import this module everywhere rather than hard-coding values.

Tunable parameters are loaded from config/predictor.yaml (gitignored).
Copy config/predictor.sample.yaml to get started.
"""

import logging
import os
from pathlib import Path

import yaml

from krepis.secrets import get_secret
# Canonical experiment-package config resolver (alpha-engine-config#1157): the
# lift of the five inline _find_config / load_config / config_loader copies into
# the shared-lib chokepoint. The predictor config search below delegates to it.
from nousergon_lib.config import resolve_experiment_config

_log = logging.getLogger(__name__)

# ── Load predictor config YAML ────────────────────────────────────────────────
_CONFIG_DIR = Path(__file__).parent / "config"

# Experiment package (HARNESS_EXPERIMENT_CLASSIFICATION.md §3): beliefs load
# from experiments/$ALPHA_ENGINE_EXPERIMENT_ID/predictor/ ahead of the legacy
# top-level predictor/ (kept as fallback through the transition).
#
# Delegates to the canonical nousergon-lib resolver (resolve_experiment_config,
# alpha-engine-config#1157). The repo-local fallback is the subdir-flattened
# <repo>/config/predictor.yaml (the staged-image location), passed as
# repo_local_fallback. _EXPERIMENT_ID is retained for the diagnostic logging
# that names the active slot (assert_model_specs_loaded). Divergence reconciled
# to the consensus order: the inline copy grouped candidates per-root
# (root1-exp, root1-legacy, root2-exp, root2-legacy); the lib uses the
# cross-repo-consensus per-layer order (all experiment-package, then all
# legacy) — the correct precedence the other four consumers already used, so a
# legacy copy under ~/ no longer shadows an experiment-package copy under
# <repo>/.. .
_EXPERIMENT_ID = os.environ.get("ALPHA_ENGINE_EXPERIMENT_ID", "reference")
_CONFIG_SEARCH = resolve_experiment_config(
    "predictor",
    "predictor.yaml",
    repo_root=Path(__file__).parent,
    repo_local_fallback=_CONFIG_DIR / "predictor.yaml",
    experiment_id=_EXPERIMENT_ID,
)
_CONFIG_PATH = next((p for p in _CONFIG_SEARCH if p.exists()), _CONFIG_DIR / "predictor.yaml")


def _load() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Predictor config not found. Searched:\n"
            + "\n".join(f"  {p}" for p in _CONFIG_SEARCH)
        )
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load()

# ── Phase-registry watchdog caps (L4528) ──────────────────────────────────────
# Per-phase hard caps (seconds) for the training spot's phase_registry watchdog.
# A phase exceeding its cap trips an all-thread stack dump + PhaseTimeoutError —
# converting the OOM-prone meta_training's opaque SSM ``TimedOut`` (L4511) into a
# clean, attributable failure. Absent / empty → watchdog off (no behavior
# change). The live cap value lands in alpha-engine-config/predictor/predictor.yaml.
FULL_RUN_HARD_CAPS_SECONDS = _cfg.get("full_run_hard_caps_seconds", {}) or {}

# ── S3 paths ─────────────────────────────────────────────────────────────────
S3_BUCKET = "alpha-engine-research"

MODEL_WEIGHTS_KEY = "predictor/weights/latest.pt"
MODEL_WEIGHTS_DATED_KEY = "predictor/weights/{date}.pt"

# GBM_WEIGHTS_META_KEY remains live — ``inference/stages/write_output.py::
# _load_gbm_meta`` reads it best-effort to populate ``last_trained`` in the
# inference metrics payload. The v2 ``GBM_*_WEIGHTS_KEY`` siblings were
# deleted in the v2 cleanup PR (2026-04-27) along with the v2 inference
# path that consumed them.
GBM_WEIGHTS_META_KEY = "predictor/weights/gbm_latest.txt.meta.json"

# Calibrator weights (isotonic regression on v3 meta-model output).
# Path is contractual with alpha-engine-backtester retrain_alert grace
# gate — it reads isotonic_calibrator.meta.json for deployed_at. The v2
# GBM-era path (predictor/weights/calibrator_latest.pkl) was retired
# 2026-04-15 with the binary UP/DOWN migration.
CALIBRATOR_WEIGHTS_KEY      = "predictor/weights/meta/isotonic_calibrator.pkl"
CALIBRATOR_WEIGHTS_META_KEY = "predictor/weights/meta/isotonic_calibrator.pkl.meta.json"

PREDICTIONS_KEY = "predictor/predictions/{date}.json"
PREDICTIONS_LATEST_KEY = "predictor/predictions/latest.json"

METRICS_KEY = "predictor/metrics/latest.json"

# ``PRICE_CACHE_KEY`` (was ``"predictor/price_cache/{ticker}.parquet"``) was
# deleted 2026-05-19 — Wave-3 PR3-wave-2 audit found zero references across
# the predictor tree (`git grep PRICE_CACHE_KEY` returned only this site).
# The active per-prefix constants live with their readers — see
# ``regime/features.DEFAULT_PRICE_CACHE_PREFIX`` + the Wave-3 fallback chain
# added in #181. Keeping a stale module-level constant invites a future
# `from config import PRICE_CACHE_KEY` that bypasses the read-prefix chain.

# ── Features ──────────────────────────────────────────────────────────────────
# Must stay in sync with data/feature_engineer.py::compute_features().
FEATURES = [
    "rsi_14",
    "macd_cross",
    "macd_above_zero",
    "macd_line_last",
    "price_vs_ma50",
    "price_vs_ma200",
    "momentum_20d",
    "avg_volume_20d",
    # v1.1 additions
    "dist_from_52w_high",
    "momentum_5d",
    "rel_volume_ratio",
    "return_vs_spy_5d",
    # v1.2 additions — market context features
    "vix_level",
    "dist_from_52w_low",
    "vol_ratio_10_60",
    "bollinger_pct",
    "sector_vs_spy_5d",
    "sector_vs_spy_10d",
    "sector_vs_spy_20d",
    # v1.3 additions — macro regime features
    "yield_10y",
    "yield_curve_slope",
    "gold_mom_5d",
    "oil_mom_5d",
    # v1.6 additions — investigation upgrades (A2)
    "vix_term_slope",
    "xsect_dispersion",
    # v1.4 additions — design doc Appendix A feature completions
    "price_accel",
    "ema_cross_8_21",
    "atr_14_pct",
    "realized_vol_20d",
    "volume_trend",
    "obv_slope_10d",
    "rsi_slope_5d",
    "volume_price_div",
    # v1.5 additions — regime interaction terms (cross-sectional: macro × ticker signal)
    "mom5d_x_vix",
    "rsi_x_vix",
    "sector_x_trend",
    "atr_x_vix",
    "vol_trend_x_vix",
    # v2.0 additions — alternative data signals (O10-O12)
    "earnings_surprise_pct",    # O10: PEAD — magnitude of most recent quarterly surprise
    "days_since_earnings",      # O10: PEAD — recency of last earnings (0-1, capped at 90d)
    "eps_revision_4w",          # O11: 4-week cumulative EPS revision percentage
    "revision_streak",          # O11: consecutive weeks of same-direction revisions
    "put_call_ratio",           # O12: log-transformed put/call OI ratio
    "iv_rank",                  # O12: IV percentile rank (0-1)
    "iv_vs_rv",                 # O12: implied vol / realized vol ratio
    # v3.0 additions — fundamental ratios (quarterly, from FMP)
    "pe_ratio",                  # P/E / 30, normalized
    "pb_ratio",                  # P/B / 5, normalized
    "debt_to_equity",            # D/E / 2, normalized
    "revenue_growth_yoy",        # YoY revenue growth (decimal)
    "fcf_yield",                 # Free cash flow / market cap
    "gross_margin",              # Gross profit / revenue (0-1)
    "roe",                       # Return on equity (decimal)
    "current_ratio",             # Current ratio / 3, normalized
]
N_FEATURES = 53

# Macro features — identical across all tickers on a given day, cannot predict
# cross-sectional alpha.  Excluded from GBM training/inference but kept in
# FEATURES for other callers (Research module, backtester technical scoring).
MACRO_FEATURES = {
    "vix_level", "yield_10y", "yield_curve_slope", "gold_mom_5d", "oil_mom_5d",
    "vix_term_slope",      # market-wide: identical across tickers → constant after rank norm
    "xsect_dispersion",    # market-wide: identical across tickers → constant after rank norm
}

# Fundamental features — computed and stored in feature store but excluded from
# GBM training/inference until backtester A/B validates their contribution.
# Move to GBM_FEATURES once validated with sufficient history.
_FUNDAMENTAL_EXCLUDE = {
    "pe_ratio", "pb_ratio", "debt_to_equity", "revenue_growth_yoy",
    "fcf_yield", "gross_margin", "roe", "current_ratio",
}

GBM_FEATURES = [f for f in FEATURES if f not in MACRO_FEATURES and f not in _FUNDAMENTAL_EXCLUDE]
N_GBM_FEATURES = len(GBM_FEATURES)  # 38 (31 technical + 7 alternative)

# ── Model architecture hyperparameters ───────────────────────────────────────
_model_cfg = _cfg["model"]
HIDDEN_1 = _model_cfg["hidden_1"]
HIDDEN_2 = _model_cfg["hidden_2"]
DROPOUT_1 = _model_cfg["dropout_1"]
DROPOUT_2 = _model_cfg["dropout_2"]

# ── Training hyperparameters ─────────────────────────────────────────────────
_train_cfg = _cfg["training"]
BATCH_SIZE = _train_cfg["batch_size"]
LEARNING_RATE = float(_train_cfg["learning_rate"])
WEIGHT_DECAY = float(_train_cfg["weight_decay"])
MAX_EPOCHS = _train_cfg["max_epochs"]
EARLY_STOPPING_PATIENCE = _train_cfg["early_stopping_patience"]
SCHEDULER_FACTOR = _train_cfg["scheduler_factor"]
SCHEDULER_PATIENCE = _train_cfg["scheduler_patience"]
MIN_LR = float(_train_cfg["min_lr"])
GRAD_CLIP_NORM = _train_cfg["grad_clip_norm"]
# ROADMAP L1581 — optional cutoff for the meta-trainer's data load.
# None (default) keeps all available history; an ISO date string
# ("YYYY-MM-DD") drops rows dated before that boundary. Diagnostic-only:
# set in alpha-engine-config for a single Sat retrain to compare meta IC
# against the no-clamp run, then revert. See ROADMAP entry for the
# 2020-03-31 vs ArcticDB-full-history hypothesis.
TRAIN_START_DATE: str | None = _train_cfg.get("train_start_date")

# ── Label thresholds ─────────────────────────────────────────────────────────
_label_cfg = _cfg["labeling"]
FORWARD_DAYS = _label_cfg["forward_days"]
UP_THRESHOLD = _label_cfg["up_threshold"]
DOWN_THRESHOLD = _label_cfg["down_threshold"]
LABEL_CLIP = _label_cfg["label_clip"]
ADAPTIVE_THRESHOLDS = _label_cfg.get("adaptive_thresholds", False)
ADAPTIVE_WINDOW = _label_cfg.get("adaptive_window", 63)
ADAPTIVE_UP_PCT = _label_cfg.get("adaptive_up_pct", 65)
ADAPTIVE_DOWN_PCT = _label_cfg.get("adaptive_down_pct", 35)

# ── Triple-barrier alpha labels (Stage 3, parallel-observe) ─────────────────
# LdP Ch. 3.4 — vol-scaled barriers. PR 1 of the Stage 3 arc ships the label
# generator + config knobs only; PR 2 wires parallel L1 training; PR 5 flips
# `enforce_cutover` to consume triple-barrier labels as the canonical L1
# alpha target. Plan: alpha-engine-docs/private/triple-barrier-260510.md.
_tb_cfg = _cfg.get("triple_barrier", {})
TRIPLE_BARRIER_FORWARD_WINDOW = _tb_cfg.get("forward_window", 21)
TRIPLE_BARRIER_VOL_WINDOW = _tb_cfg.get("vol_window", 20)
TRIPLE_BARRIER_VOL_MULTIPLIER = _tb_cfg.get("vol_multiplier", 2.0)
TRIPLE_BARRIER_MIN_PERIODS = _tb_cfg.get("min_periods", 10)
TRIPLE_BARRIER_ENFORCE_CUTOVER = _tb_cfg.get("enforce_cutover", False)

# ── GBM tuned hyperparameters ────────────────────────────────────────────────
_gbm_cfg = _cfg["gbm"]
GBM_N_ESTIMATORS = _gbm_cfg["n_estimators"]
GBM_EARLY_STOPPING_ROUNDS = _gbm_cfg["early_stopping_rounds"]
GBM_IC_IR_GATE = _gbm_cfg["ic_ir_gate"]
GBM_TUNED_PARAMS = _gbm_cfg["tuned_params"]

# Momentum base model — the shared GBM params above are tuned for the strong
# volatility signal. On the weak momentum target they overfit; the momentum
# base needs its own capacity config. Real values live in the private
# alpha-engine-config/predictor/predictor.yaml under gbm.momentum. Backtester
# can further override via s3://{bucket}/config/predictor_momentum_params.json.
# Missing section hard-fails — we never silently fall back to shared params.
_gbm_mom_cfg = _gbm_cfg.get("momentum")
if not _gbm_mom_cfg or "tuned_params" not in _gbm_mom_cfg:
    raise RuntimeError(
        "predictor.yaml is missing the required gbm.momentum section. "
        "Production values live in alpha-engine-config/predictor/predictor.yaml. "
        "See config/predictor.sample.yaml for the required schema."
    )
MOMENTUM_GBM_N_ESTIMATORS = _gbm_mom_cfg["n_estimators"]
MOMENTUM_GBM_EARLY_STOPPING_ROUNDS = _gbm_mom_cfg["early_stopping_rounds"]
MOMENTUM_GBM_TUNED_PARAMS = {**GBM_TUNED_PARAMS, **_gbm_mom_cfg["tuned_params"]}

# CatBoost ensemble config retired 2026-04-27 alongside the v2 inference
# path — its only consumer was ``_load_gbm`` (deleted) and the LGB-Cat
# blend mode that ``_load_meta_models`` does not implement.

# ── Production gates ─────────────────────────────────────────────────────────
_gates_cfg = _cfg["gates"]
MIN_HIT_RATE = _gates_cfg["min_hit_rate"]
MIN_IC = _gates_cfg["min_ic"]
MIN_CONFIDENCE = _gates_cfg["min_confidence"]
# Momentum veto threshold for the per-ticker ``momentum_veto`` flag on
# predictions.json. A ticker whose 20-day raw return is below this
# threshold gets ``momentum_veto=True``, signaling the executor to skip
# the ENTER (falling-knife guard). Moved from executor's inline
# momentum_gate (2026-05-11 refactor) to consolidate signal-side rules
# on the predictor side and free the executor to focus on risk rules.
# Default -0.05 (-5% over 20 days) matches the prior executor default.
# ``.get(..., -0.05)`` fallback lets this ship before predictor.yaml is
# updated in alpha-engine-config.
MOMENTUM_VETO_THRESHOLD = _gates_cfg.get("momentum_veto_threshold", -0.05)

# ── Training split (by time, no lookahead) ───────────────────────────────────
_split_cfg = _cfg["split"]
TRAIN_FRAC = _split_cfg["train_frac"]
VAL_FRAC = _split_cfg["val_frac"]

# ── Data fetching ────────────────────────────────────────────────────────────
_data_cfg = _cfg["data"]
REFRESH_BATCH_SIZE = _data_cfg["refresh_batch_size"]
BOOTSTRAP_PERIOD = _data_cfg["bootstrap_period"]
INFERENCE_PERIOD = _data_cfg["inference_period"]
DAILY_CLOSES_PERIOD = _data_cfg["daily_closes_period"]
# STALENESS_THRESHOLD_DAYS retired 2026-05-24 — the knob was parsed but
# never read in the predictor (audit during the trading-day freshness
# arc). Freshness in the predictor is enforced by `_verify_arctic_fresh`
# in `inference/stages/load_prices.py`, which uses
# `krepis.dates.is_fresh_in_trading_days(max_stale=0)` — no
# calendar-day knob to tune. The yaml key remains in `predictor.yaml`
# for backwards-compat with other config readers but is unused here.
SLIM_CACHE_LOOKBACK_DAYS = _data_cfg["slim_cache_lookback_days"]
INFERENCE_BATCH_SIZE = _data_cfg["inference_batch_size"]
MIN_ROWS_FOR_FEATURES = _data_cfg["min_rows_for_features"]
SPLIT_RETURN_THRESHOLD = _data_cfg["split_return_threshold"]

# ── Walk-forward validation ────────────────────────────────────────────────
_wf_cfg = _cfg.get("walk_forward", {})
WF_ENABLED = _wf_cfg.get("enabled", False)
# Renamed from test_window_days → test_window_trading_days; fallback to old key
WF_TEST_WINDOW_DAYS = _wf_cfg.get(
    "test_window_trading_days", _wf_cfg.get("test_window_days", 126)
)
WF_MIN_TRAIN_DAYS = _wf_cfg.get("min_train_days", 504)
WF_PURGE_DAYS = _wf_cfg.get("purge_days", 5)
# W1.1b / L4488a (ROADMAP): post-test embargo (López de Prado). Trading-day
# count dropped from training AFTER each test block. A no-op in the single-path
# expanding-forward walk-forward (train is always before test) but LOAD-BEARING
# in CPCV (W1.2), which interleaves test groups → train rows sit just AFTER an
# interior test block and leak via OVERLAPPING LABELS (a train row at b+k has a
# forward label that overlaps the test's late-obs labels for k ≤ horizon).
# Default is now **None = auto = the label horizon** (forward_days / the
# per-horizon h) — the correct overlapping-label embargo that makes the CPCV
# exclusion symmetric. (The old default 0 left that boundary open, inflating the
# leak-free reads — more at longer horizons, a confound for the 60d-vs-21d call;
# the prior "~0.01·T" target was only the residual serial-correlation buffer,
# undersized for overlapping labels.) An explicit integer in predictor.yaml
# still overrides (e.g. 0 to reproduce the legacy leaky behavior).
WF_EMBARGO_DAYS = _wf_cfg.get("embargo_days", None)
# W1.2 (ROADMAP L4469): combinatorial purged CV (López de Prado Ch. 12). The
# meta rows' unique dates split into WF_CPCV_N_GROUPS contiguous groups; every
# combination of WF_CPCV_K_TEST test groups yields one purged+embargoed OOS fit
# → a DISTRIBUTION of cross-sectional ICs (vs the single-path WF). The
# distribution feeds the W1.3 DSR/PBO promotion gate.
# L4565c: default raised 6 → 10. At the 21d horizon with ~45 OOS meta-dates,
# each test combo's 21d purge + 21d overlapping-label embargo consumes ~42
# dates, so at n_groups=6 only ~4 combos leave a usable train block — BELOW the
# downside/overfit battery's n≥5 floor, so the gate reported `insufficient` and
# select_winner could never promote. More groups → more (purged+embargoed —
# leak protection UNCHANGED) combinations clear the train-size guard (validated:
# 6→~4, 10→~12 valid combos on a 45-date shape), lifting the distribution over
# the n≥5 floor. The combos remain edge-dominated until more dates accumulate.
WF_CPCV_N_GROUPS = _wf_cfg.get("cpcv_n_groups", 10)
WF_CPCV_K_TEST = _wf_cfg.get("cpcv_k_test", 2)
# W1.3 (ROADMAP L4469): Deflated Sharpe Ratio gate params (Bailey-LdP). The
# Deflated Sharpe deflates the IC-IR's significance for the number of trials
# searched (anti-false-discovery). WF_DSR_N_TRIALS is a conservative placeholder
# for the effective config-search count until cumulative cross-week trial-count
# tracking lands (W1.3b); WF_DSR_THRESHOLD is the DSR a model must clear once
# W1.4 makes it gating. OBSERVE only today (reported, not gating).
WF_DSR_N_TRIALS = _wf_cfg.get("dsr_n_trials", 10)
WF_DSR_THRESHOLD = _wf_cfg.get("dsr_threshold", 0.95)
# L4582 two-threshold discipline: the REGISTRY-entry bar sits BELOW the
# promotion bar so near-misses are labeled in the zoo leaderboard and
# accumulate evidence across rotations instead of vanishing as a flat
# "gate_failed". Entry bar = downside gate passes AND dsr >= this; promotion
# bar = downside AND dsr >= WF_DSR_THRESHOLD (unchanged).
WF_DSR_REGISTRY_THRESHOLD = _wf_cfg.get("dsr_registry_threshold", 0.80)
# W1.3 (ROADMAP L4469): downside-aware IC performance lens — Sortino-of-IC
# threshold (skilled-risk basket: Sortino + CVaR + maxDD, anchor-gates-on-
# skilled-risk-not-sharpe). The predictor judges skill on DOWNSIDE risk, not
# symmetric volatility; Sortino-of-IC = mean(IC)/downside-deviation(IC). OBSERVE
# only today; W1.4 sets the real gating bar (composed with the DSR significance
# leg). 0.0 = informational placeholder.
WF_SORTINO_IC_THRESHOLD = _wf_cfg.get("sortino_ic_threshold", 0.0)
WF_MIN_FOLDS_POSITIVE = _wf_cfg.get("min_folds_positive", 0.60)
WF_MEDIAN_IC_GATE = _wf_cfg.get("median_ic_gate", 0.02)

# Audit Phase 2a output-distribution gate (2026-05-07). When True, a
# calibrator-output-shape check is added to the promotion gate alongside
# the existing meta-IC and per-component subsample gates. Default False
# for one Saturday-SF observation cycle so we measure baseline pass rate
# before promoting the gate to blocking. Flip to True via predictor.yaml's
# walk_forward.output_distribution_gate_blocking once a baseline run
# confirms the gate doesn't false-block on healthy models.
OUTPUT_DISTRIBUTION_GATE_BLOCKING = _wf_cfg.get(
    "output_distribution_gate_blocking", False
)

# Audit Phase 2a-INFER inference-time variant of the output-distribution
# gate (2026-05-07). When True, write_predictions raises before any S3
# write if the live batch's p_up distribution fails the same four
# invariants that the promotion gate checks on synthetic input.
# Fail-closed semantics: failure → exception → SF Catch → flow-doctor
# alert → predictions.json NOT written → executor reads prior-day
# fallback. Default False for one observation cycle; flip via
# predictor.yaml's walk_forward.output_distribution_gate_inference_blocking
# after the metrics_out forensic trail confirms no false-blocks on
# healthy live batches.
OUTPUT_DISTRIBUTION_GATE_INFERENCE_BLOCKING = _wf_cfg.get(
    "output_distribution_gate_inference_blocking", False
)

WF_N_ESTIMATORS = _wf_cfg.get("wf_n_estimators", None)  # None → use GBM_N_ESTIMATORS
WF_EARLY_STOPPING = _wf_cfg.get("wf_early_stopping", None)  # None → use GBM_EARLY_STOPPING_ROUNDS

# ── Residual / idiosyncratic-momentum L1 (W2, ROADMAP L4469) ────────────────
# OBSERVE-mode component reviving the dead raw-momentum L1 (WF median IC
# ≈ -0.001) with residual/idiosyncratic momentum + vol scaling (Blitz/Hanauer)
# plus a price-trend decomposition. ``enabled`` is the gate: when False
# (default) the L1 is trained + measured by the leak-free machinery + emitted
# to manifest.json, but its ``residual_momentum_score`` column is NOT added to
# META_FEATURES — the live ensemble + inference path are byte-identical. Flip
# to True (off-cycle / Saturday) only AFTER 2-3 observe firings confirm a
# positive purged-CV IC (composes with W1.4). ``.get`` with defaults so the
# code ships before alpha-engine-config's predictor.yaml carries the section.
_resid_mom_cfg = _cfg.get("residual_momentum", {})
RESIDUAL_MOMENTUM_ENABLED = _resid_mom_cfg.get("enabled", False)

# ── Raw-momentum L1 → meta-L2 inclusion gate (W4.2, L4469) ──────────────────
# Whether the raw-momentum L1's ``momentum_score`` column feeds the meta-L2
# input vector. The 2026-06-06 manifest shows ``momentum_score`` is a DEAD L1
# input — standalone leak-free xsec-IC ≈ 0.025 and an L2 coefficient of
# ≈ -0.045 (the BayesianRidge has already shrunk it to ~0) — while the
# residual-momentum L1 it is meant to replace scores xsec-IC ≈ 0.18 and clears
# both the downside (Sortino 3.33) and overfit (DSR 1.0) gates. Setting this
# False is the meta-L2 half of the residual-for-raw momentum SWAP: it drops
# ``momentum_score`` from the meta feature vector ONLY. The momentum L1 itself
# still trains and still emits ``momentum_score`` into predictions.json, so the
# executor's momentum-veto / reversal-confirmation gate (which read that field)
# are unaffected. Default True ⇒ the live meta feature set + persisted
# ``meta_model._feature_names`` + inference path are byte-identical to today;
# the swap is exercised only by the model-zoo ``residual-momentum`` spec
# (observe-first, CPCV-scored, never auto-promoted until the cutover flag).
# ``.get`` with a default so the code ships before predictor.yaml carries it.
_meta_stack_cfg = _cfg.get("meta_stack", {})
MOMENTUM_L1_IN_META = _meta_stack_cfg.get("momentum_l1_in_meta", True)

# ── L4565 SOTA directional-combine levers (model-zoo gated, observe-first) ──
# EXPECTED_MOVE_IN_META: keep the volatility L1's ``expected_move`` in the L2
# directional meta-vector. Default True ⇒ byte-identical to today. The
# ``sota-directional-combine`` zoo spec flips it False — ``expected_move`` is a
# directionless MAGNITUDE (E[|move|]); fed as a positive additive directional
# term it makes the highest-vol name (COIN: ~6σ) the #1 alpha pick (L4549b root
# cause). With it off the vol L1 still trains + emits the field for sizing /
# confidence / the barrier model — it just stops driving DIRECTIONAL alpha.
EXPECTED_MOVE_IN_META = _meta_stack_cfg.get("expected_move_in_meta", True)
# RESEARCH_FEATURES_IN_META: keep the four research-derived columns (research
# score / conviction / sector + calibrator) in the L2 meta-vector. Default
# True ⇒ byte-identical to today. A long-horizon model-zoo spec (horizon-60d/
# 90d) flips it False: its label window predates the short signals.json
# history, so the research-signal join would drop every row. With it off the
# join is skipped and those columns leave the meta vector, letting the spec
# train on the years-deep price/macro features alone (validate-only —
# select_winner's horizon filter keeps a longer-horizon model out of the live
# 21d serving slot).
RESEARCH_FEATURES_IN_META = _meta_stack_cfg.get("research_features_in_meta", True)
# RESEARCH_FREE_MAX_OOS_ROWS: recency cap on the OOS meta-training rows when a
# spec is research-free (RESEARCH_FEATURES_IN_META=False). Research-free specs
# keep snapshot-less rows, which over the full walk-forward is ~1.7M rows and
# OOMs the meta_training CPCV on the spot. The cap (recency-preserving deque)
# bounds memory; the deep history is still learned by the L1 GBMs. No effect on
# research-USING specs (snapshot-limited to ~1.3K rows, far under the cap).
RESEARCH_FREE_MAX_OOS_ROWS = int(_meta_stack_cfg.get("research_free_max_oos_rows", 150000))
# META_STANDARDIZE_ENABLED: standardize (z-score) + winsorize the DIRECTIONAL
# meta-features before the Ridge (Asness-Moskowitz-Pedersen signal-level
# combine + outlier guard). Default False ⇒ raw features, byte-identical. The
# scaler is fit at training time and persisted with the model so inference
# applies the identical transform. ``.get`` defaults so code ships before yaml.
META_STANDARDIZE_ENABLED = _meta_stack_cfg.get("meta_standardize_enabled", False)

# ── Cross-sectional level-neutralization of predicted_alpha (L4487) ─────────
# The meta-L2's macro features are common-mode across the daily cross-section,
# so an adverse-macro day shifts the ENTIRE predicted-alpha vector negative →
# the optimizer (which anchors at SPY=0) flushes the book + gbm_veto + the
# distribution-gate skew the same way. Canonical alpha is market-relative, so
# the cross-section mean SHOULD be ~0; centering removes the common-mode so the
# real cross-sectional skill drives direction/veto/allocation. When False
# (default) the inference path is byte-identical and the level_neutralization
# observe block in predictions.json/metrics.json just records what centering
# WOULD do (mean removed, direction flips, skew before/after). Flip to True
# (via predictor.yaml) only AFTER 2-3 firings confirm a consistent common-mode
# mean + skew-correction (observe-before-cutover). Producer-side single source
# of truth — the executor optimizer + gbm_veto inherit the centered value.
def _flag_env_or_yaml(env_name: str, yaml_value: bool) -> bool:
    """Boolean flag with an env-var override falling back to the predictor.yaml
    value. An explicit env value (1/true/yes/on or 0/false/no/off) wins;
    unset/blank → the yaml/default. Lets an operator flip a gate via the
    Lambda's environment without rebuilding the container image.

    ⚠️ L4497 — env-flag flips on an ALIAS-PINNED Lambda are NOT a bare
    `update-function-configuration`. `alpha-engine-predictor-inference` is
    invoked at the `live` alias pinned to a PUBLISHED version, and a published
    version's env is FROZEN — `update-function-configuration` only mutates
    `$LATEST`, so it is a SILENT no-op on the alias. The correct (and revert)
    procedure is `update-function-configuration ($LATEST) → publish-version →
    update-alias live→<new version>` (revert = `update-alias live→<prior
    version>`). "No redeploy" means "no container rebuild," NOT "env change
    takes effect on the alias immediately." See OBSERVATION_REGISTRY
    `predictor_xsec_level_neutralization` + ROADMAP L4497."""
    raw = os.environ.get(env_name)
    if raw is not None and raw.strip() != "":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(yaml_value)


_level_neut_cfg = _cfg.get("level_neutralization", {})
# Env-overridable (L4469): flip the level fix via the inference Lambda's
# XSEC_DEMEAN_ALPHA_ENABLED env var (no container rebuild) once the 6/3 canary
# observe-block confirms the common-mode + skew correction. FLIPPED 2026-06-03
# (live alias → v234). NB (L4497): the `live` alias is version-pinned, so the
# flip required update-function-configuration → publish-version → update-alias
# live→234 (a bare update-function-configuration is a silent no-op); revert =
# update-alias live→233. See _flag_env_or_yaml above.
XSEC_DEMEAN_ALPHA_ENABLED = _flag_env_or_yaml(
    "XSEC_DEMEAN_ALPHA_ENABLED", _level_neut_cfg.get("enabled", False)
)

# ── Challenger-first promotion (L4469; UNCONDITIONAL since config#1052/#679) ─
# Training is ALWAYS challenger-first: a run that PASSES the promotion gate is
# registered as a CHALLENGER (shadow + scored) and NEVER overwrites the live
# champion. Promotion is decided SOLELY by the relative-best model-zoo
# `select_winner` step (`training/model_zoo.py`), which copies the winner's
# bundle live via `model.registry.promote_to_champion`. The former
# `TRAINING_AUTO_PROMOTE_ENABLED` flag (dead — always False in production) was
# retired here: there is no training-time auto-promote path anymore. This closes
# the auto-ship-a-broken-model hole structurally (the 5/30 8-model auto-promoted
# on an inflated in-sample IC and flushed the book to SPY).

# ── Model zoo (L4488c) ──────────────────────────────────────────────────────
# Declarative variant specs for champion/challenger rotation. Each spec is a
# config OVERLAY over the existing training knobs — the model zoo runs a spec
# via `python -m training.model_zoo --spec <id>` (challenger-first, so it
# registers a challenger, never overwrites the champion). MODEL_VERSION_LABEL
# is the per-run label the spec sets so each variant gets its own registry
# version_id ({label}-{date}-{fingerprint}); default = the base label. See
# training/model_zoo.py for the override allowlist + the runtime override is
# applied around the train call, not persisted.
MODEL_SPECS = _cfg.get("model_specs", []) or []
MODEL_VERSION_LABEL = _cfg.get("model_version_label", "v3.0-meta")
# L4488g: weekly model-zoo ROTATION budget — how many challenger specs the
# rotation scheduler (`model_zoo --weekly-rotation`) trains per run. Bounds
# Saturday compute (each spec is ~one full train); the scheduler picks the N
# STALEST active specs (oldest/absent registered version) so the whole zoo is
# refreshed round-robin over ~ceil(len(active)/N) weeks. The champion retrain
# is separate + always-on; this budget is the challenger zoo on top of it.
MODEL_ZOO_WEEKLY_BUDGET = int(_cfg.get("model_zoo_weekly_budget", 4))

# config#1051 (no-silent-fails): an EMPTY ``model_specs`` roster while the
# rotation budget is >= 1 is never intentional on the rotation box — it means
# the staged ``predictor.yaml`` failed to parse / the experiment-package search
# (``_CONFIG_PATH`` above) resolved to the wrong / a spec-less file. The 6/13
# rotation trained 0 challengers SILENTLY for exactly this reason (MODEL_SPECS
# empty at runtime → ``select_rotation_specs`` returned []). We WARN at import
# (every config consumer surfaces it in logs) and expose a hard guard the
# rotation entrypoint calls before training so the inert state RAISES there
# rather than degrading to a benign "no eligible challenger" INFO. Import stays
# non-fatal: inference Lambdas and most tests legitimately don't carry specs.
if not MODEL_SPECS and MODEL_ZOO_WEEKLY_BUDGET >= 1:
    _log.warning(
        "config#1051: MODEL_SPECS is EMPTY while MODEL_ZOO_WEEKLY_BUDGET=%d — the "
        "model-zoo rotation would train 0 challengers. Resolved config: %s "
        "(ALPHA_ENGINE_EXPERIMENT_ID=%s). On the rotation box this is a config "
        "load failure, not an intentional empty roster; the rotation entrypoint "
        "will RAISE via assert_model_specs_loaded().",
        MODEL_ZOO_WEEKLY_BUDGET, _CONFIG_PATH, _EXPERIMENT_ID,
    )


class ModelSpecsEmptyError(RuntimeError):
    """config#1051: the model-zoo rotation roster (``MODEL_SPECS``) is empty
    while a rotation budget >= 1 is configured — a config-load failure on the
    rotation box, never an intentional empty roster."""


def assert_model_specs_loaded() -> None:
    """Fail loud if the rotation roster is empty while a budget is set. Called by
    the model-zoo rotation entrypoint so an empty-spec child spot RAISES at the
    earliest callsite instead of silently training 0 challengers (config#1051,
    no-silent-fails). The resolved config path + experiment id are in the message
    so the next run pins WHICH file loaded empty."""
    if not MODEL_SPECS and MODEL_ZOO_WEEKLY_BUDGET >= 1:
        raise ModelSpecsEmptyError(
            f"config#1051: MODEL_SPECS is EMPTY but MODEL_ZOO_WEEKLY_BUDGET="
            f"{MODEL_ZOO_WEEKLY_BUDGET} — the rotation cannot train any "
            f"challenger. Resolved config: {_CONFIG_PATH} "
            f"(ALPHA_ENGINE_EXPERIMENT_ID={_EXPERIMENT_ID}). The staged "
            f"predictor.yaml likely failed to parse or the experiment-package "
            f"search resolved to a spec-less file. See config#1051."
        )


# L4544: model-zoo IMMEDIATE selection — after the weekly rotation trains the
# challenger variants, the selection step ranks them by leak-free CPCV mean IC
# (gated by downside-Sortino + DSR) and picks a winner that beats the live
# champion by MODEL_ZOO_PROMOTE_MARGIN. OBSERVE-FIRST: with auto-promote-winner
# False (default), the step writes a leaderboard + recommended promotion to S3
# but does NOT promote; flip True (yaml or env) after a soak to activate
# auto-promote of the winning challenger (the executor turnover governor caps the
# resulting book move). Horizon variants (forward_days != champion) are never
# promote-eligible — observe-only. Env-overridable so the cutover flip needs no
# redeploy.
MODEL_ZOO_AUTO_PROMOTE_WINNER = _flag_env_or_yaml(
    "MODEL_ZOO_AUTO_PROMOTE_WINNER", _cfg.get("model_zoo_auto_promote_winner", False)
)
# CPCV mean-IC margin a challenger must beat the champion by to be selected.
MODEL_ZOO_PROMOTE_MARGIN = float(_cfg.get("model_zoo_promote_margin", 0.01))
# config#671/#673/#1052 — RELATIVE-BEST promotion: the absolute DSR-0.95 hurdle is
# NO LONGER a promotion blocker (it's an observability number — both champion and
# challenger are equally data-starved estimates on ~1 independent 21d block, so the
# incumbent has no special epistemic claim; defaulting to the staler champion is
# status-quo bias, and 21d-alpha models go stale). Promotion fires on beats-champion
# +margin. The ONLY absolute floor is this positive-IC guard: never promote a model
# whose leak-free-CPCV mean IC is <= MODEL_ZOO_PROMOTE_MIN_IC, even if it's best-of-N
# (a best-of-N with non-positive predictive IC is noise, not edge). Default 0.0.
MODEL_ZOO_PROMOTE_MIN_IC = float(_cfg.get("model_zoo_promote_min_ic", 0.0))
# L4582: CSCV-PBO target for the rotation's selection step (Bailey-LdP 2014,
# probability the in-sample winner lands in the bottom half out-of-sample).
# OBSERVE-only — logged to the leaderboard, not gating; <0.2 is the bar any
# public DSR/selection claim must clear.
MODEL_ZOO_PBO_TARGET = float(_cfg.get("model_zoo_pbo_target", 0.2))

# ── Champion/challenger Phase 1 shadow runner (L4469) ───────────────────────
# After the live (champion) inference writes predictions/{date}.json, the
# shadow runner re-scores the SAME prices/universe with each registered
# CHALLENGER's weights and writes predictor/predictions_shadow/{version_id}/
# {date}.json. Challengers TRADE ON NONE — only the live predictions feed the
# executor. The stage is non-critical (a shadow failure never aborts inference)
# and time-guarded (skips remaining challengers near the Lambda soft-timeout),
# so it can never delay or break the live path. No-op until challengers exist
# in the registry (they register from the next retrain via the L4469 capture
# fix), so default-on is safe. ``max_n`` bounds the per-run challenger count.
_shadow_cfg = _cfg.get("shadow_versions", {})
SHADOW_VERSIONS_ENABLED = _shadow_cfg.get("enabled", True)
SHADOW_VERSIONS_MAX_N = int(_shadow_cfg.get("max_n", 3))
RESID_MOM_BETA_WINDOW = _resid_mom_cfg.get("beta_window", 60)
RESID_MOM_WINDOW = _resid_mom_cfg.get("window", 252)
RESID_MOM_SKIP_DAYS = _resid_mom_cfg.get("skip_days", 21)
RESID_MOM_VOL_WINDOW = _resid_mom_cfg.get("vol_window", 20)
RESID_MOM_CHANGE_WINDOW = _resid_mom_cfg.get("change_window", 21)

# Regime classifier promotion gate. 3-class balanced-random baseline is 0.333;
# 0.40 requires the model to clear random by a meaningful margin on walk-forward
# OOS. Macro-F1 guards against a degenerate majority-class classifier that scores
# high on accuracy while having zero recall on bear or bull. Thresholds are
# deliberately modest while the model is unvalidated — raise after we see real
# OOS numbers from the first walk-forward run.
REGIME_OOS_ACCURACY_GATE = _wf_cfg.get("regime_oos_accuracy_gate", 0.40)
REGIME_OOS_MACRO_F1_GATE = _wf_cfg.get("regime_oos_macro_f1_gate", 0.30)
# Per-class recall floor. Bear/neutral/bull must each recall at least this
# fraction of their true instances on walk-forward OOS. Catches the failure
# mode where average metrics look OK but the classifier ignores a class
# (e.g. bear recall = 0 under L2 regularization + class imbalance, smoke
# test 2026-04-16). Set conservatively — a model that can't identify 15%
# of bear periods is useless as a regime signal regardless of macro-F1.
REGIME_OOS_PER_CLASS_RECALL_FLOOR = _wf_cfg.get("regime_oos_per_class_recall_floor", 0.15)

# ── Feature selection / noise detection ─────────────────────────────────────
_fs_cfg = _cfg.get("feature_selection", {})
SHAP_NOISE_THRESHOLD_PCT = _fs_cfg.get("shap_noise_threshold_pct", 1.0)
IC_NOISE_THRESHOLD = _fs_cfg.get("ic_noise_threshold", 0.005)
AUTO_PRUNE_NOISE_FEATURES = _fs_cfg.get("auto_prune", False)

# ── Calibration ────────────────────────────────────────────────────────────
_cal_cfg = _cfg.get("calibration", {})
CALIBRATION_METHOD = _cal_cfg.get("method", "platt")  # "platt" or "isotonic"
CALIBRATION_ENABLED = _cal_cfg.get("enabled", True)
# Optional class-balance + regularisation knobs for the LIVE calibrator
# (only consulted when CALIBRATION_METHOD == 'platt'). Defaults preserve
# historical behavior (no class weighting, C=1.0).
CALIBRATION_CLASS_WEIGHT = _cal_cfg.get("class_weight")  # None | "balanced" | {0:.., 1:..}
CALIBRATION_C = float(_cal_cfg.get("C", 1.0))

# Shadow calibrator — fit alongside the live calibrator on the same OOS
# rows, run through the output-distribution + stratified-per-regime gates,
# persist results in the manifest. Pure observability; does NOT gate
# promotion or feed inference. Set ``shadow_method: null`` (or omit) to
# disable shadow entirely. Added 2026-05-24 in response to the 5/23
# Platt-collapse incident: lets us tune Platt knobs (class_weight, C)
# across Saturday cycles without breaking the live calibrator, then
# flip live by swapping ``method`` once shadow gates pass for ≥1 cycle.
CALIBRATION_SHADOW_METHOD = _cal_cfg.get("shadow_method")  # None disables shadow
CALIBRATION_SHADOW_CLASS_WEIGHT = _cal_cfg.get("shadow_class_weight")
CALIBRATION_SHADOW_C = float(_cal_cfg.get("shadow_C", 1.0))

# Multi-horizon prediction config retired 2026-04-27 alongside the v2
# inference path — the v3 meta-model trains one horizon (cfg.FORWARD_DAYS).

# ── Meta-model architecture (v3.0) ─────────────────────────────────────────
_meta_cfg = _cfg.get("meta_model", {})
META_MODEL_ENABLED = _meta_cfg.get("enabled", False)

# Per-model feature sets (subsets of FEATURES computed by alpha-engine-data).
#
# v3.1 iteration (2026-04-15, predictor ROADMAP P2):
#
#   First pass added 6 features (return_60d, return_120d,
#   overnight_return_5d, intraday_return_5d, dist_from_5d_high,
#   dist_from_20d_high). Post-training Layer-1 feature importance
#   analysis (2026-04-15 full training run) showed:
#     return_60d          23.3% gain  ← #1 feature in momentum model
#     return_120d         14.0% gain  ← #4 feature
#     intraday_return_5d   8.7% gain  ← modest but non-zero
#     overnight_return_5d  0.0% gain  ← DEAD, removed
#     dist_from_20d_high   0.7% gain  ← near-dead, removed
#     dist_from_5d_high    0.3% gain  ← DEAD, removed
#
#   Dropped the 3 dead features from the subscription lists. They
#   remain in alpha-engine-data's feature store and ArcticDB rows
#   (compute cost is trivial, storage is trivial, and they may be
#   useful in a different model architecture or different horizon
#   later). This PR only updates which columns the Layer-1 GBMs
#   subscribe to.
# L4597-resid / config#635: the L1 feature SUBSCRIPTIONS are experiment
# beliefs (HARNESS_EXPERIMENT_CLASSIFICATION.md §1) — the experiment package
# (predictor.yaml `l1_features:`) overrides them; these literals are the
# public BASELINE the repo runs with when no package is present.
_BASELINE_MOMENTUM_FEATURES = [
    "momentum_5d", "momentum_20d", "price_vs_ma50", "price_vs_ma200",
    "rsi_14", "macd_cross",
    # v3.1: longer-horizon returns — pulled real weight in Layer-1
    "return_60d", "return_120d",
    # v3.1: intraday component of 5d return — marginal but non-zero
    "intraday_return_5d",
]
_BASELINE_VOLATILITY_FEATURES = [
    "atr_14_pct", "realized_vol_20d", "vol_ratio_10_60",
    "iv_rank", "dist_from_52w_high", "dist_from_52w_low",
]
_l1_cfg = _cfg.get("l1_features") or {}
MOMENTUM_FEATURES = list(_l1_cfg.get("momentum") or _BASELINE_MOMENTUM_FEATURES)
VOLATILITY_FEATURES = list(_l1_cfg.get("volatility") or _BASELINE_VOLATILITY_FEATURES)

# Stage 2b of regime-conditioning rebuild (plan: regime-conditioning-260510.md)
# — per-ticker risk features written to ArcticDB by alpha-engine-data Stage
# 2a (#202) and Stage 2a-extended (realized_vol_63d). Each varies cross-
# sectionally on a given date, so the existing rank-norm pipeline applies
# directly. Feeds the parallel ``prod_vol_risk_aug`` GBM (Stage 2b parallel
# observation) alongside the plain volatility GBM. Captures distinct risk
# dimensions the existing 6 vol features cannot split on:
#   - beta_60d:        systematic market exposure
#   - idio_vol_60d:    residual risk (after removing market-beta exposure)
#   - vol_of_vol_30d:  vol-regime stability
#   - max_drawdown_60d: recent left-tail risk
#   - realized_vol_63d: 3-month vol regime (pairs with 20d for term structure)
RISK_AUG_FEATURES = [
    "beta_60d",
    "idio_vol_60d",
    "vol_of_vol_30d",
    "max_drawdown_60d",
    "realized_vol_63d",
]

# Stage 1 of regime-conditioning rebuild (plan: regime-conditioning-260510.md)
# — macros consumed by L1 GBMs as time-series-z-scored features. Cross-
# sectional rank-norm degenerates these to 0.5 for every ticker because
# they're constant across tickers on a given date; time-series z-score
# over a rolling window preserves the regime signal (high VIX vs low VIX).
# Trees discover interactions with rank-normed per-ticker features
# through sequential splits — institutional SOTA per-feature normalization.
#
# Source: ``regime_features_df`` columns (built by
# ``model.regime_predictor.RegimePredictor.build_features()`` — kept as
# feature-engineering utility after the Tier-0 classifier was retired).
MACRO_NORM_FEATURES = [
    "spy_20d_return",
    "spy_20d_vol",
    "vix_level",
    "vix_term_slope",
    "yield_curve_slope",
    "market_breadth",
    # Stage 2c-partial additions (2026-05-10):
    # - vix_vix3m_ratio: institutional-canonical VIX term-structure
    #   normalization (ratio > 1 = backwardation/stress; < 1 = contango).
    #   Distinct from vix_term_slope which is bps-style normalized diff.
    # - market_breadth_200d: secular bull-vs-bear regime indicator.
    #   Distinct from market_breadth (50d cyclical breadth).
    # Stage 2c-full (gated on Stage 2.5 + 2.5b data ingestion + backfill):
    # yield_curve_10y_2y, hy_oas_level, hy_oas_change_21d.
    "vix_vix3m_ratio",
    "market_breadth_200d",
    # Stage 2c-full additions (2026-05-10):
    # - yield_curve_10y_2y: recession-canonical curve, distinct from
    #   yield_curve_slope (10Y-3M cyclical). Inverted values precede
    #   recessions historically.
    # - hy_oas_level / hy_oas_change_21d: HY-specific credit stress
    #   regime. License-gated to 2023+ on FRED (forward signal).
    # - baa10y_level / baa10y_change_21d: BBB-rated corporate spread,
    #   full 40y FRED history. Provides credit-regime signal across
    #   the full predictor training corpus.
    "yield_curve_10y_2y",
    "hy_oas_level",
    "hy_oas_change_21d",
    "baa10y_level",
    "baa10y_change_21d",
]
# Rolling window for time-series z-score. 252 trading days = 1 year, the
# institutional default for macro normalization (captures one full regime
# cycle without being overly slow to adapt). ``min_periods`` is set in
# ``data.dataset.time_series_zscore_normalize`` (default 60 = 3 months).
MACRO_NORM_WINDOW = 252

# Regime predictor uses macro series directly (not GBM features)
# Research calibrator uses signals.json fields (not price features)

# S3 keys for meta-model weights
META_WEIGHTS_PREFIX = "predictor/weights/meta/"
META_MANIFEST_KEY = "predictor/weights/meta/manifest.json"
META_FEATURE_LIST_KEY = "predictor/weights/meta/feature_list.json"

# ── Feature engineering parameters ───────────────────────────────────────────
FEATURE_CFG: dict = _cfg["features"]

# ── Feature Store (read-only — writes handled by alpha-engine-data) ──────────
FEATURE_STORE_PREFIX = "features/"

# Fundamental features (v3.0 — quarterly FMP data)
FUNDAMENTAL_FEATURES = [
    "pe_ratio",
    "pb_ratio",
    "debt_to_equity",
    "revenue_growth_yoy",
    "fcf_yield",
    "gross_margin",
    "roe",
    "current_ratio",
]

# ── AWS / Email ──────────────────────────────────────────────────────────────
AWS_REGION       = os.environ.get("AWS_REGION", "us-east-1")
EMAIL_SENDER     = get_secret("EMAIL_SENDER", required=False, default="") or ""
EMAIL_RECIPIENTS = [
    r.strip()
    for r in (get_secret("EMAIL_RECIPIENTS", required=False, default="") or "").split(",")
    if r.strip()
]
