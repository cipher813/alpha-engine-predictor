"""Stage: load_model — Load v3 meta-model weights from S3 / local cache.

The v2 single-GBM and multi-horizon-ensemble inference paths were retired
2026-04-13 when ``train_handler`` collapsed its dispatcher to
``run_meta_training()``. The corresponding inference loaders
(``_load_gbm`` + ``load_gbm_local`` + ``load_gbm_s3``) plus their
config switches (``GBM_ENSEMBLE_LAMBDARANK``, ``MULTI_HORIZON_*``,
``CATBOOST_*``) were deleted in the v2 cleanup PR after a clean grep
confirmed v3's ``_load_meta_models`` was the sole consumer reachable in
production. ``META_MODEL_ENABLED`` is hardcoded ``true`` in the config
repo, so the dispatcher's else-branch was dead code.
"""

from __future__ import annotations

import logging
from pathlib import Path

import config as cfg
from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)


# ── Stage entry point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Load v3 meta-model weights — the sole production path."""
    _load_meta_models(ctx)


def _load_meta_models(ctx: PipelineContext) -> None:
    """Load all Layer 1 + meta-model weights from S3."""
    import tempfile as _tmp
    tmp_dir = Path(_tmp.gettempdir())

    ctx.inference_mode = "meta"
    ctx.meta_models = {}
    prefix = cfg.META_WEIGHTS_PREFIX

    def _dl(s3_key, local_name):
        """Download from S3 to temp, return path."""
        local = tmp_dir / local_name
        import boto3 as _b3
        _s3 = _b3.client("s3")
        _s3.download_file(ctx.bucket, s3_key, str(local))
        # Also try meta.json
        try:
            _s3.download_file(ctx.bucket, f"{s3_key}.meta.json", str(local) + ".meta.json")
        except Exception:
            pass
        return local

    # Momentum L1 component is the deterministic weighted-blend baseline
    # (model/momentum_scorer.py) — no S3 weights to load. Existing
    # s3://.../momentum_model.txt from prior training cycles is left in
    # place; cleanup is a separate concern.

    # Volatility model (GBM)
    try:
        from model.gbm_scorer import GBMScorer
        path = _dl(f"{prefix}volatility_model.txt", "meta_volatility.txt")
        ctx.meta_models["volatility"] = GBMScorer.load(path)
        log.info("Loaded volatility model")
    except Exception as e:
        log.warning("Volatility model not available: %s", e)

    # Regime predictor load retired 2026-04-16. Tier 0 classifier failed
    # honest walk-forward validation and was removed from the critical path.
    # Raw macro features feed the ridge directly via build_features() in
    # run_inference.py. Tier 1 regime rewrite is on the roadmap; when it
    # ships it will re-introduce a loader block here with its own path.

    # Research calibrator (bucket lookup — canonical source for
    # research_calibrator_prob META_FEATURE).
    try:
        from model.research_calibrator import ResearchCalibrator
        path = _dl(f"{prefix}research_calibrator.json", "meta_research_cal.json")
        ctx.meta_models["research_calibrator"] = ResearchCalibrator.load(path)
        log.info("Loaded research calibrator")
    except Exception as e:
        log.warning("Research calibrator not available: %s", e)

    # Audit Phase 3 PR 3/5 (2026-05-07): ResearchGBMScorer ships alongside
    # the bucket-lookup in observe-only mode. The LGB exists in S3 (since
    # PR #95 wired the training-side persistence) but its output rides
    # alongside the bucket-lookup as a parallel diagnostic. The
    # bucket-lookup remains the canonical source for the
    # research_calibrator_prob META_FEATURE the meta-Ridge consumes;
    # the LGB output is captured in each prediction dict as
    # `research_gbm_prob` for observation. Cutover happens in PR 4.
    #
    # Loader is tolerant of the LGB being absent (early days post-PR-#95
    # before the first Sat training cycle has produced research_gbm.pkl,
    # or any cycle where prod_research_gbm was None due to insufficient
    # data). Inference doesn't fail — it just writes None for
    # research_gbm_prob and the bucket-lookup remains the only source.
    try:
        from model.research_gbm import ResearchGBMScorer
        path = _dl(f"{prefix}research_gbm.pkl", "research_gbm.pkl")
        ctx.meta_models["research_gbm"] = ResearchGBMScorer.load(path)
        log.info("Loaded ResearchGBMScorer (Phase 3 observe-only)")
    except Exception as e:
        log.debug("ResearchGBMScorer not available (observe-only OK): %s", e)

    # Stage 1c (regime-conditioning rebuild): macro-augmented volatility
    # GBM ships ALONGSIDE the plain volatility GBM in observe-only mode.
    # Inference predicts with both; ``expected_move_macro_aug`` rides as
    # a parallel field in predictions JSON. The plain ``expected_move``
    # remains canonical until Stage 1d cuts over after the
    # variant_cutover_gate clears (≥15% relative IC lift on accumulated
    # parallel-observation data).
    #
    # Loader is tolerant: if the file is absent (early observation period
    # before Sat training fires, or any cycle where parallel fit failed),
    # inference proceeds with plain volatility only. The parallel field
    # rides as null in the output JSON.
    try:
        from model.gbm_scorer import GBMScorer
        path = _dl(
            f"{prefix}volatility_macro_aug_model.txt",
            "volatility_macro_aug.txt",
        )
        ctx.meta_models["volatility_macro_aug"] = GBMScorer.load(path)
        log.info("Loaded volatility GBM (Stage 1b macro-aug, observe-only)")
    except Exception as e:
        log.debug("volatility_macro_aug not available (observe-only OK): %s", e)

    # Stage 2b (regime-conditioning rebuild): parallel risk-augmented
    # volatility GBM. Reads per-ticker risk features (beta_60d, idio_vol_60d,
    # vol_of_vol_30d, max_drawdown_60d, realized_vol_63d) from ArcticDB
    # alongside the existing vol features. expected_move_risk_aug rides
    # alongside expected_move as the parallel field. Same loader pattern
    # as Stage 1b: tolerant of absence (early observation period or any
    # cycle where the parallel fit failed).
    try:
        from model.gbm_scorer import GBMScorer
        path = _dl(
            f"{prefix}volatility_risk_aug_model.txt",
            "volatility_risk_aug.txt",
        )
        ctx.meta_models["volatility_risk_aug"] = GBMScorer.load(path)
        log.info("Loaded volatility GBM (Stage 2b risk-aug, observe-only)")
    except Exception as e:
        log.debug("volatility_risk_aug not available (observe-only OK): %s", e)

    # Meta-model (ridge stacker)
    try:
        from model.meta_model import MetaModel
        path = _dl(f"{prefix}meta_model.pkl", "meta_model.pkl")
        ctx.meta_models["meta"] = MetaModel.load(path)
        log.info("Loaded meta-model")
    except Exception as e:
        log.warning("Meta-model not available: %s", e)

    # Track A PR 5/6 cutover (2026-05-09): canonical_meta_model.pkl loader
    # retired. The single meta_model.pkl above is now trained on canonical
    # labels — no separate canonical Ridge to load. Existing
    # s3://.../canonical_meta_model.pkl from the observe-only era is
    # left in place; cleanup is a separate concern.

    # Stage 3 PR 3 (regime-conditioning rebuild): parallel triple-barrier
    # L2 Ridge ships ALONGSIDE the canonical Ridge in observe-only mode.
    # Inference predicts with both; ``meta_alpha_tb`` rides as a parallel
    # field in predictions JSON. The canonical ``meta_alpha`` remains
    # authoritative until Stage 3 PR 5 cuts over after the
    # variant_cutover_gate clears (≥15% relative IC lift on accumulated
    # parallel-observation data over ≥3 Saturday SF firings).
    #
    # Loader is tolerant: if the file is absent (early observation period
    # before Sat training fires, or any cycle where parallel fit had
    # n_tb_finite<100 and was skipped), inference proceeds with the
    # canonical Ridge only. The parallel field rides as null in the
    # output JSON.
    try:
        from model.meta_model import MetaModel
        path = _dl(f"{prefix}meta_model_tb.pkl", "meta_model_tb.pkl")
        ctx.meta_models["meta_tb"] = MetaModel.load(path)
        log.info("Loaded meta-model (Stage 3 triple-barrier, observe-only)")
    except Exception as e:
        log.debug("meta_model_tb not available (observe-only OK): %s", e)

    # Task B (observe-only): meta-label classifier ships ALONGSIDE the
    # canonical Ridge. Inference predicts P(up barrier before down) and emits
    # the parallel ``barrier_win_prob`` field; the executor sizing consumer
    # stays dormant until the Task B2 cutover. Loader is tolerant: absent file
    # (early observation period, classifier disabled, or a cycle with <100
    # touch-finite labels) → the field rides as null.
    try:
        from model.meta_label_classifier import MetaLabelClassifier
        path = _dl(f"{prefix}meta_label_classifier.pkl", "meta_label_classifier.pkl")
        ctx.meta_models["meta_label_clf"] = MetaLabelClassifier.load(path)
        log.info("Loaded meta-label classifier (Task B, observe-only)")
    except Exception as e:
        log.debug("meta_label_classifier not available (observe-only OK): %s", e)

    # Calibrator (Platt scaling on meta-model output)
    ctx.calibrator = None
    if getattr(cfg, "CALIBRATION_ENABLED", True):
        try:
            from model.calibrator import PlattCalibrator
            path = _dl(cfg.CALIBRATOR_WEIGHTS_KEY, "meta_calibrator.pkl")
            ctx.calibrator = PlattCalibrator.load(path)
            log.info("Loaded Platt calibrator for meta-model output")
        except Exception:
            pass

    n_loaded = len(ctx.meta_models)
    if n_loaded == 0:
        raise RuntimeError("No meta-models available — cannot run inference")

    ctx.model_version = f"meta-v3.0-{n_loaded}models"
    ctx.val_loss = ctx.meta_models.get("meta", type("", (), {"_val_ic": 0.0}))._val_ic
    log.info("Meta-model inference ready: %d models loaded", n_loaded)


