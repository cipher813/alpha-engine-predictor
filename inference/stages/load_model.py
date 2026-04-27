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

    # Momentum model (GBM)
    try:
        from model.gbm_scorer import GBMScorer
        path = _dl(f"{prefix}momentum_model.txt", "meta_momentum.txt")
        ctx.meta_models["momentum"] = GBMScorer.load(path)
        log.info("Loaded momentum model")
    except Exception as e:
        log.warning("Momentum model not available: %s", e)

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

    # Research calibrator
    try:
        from model.research_calibrator import ResearchCalibrator
        path = _dl(f"{prefix}research_calibrator.json", "meta_research_cal.json")
        ctx.meta_models["research_calibrator"] = ResearchCalibrator.load(path)
        log.info("Loaded research calibrator")
    except Exception as e:
        log.warning("Research calibrator not available: %s", e)

    # Meta-model (ridge stacker)
    try:
        from model.meta_model import MetaModel
        path = _dl(f"{prefix}meta_model.pkl", "meta_model.pkl")
        ctx.meta_models["meta"] = MetaModel.load(path)
        log.info("Loaded meta-model")
    except Exception as e:
        log.warning("Meta-model not available: %s", e)

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


