"""Stage: load_model — Load GBM or MLP model weights."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import config as cfg
from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)


def run(ctx: PipelineContext) -> None:
    """Load model weights from S3 or local checkpoint."""
    from inference.daily_predict import (
        load_gbm_local, load_gbm_s3, load_model, load_model_local,
    )

    if ctx.model_type == "gbm":
        _load_gbm(ctx)
    else:
        _load_mlp(ctx)


def _load_gbm(ctx: PipelineContext) -> None:
    from inference.daily_predict import load_gbm_local, load_gbm_s3

    # Determine inference mode from gbm_mode.json
    ctx.inference_mode = "mse"
    if getattr(cfg, "GBM_ENSEMBLE_LAMBDARANK", True):
        if ctx.local:
            mode_path = Path("checkpoints/gbm_mode.json")
            if mode_path.exists():
                try:
                    ctx.inference_mode = json.loads(mode_path.read_text()).get("mode", "mse")
                except Exception as exc:
                    log.warning("Could not read local gbm_mode.json: %s — defaulting to mse", exc)
        else:
            try:
                import boto3 as _boto3_mode
                _s3_mode = _boto3_mode.client("s3")
                _mode_obj = _s3_mode.get_object(Bucket=ctx.bucket, Key=cfg.GBM_MODE_KEY)
                ctx.inference_mode = json.loads(_mode_obj["Body"].read()).get("mode", "mse")
            except Exception as exc:
                log.info("gbm_mode.json not found on S3 — defaulting to mse: %s", exc)
    log.info("Inference mode: %s", ctx.inference_mode)

    # Load BOTH MSE and lambdarank models
    try:
        if ctx.local:
            ctx.mse_scorer = load_gbm_local("checkpoints/gbm_best.txt")
        else:
            ctx.mse_scorer = load_gbm_s3(ctx.bucket, cfg.GBM_MSE_WEIGHTS_KEY)
        log.info("MSE model loaded for alpha estimation")
    except Exception as exc:
        log.warning("MSE model not available: %s", exc)

    try:
        if ctx.local:
            ctx.rank_scorer = load_gbm_local("checkpoints/gbm_rank_best.txt")
        else:
            ctx.rank_scorer = load_gbm_s3(ctx.bucket, cfg.GBM_RANK_WEIGHTS_KEY)
        log.info("Lambdarank model loaded for ranking")
    except Exception as exc:
        log.warning("Lambdarank model not available: %s", exc)

    # Select primary scorer
    if ctx.inference_mode == "rank" and ctx.rank_scorer is not None:
        ctx.scorer = ctx.rank_scorer
    elif ctx.inference_mode == "ensemble" and ctx.mse_scorer is not None:
        ctx.scorer = ctx.mse_scorer
    elif ctx.mse_scorer is not None:
        ctx.scorer = ctx.mse_scorer
    elif ctx.rank_scorer is not None:
        ctx.scorer = ctx.rank_scorer
    else:
        raise RuntimeError("No GBM model available — both MSE and rank failed to load")

    ctx.model_version = f"GBM-v{ctx.scorer._best_iteration}"
    ctx.val_loss = ctx.scorer._val_ic

    # Load CatBoost model (optional — for LGB-Cat ensemble)
    ctx.cat_scorer = None
    if getattr(cfg, "CATBOOST_ENABLED", False) and ctx.inference_mode in ("lgb_cat_blend", "catboost"):
        try:
            from model.catboost_scorer import CatBoostScorer
            if ctx.local:
                cat_path = Path("checkpoints/catboost_best.cbm")
                if cat_path.exists():
                    ctx.cat_scorer = CatBoostScorer.load(cat_path)
            else:
                import boto3 as _boto3_cat
                import tempfile as _tmpmod_cat
                _s3_cat = _boto3_cat.client("s3")
                _cat_tmp = Path(_tmpmod_cat.gettempdir()) / "catboost_latest.cbm"
                _cat_meta_tmp = Path(str(_cat_tmp) + ".meta.json")
                _s3_cat.download_file(ctx.bucket, cfg.CATBOOST_WEIGHTS_KEY, str(_cat_tmp))
                try:
                    _s3_cat.download_file(ctx.bucket, cfg.CATBOOST_WEIGHTS_META_KEY, str(_cat_meta_tmp))
                except Exception:
                    pass
                ctx.cat_scorer = CatBoostScorer.load(_cat_tmp)
            if ctx.cat_scorer:
                log.info("CatBoost model loaded (val_IC=%.4f)", ctx.cat_scorer._val_ic)
        except Exception as exc:
            log.info("CatBoost model not available: %s", exc)

    # Load blend weights from mode.json
    ctx.blend_weights = None
    if ctx.inference_mode == "lgb_cat_blend":
        try:
            if ctx.local:
                mode_path = Path("checkpoints/gbm_mode.json")
                if mode_path.exists():
                    _mode_data = json.loads(mode_path.read_text())
                    ctx.blend_weights = _mode_data.get("blend_weights")
            else:
                import boto3 as _boto3_bw
                _s3_bw = _boto3_bw.client("s3")
                _mode_obj = _s3_bw.get_object(Bucket=ctx.bucket, Key=cfg.GBM_MODE_KEY)
                _mode_data = json.loads(_mode_obj["Body"].read())
                ctx.blend_weights = _mode_data.get("blend_weights")
        except Exception:
            pass
        if ctx.blend_weights:
            log.info("Blend weights loaded: %s", ctx.blend_weights)
        else:
            ctx.blend_weights = {"lgb": 0.5, "cat": 0.5}  # default

    # Load multi-horizon auxiliary models (optional)
    ctx.horizon_scorers = {}
    if getattr(cfg, "MULTI_HORIZON_ENABLED", False):
        for h in getattr(cfg, "MULTI_HORIZON_LIST", [1, 10, 20]):
            if h == getattr(cfg, "FORWARD_DAYS", 5):
                continue  # primary model already loaded
            try:
                _h_key = f"predictor/weights/gbm_mse_{h}d_latest.txt"
                if ctx.local:
                    _h_path = Path(f"checkpoints/gbm_mse_{h}d_best.txt")
                    if _h_path.exists():
                        from inference.daily_predict import load_gbm_local
                        ctx.horizon_scorers[h] = load_gbm_local(str(_h_path))
                else:
                    from inference.daily_predict import load_gbm_s3
                    ctx.horizon_scorers[h] = load_gbm_s3(ctx.bucket, _h_key)
                log.info("Loaded %dd horizon model", h)
            except Exception as exc:
                log.info("Horizon %dd model not available: %s", h, exc)

    # Load Platt calibrator (optional — graceful degradation to linear)
    ctx.calibrator = None
    if getattr(cfg, "CALIBRATION_ENABLED", True):
        try:
            from model.calibrator import PlattCalibrator
            if ctx.local:
                cal_path = Path("checkpoints/calibrator.pkl")
                if cal_path.exists():
                    ctx.calibrator = PlattCalibrator.load(cal_path)
            else:
                import boto3 as _boto3_cal
                _s3_cal = _boto3_cal.client("s3")
                import tempfile as _tmpmod
                _cal_tmp = Path(_tmpmod.gettempdir()) / "calibrator_latest.pkl"
                _cal_meta_tmp = Path(str(_cal_tmp) + ".meta.json")
                _s3_cal.download_file(ctx.bucket, cfg.CALIBRATOR_WEIGHTS_KEY, str(_cal_tmp))
                try:
                    _s3_cal.download_file(ctx.bucket, cfg.CALIBRATOR_WEIGHTS_META_KEY, str(_cal_meta_tmp))
                except Exception:
                    pass  # meta is optional for loading
                ctx.calibrator = PlattCalibrator.load(_cal_tmp)
            if ctx.calibrator and ctx.calibrator.is_fitted:
                log.info("Calibrator loaded (method=%s, ECE=%.4f)",
                         ctx.calibrator.method,
                         ctx.calibrator._ece_after or 0.0)
        except Exception as exc:
            log.info("Calibrator not available — using linear fallback: %s", exc)


def _load_mlp(ctx: PipelineContext) -> None:
    from inference.daily_predict import load_model, load_model_local

    if ctx.local:
        ctx.model, ctx.checkpoint = load_model_local("checkpoints/best.pt")
    else:
        ctx.model, ctx.checkpoint = load_model(ctx.bucket, cfg.MODEL_WEIGHTS_KEY)
    norm_stats = ctx.checkpoint.get("norm_stats", {})
    if not norm_stats:
        log.warning("No norm_stats in checkpoint — features may not normalize correctly")
    ctx.model_version = ctx.checkpoint.get("model_version", "unknown")
    ctx.val_loss = ctx.checkpoint.get("val_loss", float("nan"))
