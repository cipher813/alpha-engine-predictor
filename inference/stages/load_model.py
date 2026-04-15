"""Stage: load_model — Load GBM or MLP model weights."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import config as cfg
from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)


# ── Model loading functions (migrated from daily_predict.py) ─────────────────


def load_gbm_local(path: str = "checkpoints/gbm_best.txt"):
    """Load GBMScorer from a local file path."""
    from model.gbm_scorer import GBMScorer
    scorer = GBMScorer.load(path)
    log.info(
        "GBMScorer loaded (local): val_IC=%.4f  best_iter=%d",
        scorer._val_ic, scorer._best_iteration,
    )
    return scorer


def load_gbm_s3(s3_bucket: str, weights_key: str):
    """
    Download GBM booster + meta from S3 to /tmp and load.

    Parameters
    ----------
    s3_bucket :   S3 bucket name.
    weights_key : S3 key for the booster text file (e.g. predictor/weights/gbm_latest.txt).

    Returns
    -------
    GBMScorer instance with booster loaded.
    """
    from model.gbm_scorer import GBMScorer
    try:
        import boto3
        s3 = boto3.client("s3")
        tmp_dir = Path(tempfile.mkdtemp())
        local_path = tmp_dir / "gbm_model.txt"
        meta_path  = tmp_dir / "gbm_model.txt.meta.json"
        log.info("Downloading GBM booster from s3://%s/%s", s3_bucket, weights_key)
        s3.download_file(s3_bucket, weights_key, str(local_path))
        # Meta file is best-effort — GBMScorer.load() handles its absence gracefully
        try:
            s3.download_file(s3_bucket, weights_key + ".meta.json", str(meta_path))
        except Exception:
            pass
        scorer = GBMScorer.load(str(local_path))
        log.info(
            "GBMScorer loaded from S3: val_IC=%.4f  best_iter=%d",
            scorer._val_ic, scorer._best_iteration,
        )
        return scorer
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download GBM booster from s3://{s3_bucket}/{weights_key}: {exc}"
        ) from exc


# ── Stage entry point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Load model weights from S3 or local checkpoint."""

    if getattr(cfg, "META_MODEL_ENABLED", False) and ctx.model_type == "gbm":
        _load_meta_models(ctx)
        return

    _load_gbm(ctx)


def _load_gbm(ctx: PipelineContext) -> None:
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
                        ctx.horizon_scorers[h] = load_gbm_local(str(_h_path))
                else:
                    ctx.horizon_scorers[h] = load_gbm_s3(ctx.bucket, _h_key)
                log.info("Loaded %dd horizon model", h)
            except Exception as exc:
                log.info("Horizon %dd model not available: %s", h, exc)

    # Load isotonic calibrator. Distinguish two failure modes:
    #   1. Calibrator file absent (S3 NoSuchKey / local file missing) —
    #      expected during the pre-first-retrain window after the 2026-04-15
    #      binary UP/DOWN migration. Log WARNING; inference falls back to the
    #      linear heuristic rescaling in run_inference.py.
    #   2. Calibrator file present but corrupted / incompatible — this is a
    #      silent-quality regression if swallowed. Raise.
    # Per feedback_no_silent_fails + feedback_hard_fail_until_stable.
    ctx.calibrator = None
    if getattr(cfg, "CALIBRATION_ENABLED", True):
        from model.calibrator import PlattCalibrator
        if ctx.local:
            cal_path = Path("checkpoints/calibrator.pkl")
            if cal_path.exists():
                ctx.calibrator = PlattCalibrator.load(cal_path)
            else:
                log.warning(
                    "Local calibrator missing at %s — falling back to linear "
                    "heuristic. Expected only before first post-migration retrain.",
                    cal_path,
                )
        else:
            import boto3 as _boto3_cal
            from botocore.exceptions import ClientError
            _s3_cal = _boto3_cal.client("s3")
            import tempfile as _tmpmod
            _cal_tmp = Path(_tmpmod.gettempdir()) / "isotonic_calibrator.pkl"
            _cal_meta_tmp = Path(str(_cal_tmp) + ".meta.json")
            try:
                _s3_cal.download_file(ctx.bucket, cfg.CALIBRATOR_WEIGHTS_KEY, str(_cal_tmp))
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                    log.warning(
                        "S3 calibrator missing at s3://%s/%s — falling back to "
                        "linear heuristic. Expected only before first "
                        "post-migration retrain writes the calibrator.",
                        ctx.bucket, cfg.CALIBRATOR_WEIGHTS_KEY,
                    )
                else:
                    raise
            else:
                # Sidecar is best-effort — calibrator pickle is authoritative
                try:
                    _s3_cal.download_file(ctx.bucket, cfg.CALIBRATOR_WEIGHTS_META_KEY, str(_cal_meta_tmp))
                except ClientError as exc:
                    if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                        log.warning("Calibrator sidecar missing: %s", cfg.CALIBRATOR_WEIGHTS_META_KEY)
                    else:
                        raise
                # Pickle present but load failure is a real error — raise.
                ctx.calibrator = PlattCalibrator.load(_cal_tmp)
        if ctx.calibrator and ctx.calibrator.is_fitted:
            log.info("Calibrator loaded (method=%s, ECE=%.4f)",
                     ctx.calibrator.method,
                     ctx.calibrator._ece_after or 0.0)


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

    # Regime predictor
    try:
        from model.regime_predictor import RegimePredictor
        path = _dl(f"{prefix}regime_predictor.pkl", "meta_regime.pkl")
        ctx.meta_models["regime"] = RegimePredictor.load(path)
        log.info("Loaded regime predictor")
    except Exception as e:
        log.warning("Regime predictor not available: %s", e)

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


