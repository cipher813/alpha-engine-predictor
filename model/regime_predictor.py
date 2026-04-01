"""
model/regime_predictor.py — Market regime predictor.

Classifies the current market environment into bull/neutral/bear using
macro indicators (SPY returns, VIX, yield curve, market breadth). Outputs
posterior probabilities rather than hard classifications, enabling the
meta-model to blend strategies proportionally during transitions.

Starts with multinomial logistic regression (simple, interpretable).
Upgrade path: Bayesian HMM for regime persistence modeling.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Regime labels derived from subsequent 20-day SPY return
REGIME_LABELS = ["bear", "neutral", "bull"]
REGIME_MAP = {"bear": 0, "neutral": 1, "bull": 2}

# Thresholds for labeling historical regimes from SPY 20d forward return
BEAR_THRESHOLD = -0.03   # SPY down >3% over next 20d → bear
BULL_THRESHOLD = 0.03    # SPY up >3% over next 20d → bull


class RegimePredictor:
    """Predict market regime probabilities from macro indicators."""

    # Feature names expected in the input array (order matters)
    FEATURE_NAMES = [
        "spy_20d_return",
        "spy_20d_vol",
        "vix_level",
        "vix_term_slope",
        "yield_curve_slope",
        "market_breadth",
    ]

    def __init__(self):
        self._model = None
        self._fitted = False
        self._n_samples = 0
        self._accuracy = None

    def build_features(
        self,
        spy_series: pd.Series,
        vix_series: pd.Series | None = None,
        vix3m_series: pd.Series | None = None,
        tnx_series: pd.Series | None = None,
        irx_series: pd.Series | None = None,
        all_close_prices: dict[str, pd.Series] | None = None,
    ) -> pd.DataFrame:
        """
        Build regime feature matrix from macro time series.

        Returns DataFrame indexed by date with columns matching FEATURE_NAMES.
        """
        df = pd.DataFrame(index=spy_series.index)

        # SPY 20-day return
        df["spy_20d_return"] = (spy_series / spy_series.shift(20)) - 1.0

        # SPY 20-day realized volatility (annualized)
        log_ret = np.log(spy_series / spy_series.shift(1))
        df["spy_20d_vol"] = log_ret.rolling(20).std() * np.sqrt(252)

        # VIX level (normalized by baseline ~20)
        if vix_series is not None:
            vix_aligned = vix_series.reindex(df.index, method="ffill")
            df["vix_level"] = vix_aligned / 20.0
        else:
            df["vix_level"] = 1.0

        # VIX term structure slope
        if vix_series is not None and vix3m_series is not None:
            vix_aligned = vix_series.reindex(df.index, method="ffill")
            vix3m_aligned = vix3m_series.reindex(df.index, method="ffill")
            df["vix_term_slope"] = (vix_aligned - vix3m_aligned) / 20.0
        else:
            df["vix_term_slope"] = 0.0

        # Yield curve slope (10Y - 3M, normalized)
        if tnx_series is not None and irx_series is not None:
            tnx_aligned = tnx_series.reindex(df.index, method="ffill")
            irx_aligned = irx_series.reindex(df.index, method="ffill")
            df["yield_curve_slope"] = (tnx_aligned - irx_aligned) / 10.0
        else:
            df["yield_curve_slope"] = 0.0

        # Market breadth: % of stocks above 50-day MA
        if all_close_prices and len(all_close_prices) >= 10:
            breadth_by_date: dict[pd.Timestamp, list[int]] = {}
            for ticker, close_s in all_close_prices.items():
                if close_s is None or len(close_s) < 50:
                    continue
                ma50 = close_s.rolling(50).mean()
                above = (close_s > ma50).astype(int)
                for dt, val in above.items():
                    if pd.notna(val) and dt in df.index:
                        breadth_by_date.setdefault(dt, []).append(int(val))

            breadth_series = pd.Series(
                {dt: np.mean(vals) for dt, vals in breadth_by_date.items() if len(vals) >= 10},
                dtype=float,
            )
            df["market_breadth"] = breadth_series.reindex(df.index, method="ffill").fillna(0.5)
        else:
            df["market_breadth"] = 0.5  # neutral default

        return df.dropna()

    def build_labels(self, spy_series: pd.Series) -> pd.Series:
        """
        Label each date with a regime based on subsequent 20-day SPY return.

        bear (0):    SPY 20d forward return < -3%
        neutral (1): SPY 20d forward return in [-3%, +3%]
        bull (2):    SPY 20d forward return > +3%
        """
        fwd_20d = (spy_series.shift(-20) / spy_series) - 1.0
        labels = pd.Series(1, index=spy_series.index, dtype=int)  # default neutral
        labels[fwd_20d < BEAR_THRESHOLD] = 0
        labels[fwd_20d > BULL_THRESHOLD] = 2
        return labels.dropna()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegimePredictor":
        """
        Fit multinomial logistic regression.

        Parameters
        ----------
        X : array of shape (N, 6) — macro features
        y : array of shape (N,) — regime labels (0=bear, 1=neutral, 2=bull)
        """
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(
            C=1.0, solver="lbfgs",
            max_iter=1000, random_state=42,
        )
        self._model.fit(X, y)
        self._fitted = True
        self._n_samples = len(y)

        # Training accuracy
        preds = self._model.predict(X)
        self._accuracy = float((preds == y).mean())

        log.info(
            "RegimePredictor fitted: n=%d  accuracy=%.2f%%  classes=%s",
            self._n_samples, self._accuracy * 100,
            dict(zip(*np.unique(y, return_counts=True))),
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities.

        Returns array of shape (N, 3) with columns [P(bear), P(neutral), P(bull)].
        """
        if not self._fitted:
            raise RuntimeError("RegimePredictor not fitted")
        return self._model.predict_proba(X)

    def predict_single(self, features: dict) -> dict:
        """
        Predict regime for a single observation (today's macro data).

        Parameters
        ----------
        features : dict with keys matching FEATURE_NAMES

        Returns
        -------
        {"regime_bear": float, "regime_neutral": float, "regime_bull": float}
        """
        x = np.array([[features.get(f, 0.0) for f in self.FEATURE_NAMES]])
        proba = self.predict_proba(x)[0]
        return {
            "regime_bear": round(float(proba[0]), 4),
            "regime_neutral": round(float(proba[1]), 4),
            "regime_bull": round(float(proba[2]), 4),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        meta = {
            "type": "regime_predictor",
            "n_samples": self._n_samples,
            "accuracy": round(self._accuracy, 4) if self._accuracy else None,
            "features": self.FEATURE_NAMES,
        }
        Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        log.info("RegimePredictor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimePredictor":
        path = Path(path)
        rp = cls()
        with open(path, "rb") as f:
            rp._model = pickle.load(f)
        rp._fitted = True
        meta_path = Path(str(path) + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            rp._n_samples = meta.get("n_samples", 0)
            rp._accuracy = meta.get("accuracy")
        log.info("RegimePredictor loaded from %s", path)
        return rp
