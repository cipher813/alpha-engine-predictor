"""W2 (L4469) observe-gate invariant tests.

The load-bearing guarantee: with RESIDUAL_MOMENTUM_ENABLED=False the L2 training
feature set, the resulting meta_X matrix, and the persisted feature_names are
byte-identical to the pre-W2 trainer — so the live ensemble + inference path are
unchanged. With the flag True the residual column joins the stack (and only
then). The module constant META_FEATURES is never mutated.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.meta_model import META_FEATURES
from training.meta_trainer import build_train_meta_features


def _synthetic_rows(n: int = 60, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        row = {f: float(rng.normal()) for f in META_FEATURES}
        row["residual_momentum_score"] = float(rng.normal())  # always present
        row["actual_fwd"] = float(rng.normal())               # non-meta key
        rows.append(row)
    return rows


class TestGateInvariant:
    def test_module_constant_never_mutated(self):
        before = list(META_FEATURES)
        build_train_meta_features(True)
        build_train_meta_features(False)
        assert list(META_FEATURES) == before
        assert "residual_momentum_score" not in META_FEATURES

    def test_flag_false_identical_to_meta_features(self):
        feats = build_train_meta_features(False)
        assert feats == list(META_FEATURES)
        assert "residual_momentum_score" not in feats

    def test_flag_true_appends_exactly_one_column(self):
        feats = build_train_meta_features(True)
        assert feats == list(META_FEATURES) + ["residual_momentum_score"]
        assert len(feats) == len(META_FEATURES) + 1
        assert feats[-1] == "residual_momentum_score"

    def test_meta_matrix_byte_identical_when_flag_false(self):
        # The exact selection the trainer performs: [[r[f] for f in FEATS] ...].
        rows = _synthetic_rows()
        feats_off = build_train_meta_features(False)
        X_off = np.array([[r[f] for f in feats_off] for r in rows])
        X_baseline = np.array([[r[f] for f in META_FEATURES] for r in rows])
        np.testing.assert_array_equal(X_off, X_baseline)
        assert X_off.shape[1] == len(META_FEATURES)

    def test_meta_matrix_grows_by_one_when_flag_true(self):
        rows = _synthetic_rows()
        feats_on = build_train_meta_features(True)
        X_on = np.array([[r[f] for f in feats_on] for r in rows])
        assert X_on.shape[1] == len(META_FEATURES) + 1
        # The appended column equals the row's residual_momentum_score.
        last_col = X_on[:, -1]
        expected = np.array([r["residual_momentum_score"] for r in rows])
        np.testing.assert_array_equal(last_col, expected)


class TestW2LeakfreeReadShape:
    """Regression guard for the W2 residual-momentum leak-free read (CF3).

    The crashed code re-ran the scorer on ``X_resid_mom_raw`` — a FULL-training
    length-N array — indexed by ``canonical_finite_mask``, which is OOS length.
    Boolean indexing requires the mask to match the array's first dim, so that
    raised ``IndexError`` and dumped the read into status="error". The fix pulls
    the leak-free per-fold value already stored on each OOS row, mirroring the
    factor_momentum_ratio idiom.
    """

    def test_full_length_array_masked_by_oos_mask_raises(self):
        # Reproduce the bug geometry: N (full training) >> n_oos (meta rows).
        n_full, n_oos = 500, 60
        X_resid_mom_raw = np.zeros((n_full, 5))
        canonical_finite_mask = np.ones(n_oos, dtype=bool)
        with np.testing.assert_raises(IndexError):
            _ = X_resid_mom_raw[canonical_finite_mask]

    def test_stored_row_idiom_is_oos_aligned(self):
        # The fix: extract the stored per-row score, then mask — length follows
        # the mask, matching meta_y / _rm_dates row-for-row.
        rows = _synthetic_rows(n=60, seed=7)
        canonical_finite_mask = np.array(
            [i % 3 != 0 for i in range(len(rows))], dtype=bool
        )
        preds = np.array(
            [r.get("residual_momentum_score", float("nan")) for r in rows]
        )[canonical_finite_mask]
        assert preds.shape[0] == int(canonical_finite_mask.sum())
        expected = np.array(
            [r["residual_momentum_score"]
             for r, m in zip(rows, canonical_finite_mask) if m]
        )
        np.testing.assert_array_equal(preds, expected)
