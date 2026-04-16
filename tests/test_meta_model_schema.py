"""MetaModel feature-schema persistence + backwards-compat.

The meta-model's feature list must be carried on the model instance (not
pulled from the module-level META_FEATURES at inference), so a ridge trained
on an older schema still serves correctly after META_FEATURES is extended
in code. Without this, every META_FEATURES addition would silently break
inference against the previously-deployed model until the next retrain.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from model.meta_model import META_FEATURES, MACRO_FEATURE_META_MAP, MetaModel


def _synth_training_data(feature_list, n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(feature_list)))
    # Weak linear signal so ridge has something to fit
    y = 0.01 * X[:, 0] + 0.005 * X[:, 1] + rng.normal(scale=0.02, size=n)
    return X, y


def test_feature_names_persisted_on_save(tmp_path):
    X, y = _synth_training_data(META_FEATURES)
    mm = MetaModel(alpha=1.0).fit(X, y, feature_names=META_FEATURES)
    assert mm._feature_names == list(META_FEATURES)

    pkl_path = tmp_path / "mm.pkl"
    mm.save(pkl_path)
    meta = json.loads(Path(str(pkl_path) + ".meta.json").read_text())
    assert meta["feature_names"] == list(META_FEATURES)


def test_predict_single_uses_model_feature_names_not_module(tmp_path):
    """The critical backwards-compat check: a model trained on an older,
    shorter feature list must produce predictions without shape errors even
    after META_FEATURES has grown. predict_single must read the model's own
    schema, not the module constant."""
    legacy_features = [
        "research_calibrator_prob",
        "momentum_score",
        "expected_move",
        "regime_bull",
        "regime_bear",
        "research_composite_score",
        "research_conviction",
        "sector_macro_modifier",
    ]
    # Sanity — legacy list must be a proper prefix of the current list
    assert legacy_features == META_FEATURES[: len(legacy_features)]
    # And the current list must have been extended (otherwise this test is vacuous)
    assert len(META_FEATURES) > len(legacy_features)

    X, y = _synth_training_data(legacy_features)
    mm = MetaModel(alpha=1.0).fit(X, y, feature_names=legacy_features)
    pkl_path = tmp_path / "legacy_mm.pkl"
    mm.save(pkl_path)

    # Load and predict with a feature dict that contains BOTH legacy and new
    # keys — predict_single must ignore the new keys because the loaded model
    # was not trained on them.
    mm2 = MetaModel.load(pkl_path)
    feats = {name: 0.1 for name in META_FEATURES}
    pred = mm2.predict_single(feats)
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_pre_pr34_models_without_feature_names_still_load(tmp_path):
    """Older models saved before feature_names was persisted have only a
    coefficients dict. load() must reconstruct feature_names from coefficient
    keys (minus intercept) so predict_single still finds a valid schema."""
    legacy_features = ["feat_a", "feat_b", "feat_c"]
    X, y = _synth_training_data(legacy_features)
    mm = MetaModel(alpha=1.0).fit(X, y, feature_names=legacy_features)

    # Manually strip feature_names from sidecar to simulate a pre-PR34 save
    pkl_path = tmp_path / "legacy.pkl"
    mm.save(pkl_path)
    meta_path = Path(str(pkl_path) + ".meta.json")
    meta = json.loads(meta_path.read_text())
    meta.pop("feature_names", None)
    meta_path.write_text(json.dumps(meta))

    mm2 = MetaModel.load(pkl_path)
    assert mm2._feature_names == legacy_features

    feats = {"feat_a": 0.1, "feat_b": 0.2, "feat_c": 0.3}
    pred = mm2.predict_single(feats)
    assert np.isfinite(pred)


def test_macro_feature_names_included_in_meta_features():
    """The 6 raw macro features must be in META_FEATURES with the `macro_`
    prefix so they don't collide with per-ticker features, and the map must
    point from the classifier's own column names to the prefixed meta names."""
    for src, meta_name in MACRO_FEATURE_META_MAP.items():
        assert meta_name.startswith("macro_")
        assert meta_name == f"macro_{src}"
        assert meta_name in META_FEATURES
