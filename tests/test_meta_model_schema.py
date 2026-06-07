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
    after META_FEATURES changes shape. predict_single must read the model's
    own schema, not the module constant. Uses a synthetic legacy schema with
    NON-overlapping names so we can invoke predict_single with both the legacy
    feature dict and extra keys without relying on the current META_FEATURES
    order (which changes as the stack evolves — e.g. regime removal 2026-04-16)."""
    legacy_features = [f"legacy_feat_{i}" for i in range(5)]

    X, y = _synth_training_data(legacy_features)
    mm = MetaModel(alpha=1.0).fit(X, y, feature_names=legacy_features)
    pkl_path = tmp_path / "legacy_mm.pkl"
    mm.save(pkl_path)

    # Load and predict with a feature dict that contains BOTH legacy and new
    # keys — predict_single must ignore the new keys because the loaded model
    # was not trained on them.
    mm2 = MetaModel.load(pkl_path)
    feats = {name: 0.1 for name in legacy_features}
    feats.update({name: 0.2 for name in META_FEATURES})  # extra, should be ignored
    pred = mm2.predict_single(feats)
    assert isinstance(pred, float)
    assert np.isfinite(pred)
    # And the model's feature names must match what it was trained on — not
    # what the module currently exports
    assert mm2._feature_names == legacy_features


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


def test_importance_populated_after_fit():
    """Phase 4: standardized coefficients + permutation IC drop must be
    present after fit() and their feature keys must match self._feature_names."""
    X, y = _synth_training_data(META_FEATURES, n=500, seed=7)
    mm = MetaModel(alpha=1.0).fit(X, y, feature_names=META_FEATURES)

    imp = mm._importance
    assert "standardized_coef" in imp
    assert "permutation" in imp
    assert "base_ic" in imp
    # Every feature must have both standardized and permutation values
    assert set(imp["standardized_coef"].keys()) == set(META_FEATURES)
    assert set(imp["permutation"].keys()) == set(META_FEATURES)
    # Standardized coef for the strongest synthetic driver (first column) must
    # be meaningfully larger than for a zero-weight column
    assert abs(imp["standardized_coef"][META_FEATURES[0]]) > abs(
        imp["standardized_coef"][META_FEATURES[-1]]
    )


def test_importance_roundtrip_through_save_load(tmp_path):
    """Importance metadata must survive save+load so the dashboard / email
    consumers can read it from the .meta.json sidecar without refitting."""
    X, y = _synth_training_data(META_FEATURES, n=300)
    mm = MetaModel(alpha=1.0).fit(X, y, feature_names=META_FEATURES)
    pkl_path = tmp_path / "mm.pkl"
    mm.save(pkl_path)

    meta = json.loads(Path(str(pkl_path) + ".meta.json").read_text())
    assert "importance" in meta
    assert set(meta["importance"]["standardized_coef"].keys()) == set(META_FEATURES)

    mm2 = MetaModel.load(pkl_path)
    assert mm2._importance == meta["importance"]


def test_importance_empty_on_tiny_dataset(tmp_path):
    """When n < 20 the model short-circuits fit without populating importance.
    Consumers must not assume the key is present and non-empty — serialization
    must still work."""
    X = np.random.default_rng(0).normal(size=(10, 3))
    y = np.random.default_rng(1).normal(size=10)
    mm = MetaModel(alpha=1.0).fit(X, y, feature_names=["a", "b", "c"])
    assert mm._importance == {}

def test_feature_names_embedded_in_pickle(tmp_path):
    """L4543: save() embeds feature_names IN the pickle (schema v2) so the
    load-bearing column order travels with the immutable model bytes."""
    X, y = _synth_training_data(META_FEATURES)
    mm = MetaModel(alpha=1.0).fit(X, y, feature_names=META_FEATURES)
    pkl_path = tmp_path / "mm.pkl"
    mm.save(pkl_path)

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    assert isinstance(payload, dict), "v2 pickle must be a schema dict, not a bare estimator"
    assert payload["_meta_model_schema"] >= 2
    assert payload["feature_names"] == list(META_FEATURES)


def test_stale_sidecar_cannot_corrupt_embedded_feature_names(tmp_path):
    """THE landmine fix (L4543): a registry promote can leave a PRIOR version's
    .pkl.meta.json sidecar next to the NEW model's .pkl. With feature_names
    embedded, load MUST use the embedded (authoritative) order and ignore the
    stale/mismatched sidecar — otherwise features feed in the wrong order →
    garbage predictions."""
    X, y = _synth_training_data(META_FEATURES)
    mm = MetaModel(alpha=1.0).fit(X, y, feature_names=META_FEATURES)
    pkl_path = tmp_path / "mm.pkl"
    mm.save(pkl_path)

    # Simulate a stale sidecar from a different model: SCRAMBLED feature order
    # + bogus reporting values.
    meta_path = Path(str(pkl_path) + ".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["feature_names"] = list(reversed(META_FEATURES))  # wrong order!
    meta["val_ic"] = 0.999  # bogus
    meta_path.write_text(json.dumps(meta))

    mm2 = MetaModel.load(pkl_path)
    # feature order comes from the pickle, NOT the scrambled sidecar
    assert mm2._feature_names == list(META_FEATURES)
    # reporting fields still read from the sidecar (cosmetic, non-load-bearing)
    assert mm2._val_ic == 0.999


def test_legacy_raw_model_pickle_falls_back_to_sidecar(tmp_path):
    """Backward-compat: a pre-v2 pickle is a BARE estimator (no embedded names);
    load must still source feature_names from the sidecar."""
    legacy_features = [f"legacy_feat_{i}" for i in range(6)]
    X, y = _synth_training_data(legacy_features)
    mm = MetaModel(alpha=1.0).fit(X, y, feature_names=legacy_features)

    # Write a LEGACY-format pickle: the bare estimator only (pre-L4543 save).
    pkl_path = tmp_path / "legacy_raw.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(mm._model, f)
    # ...plus a sidecar that carries the feature_names (the legacy contract).
    Path(str(pkl_path) + ".meta.json").write_text(
        json.dumps({"feature_names": legacy_features, "val_ic": 0.05, "coefficients": {}})
    )

    mm2 = MetaModel.load(pkl_path)
    assert mm2._feature_names == legacy_features
