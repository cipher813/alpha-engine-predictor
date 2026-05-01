"""Verify flow-doctor wiring in predictor entrypoints.

Asserts the canonical alpha-engine-lib pattern (module-top setup_logging
+ exclude_patterns plumbed + yaml resolvable from the entrypoint
location) is in place for both predictor entrypoints:

- ``inference/handler.py``       — daily Lambda inference (LAMBDA_TASK_ROOT)
- ``training/train_handler.py``  — weekly EC2 spot training (no Lambda)

Also includes:
- LAMBDA_TASK_ROOT regression test: the Dockerfile flattens
  inference/handler.py to /var/task/inference/handler.py, so
  dirname(dirname(__file__)) → /var/task/ which is the right answer
  AND LAMBDA_TASK_ROOT is the explicit env-var path. Test simulates
  the env-var override.
- Dockerfile lock: confirm flow-doctor.yaml is COPYed into the image
  (the missing COPY was the v33-class regression in alpha-engine-data).

Runs without firing any LLM diagnosis: setup_logging is exercised with
FLOW_DOCTOR_ENABLED=1 + stub env vars + a redirected yaml store path,
but no ERROR records are emitted (so flow-doctor's report() / diagnose()
pipeline is never triggered — no Anthropic calls, no email, no GitHub
issue).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def stub_flow_doctor_env(monkeypatch):
    """Populate the env vars that flow-doctor.yaml's ${VAR} refs resolve.

    flow_doctor.init() substitutes these at load time. Stubs are non-empty
    strings; nothing actually contacts SMTP/GitHub since no report() fires.
    """
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    monkeypatch.setenv("EMAIL_SENDER", "test@example.com")
    monkeypatch.setenv("EMAIL_RECIPIENTS", "test@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "stub-password")
    monkeypatch.setenv("FLOW_DOCTOR_GITHUB_TOKEN", "stub-token")


@pytest.fixture
def reset_root_logger():
    """Snapshot + restore root logger handlers around each test."""
    root = logging.getLogger()
    saved = list(root.handlers)
    yield
    root.handlers = saved


@pytest.fixture
def temp_flow_doctor_yaml(tmp_path):
    """Write a copy of the production flow-doctor.yaml with store.path
    redirected into the test's tmp_path.

    The production yaml hardcodes /tmp/flow_doctor.db (Lambda ephemeral
    convention) which isn't writable in every CI/sandbox env. Tests that
    actually invoke flow_doctor.init() need a redirectable path.
    """
    import yaml as yamllib
    with open(REPO_ROOT / "flow-doctor.yaml") as f:
        cfg = yamllib.safe_load(f)
    cfg["store"]["path"] = str(tmp_path / "flow_doctor_test.db")
    yaml_path = tmp_path / "flow-doctor.yaml"
    with open(yaml_path, "w") as f:
        yamllib.safe_dump(cfg, f)
    return str(yaml_path)


def _flow_doctor_available() -> bool:
    try:
        import flow_doctor  # noqa: F401
        return True
    except ImportError:
        return False


flow_doctor_required = pytest.mark.skipif(
    not _flow_doctor_available(),
    reason="flow-doctor not installed (pip install alpha-engine-lib[flow_doctor])",
)


class TestFlowDoctorYamlPresence:
    """The yaml file each entrypoint resolves must exist at that path."""

    def test_inference_yaml_at_repo_root_exists(self):
        assert (REPO_ROOT / "flow-doctor.yaml").is_file()

    def test_training_yaml_at_repo_root_exists(self):
        assert (REPO_ROOT / "flow-doctor-training.yaml").is_file()

    def test_yaml_path_resolved_by_inference_handler_exists(self):
        # Mirrors inference/handler.py's local-dev path (LAMBDA_TASK_ROOT unset):
        #   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        handler_path = REPO_ROOT / "inference" / "handler.py"
        resolved = Path(os.path.dirname(os.path.dirname(os.path.abspath(handler_path)))) / "flow-doctor.yaml"
        assert resolved.is_file(), f"inference/handler.py resolves to {resolved}"

    def test_yaml_path_resolved_by_inference_handler_under_lambda_runtime(
        self, tmp_path, monkeypatch
    ):
        # Lambda flattens inference/handler.py to /var/task/inference/handler.py.
        # Two-dirs-up from /var/task/inference/handler.py = /var/task — same as
        # LAMBDA_TASK_ROOT — but the explicit env-var honor protects against
        # future Dockerfile flattening changes (e.g. predictor's path differs
        # from research's, where /var/task/handler.py would otherwise resolve
        # to /var/).
        fake_task_root = tmp_path / "fake_lambda_task_root"
        fake_task_root.mkdir()
        (fake_task_root / "flow-doctor.yaml").write_text("flow_name: test\n")
        monkeypatch.setenv("LAMBDA_TASK_ROOT", str(fake_task_root))
        resolved = os.path.join(
            os.environ.get(
                "LAMBDA_TASK_ROOT",
                "/should-not-fall-back",
            ),
            "flow-doctor.yaml",
        )
        assert os.path.isfile(resolved), (
            "inference/handler.py must honor LAMBDA_TASK_ROOT to be resilient "
            "to Dockerfile layout changes"
        )

    def test_yaml_path_resolved_by_training_handler_exists(self):
        # Training runs on EC2 spot (no LAMBDA_TASK_ROOT). The handler uses
        #   Path(__file__).resolve().parent.parent / "flow-doctor-training.yaml"
        train_path = REPO_ROOT / "training" / "train_handler.py"
        resolved = train_path.resolve().parent.parent / "flow-doctor-training.yaml"
        assert resolved.is_file(), f"training/train_handler.py resolves to {resolved}"


class TestFlowDoctorYamlSchema:
    """Both yamls must declare keys consistent with the lib contract."""

    def test_inference_yaml_has_required_top_level_keys(self):
        import yaml
        with open(REPO_ROOT / "flow-doctor.yaml") as f:
            cfg = yaml.safe_load(f)
        for key in ("flow_name", "repo", "notify", "store", "rate_limits"):
            assert key in cfg, f"missing top-level key: {key}"
        assert cfg["repo"] == "cipher813/alpha-engine-predictor"

    def test_training_yaml_has_required_top_level_keys(self):
        import yaml
        with open(REPO_ROOT / "flow-doctor-training.yaml") as f:
            cfg = yaml.safe_load(f)
        for key in ("flow_name", "repo", "notify", "store", "rate_limits"):
            assert key in cfg, f"missing top-level key: {key}"
        assert cfg["repo"] == "cipher813/alpha-engine-predictor"


@flow_doctor_required
class TestSetupLoggingAttach:
    """setup_logging() should attach FlowDoctorHandler when ENABLED=1.

    Does NOT fire any ERROR records, so flow-doctor's diagnose() / Anthropic
    calls are never invoked. Verifies wiring shape only.
    """

    def test_disabled_attaches_no_flow_doctor_handler(self, monkeypatch, reset_root_logger):
        monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "0")
        from alpha_engine_lib.logging import setup_logging
        setup_logging(
            "predictor-test-disabled",
            flow_doctor_yaml=str(REPO_ROOT / "flow-doctor.yaml"),
            exclude_patterns=[],
        )
        import flow_doctor
        attached = [h for h in logging.getLogger().handlers
                    if isinstance(h, flow_doctor.FlowDoctorHandler)]
        assert attached == []

    def test_enabled_attaches_flow_doctor_handler(
        self, stub_flow_doctor_env, reset_root_logger, temp_flow_doctor_yaml
    ):
        from alpha_engine_lib.logging import setup_logging, get_flow_doctor
        setup_logging(
            "predictor-test-enabled",
            flow_doctor_yaml=temp_flow_doctor_yaml,
            exclude_patterns=[],
        )
        import flow_doctor
        attached = [h for h in logging.getLogger().handlers
                    if isinstance(h, flow_doctor.FlowDoctorHandler)]
        assert len(attached) == 1
        assert get_flow_doctor() is not None

    def test_exclude_patterns_plumbed_to_handler(
        self, stub_flow_doctor_env, reset_root_logger, temp_flow_doctor_yaml
    ):
        from alpha_engine_lib.logging import setup_logging
        patterns = [r"yfinance possibly delisted", r"polygon transient 5\d\d"]
        setup_logging(
            "predictor-test-patterns",
            flow_doctor_yaml=temp_flow_doctor_yaml,
            exclude_patterns=patterns,
        )
        import flow_doctor
        attached = [h for h in logging.getLogger().handlers
                    if isinstance(h, flow_doctor.FlowDoctorHandler)]
        assert len(attached) == 1
        compiled = attached[0]._exclude_re
        assert [p.pattern for p in compiled] == patterns


class TestEntrypointModuleTopWiring:
    """Each entrypoint must call setup_logging at MODULE-TOP, not inside a
    function. Source-text checks; no flow_doctor.init() side effects.

    Was a real regression in the pre-PR-3 state — both handlers had
    setup_logging inside handler() / main(), so module-import errors
    (cold-start crashes, heavy GBM imports raising) bypassed flow-doctor.
    """

    @staticmethod
    def _index_of(needle: str, text: str) -> int:
        idx = text.find(needle)
        assert idx != -1, f"missing required text: {needle!r}"
        return idx

    @staticmethod
    def _strip_comments_and_docstrings(text: str) -> str:
        import re
        stripped = re.sub(r'"""[\s\S]*?"""', "", text)
        stripped = re.sub(r"^\s*#.*$", "", stripped, flags=re.MULTILINE)
        return stripped

    def test_inference_handler_calls_setup_logging_at_module_top(self):
        text = (REPO_ROOT / "inference" / "handler.py").read_text()
        setup_idx = self._index_of("setup_logging(", text)
        handler_def_idx = self._index_of("def handler(", text)
        assert setup_idx < handler_def_idx, (
            "setup_logging must be called at module-top, before def handler()"
        )
        assert "exclude_patterns=" in text[setup_idx:handler_def_idx]
        # Not duplicated inside handler() — strip comments/docstrings to
        # avoid false positives on commentary about the migration.
        body = self._strip_comments_and_docstrings(text[handler_def_idx:])
        assert "setup_logging(" not in body, (
            "duplicate setup_logging call inside handler() — should only run once at module-top"
        )

    def test_training_handler_calls_setup_logging_at_module_top(self):
        text = (REPO_ROOT / "training" / "train_handler.py").read_text()
        setup_idx = self._index_of("setup_logging(", text)
        main_def_idx = self._index_of("def main(", text)
        assert setup_idx < main_def_idx, (
            "setup_logging must be called at module-top, before def main()"
        )
        assert "exclude_patterns=" in text[setup_idx:main_def_idx]
        body = self._strip_comments_and_docstrings(text[main_def_idx:])
        assert "setup_logging(" not in body, (
            "duplicate setup_logging call inside main() — should only run once at module-top"
        )


class TestColdStartDeferral:
    """inference/handler.py must defer load_secrets() to first invocation
    via _ensure_init(), keeping cold-start under Lambda's 10s init wall.

    The pre-fixup state (PR #70 first commit) called load_secrets at
    module-top, which combined with module-top setup_logging +
    flow_doctor.init() + heavy LightGBM/CatBoost imports plausibly
    pushed cold-start over the 10s init timeout — manifested as
    canary version 72 returning statusCode=0, FunctionError= empty.
    Mirrors alpha-engine-research/lambda/handler.py's pattern.
    """

    @staticmethod
    def _strip_comments_and_docstrings(text: str) -> str:
        import re
        stripped = re.sub(r'"""[\s\S]*?"""', "", text)
        stripped = re.sub(r"^\s*#.*$", "", stripped, flags=re.MULTILINE)
        return stripped

    def test_inference_handler_does_not_call_load_secrets_at_module_top(self):
        text = (REPO_ROOT / "inference" / "handler.py").read_text()
        body = self._strip_comments_and_docstrings(text)
        ensure_def_idx = body.find("def _ensure_init(")
        assert ensure_def_idx != -1, "_ensure_init() must be defined at module-top"
        # Anything before _ensure_init's def must NOT call load_secrets()
        # (the import is fine; only the call is expensive).
        prefix = body[:ensure_def_idx]
        assert "load_secrets()" not in prefix, (
            "inference/handler.py calls load_secrets() at module-top; defer "
            "it inside _ensure_init() to keep cold-start under Lambda's "
            "10s init timeout"
        )

    def test_inference_handler_calls_ensure_init_first_in_handler(self):
        text = (REPO_ROOT / "inference" / "handler.py").read_text()
        handler_idx = text.find("def handler(")
        assert handler_idx != -1
        body = text[handler_idx:]
        ensure_call_idx = body.find("_ensure_init()")
        preflight_idx = body.find("PredictorPreflight(")
        assert ensure_call_idx != -1, (
            "handler() must call _ensure_init() to fire deferred SSM load"
        )
        assert ensure_call_idx < preflight_idx, (
            "_ensure_init() must run before PredictorPreflight (preflight "
            "uses ANTHROPIC_API_KEY etc. that load_secrets populates)"
        )


class TestDockerfileShipsFlowDoctorYaml:
    """The Lambda image must COPY flow-doctor.yaml into LAMBDA_TASK_ROOT.

    Pre-PR-3: the Dockerfile didn't COPY the yaml and setup_logging was
    only fired inside handler() at runtime — same v33-class regression
    that hit alpha-engine-data on PR #116. Hoisting setup_logging to
    module-top would surface this at cold-start without the COPY in
    place. Lock the COPY in to prevent future regressions.
    """

    def test_dockerfile_copies_flow_doctor_yaml(self):
        text = (REPO_ROOT / "Dockerfile").read_text()
        # Either explicit "COPY flow-doctor.yaml ..." or a wildcard COPY
        # that includes it; the literal path COPY is what we ship.
        assert "COPY flow-doctor.yaml" in text, (
            "Dockerfile must COPY flow-doctor.yaml into the image — "
            "module-top setup_logging requires the file at LAMBDA_TASK_ROOT"
        )
