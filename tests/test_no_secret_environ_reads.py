"""Regression: no module in this repo reads a secret via ``os.environ.get``.

After the 2026-05-12 ``.env`` → SSM migration (PR 4 of the arc), every
secret-bearing call site routes through ``krepis.secrets.get_secret()``.
This test re-greps the codebase on every CI run so a future commit can't
silently re-introduce an ``os.environ.get("POLYGON_API_KEY")`` style read.

Non-secret env vars (``LANGCHAIN_PROJECT``, ``EMAIL_SENDER``,
``PREDICTOR_PARAMS_CACHE``, etc.) are allowed for now — they migrate to
alpha-engine-config YAML in PR 8 of the arc.

``ssm_secrets.py`` is allowlisted — it's the per-repo bulk-load shim
that stays alive until PR 9 of the arc.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_PINNED_SECRETS = frozenset(
    [
        "ANTHROPIC_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
        "VOYAGE_API_KEY",
        "POLYGON_API_KEY",
        "FMP_API_KEY",
        "FINNHUB_API_KEY",
        "FRED_API_KEY",
        "GMAIL_APP_PASSWORD",
        "GITHUB_TOKEN",
        "RAG_DATABASE_URL",
        "EDGAR_IDENTITY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "EMAIL_SENDER",
        "EMAIL_RECIPIENTS",
    ]
)

_ALLOWED_FILES = frozenset(["ssm_secrets.py"])

_ENV_READ_RE = re.compile(
    r'os\.(?:environ\.get|getenv)\(\s*["\']([A-Z_][A-Z0-9_]*)["\']'
)


def _iter_python_files():
    for path in _REPO_ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {".venv", "build", "tests", "node_modules", "package"}:
            continue
        if path.name in _ALLOWED_FILES:
            continue
        yield path


def test_no_secret_environ_reads():
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_python_files():
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _ENV_READ_RE.finditer(line):
                name = match.group(1)
                if name in _PINNED_SECRETS:
                    violations.append((path.relative_to(_REPO_ROOT), lineno, name))
    assert not violations, (
        "Found os.environ.get reads of pinned secrets — use "
        "`from krepis.secrets import get_secret` instead:\n"
        + "\n".join(f"  {p}:{ln}  {name}" for p, ln, name in violations)
    )
