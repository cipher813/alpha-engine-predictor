"""AST-walk regression pin: freshness checks must use trading-day arithmetic.

Closes the cross-repo "calendar-day arithmetic in freshness checks" defect
class surfaced by the 2026-05-24 Sunday SF recovery: every post-Saturday
redrive trips a calendar-day gate even when the data carries the most
recent NYSE close. Lifted into ``krepis.dates`` (v0.27.0) as the
``trading_days_stale`` + ``is_fresh_in_trading_days`` chokepoint.

This test walks the repo's production Python and rejects calendar-day
arithmetic inside any function whose name signals freshness intent.

Rules:

  1. Walk all .py files under repo root except .venv/__pycache__/tests.
  2. For each ``FunctionDef`` whose lowercased name contains any of
     ``{fresh, stale, preflight, postflight}``, scan the function body
     for ``.days`` attribute access.
  3. Any hit fails the test unless the line carries the inline marker
     ``# noqa: trading-day`` (escape hatch for documented exceptions).

Allowlist sites (deliberate calendar-day; documented): none in this repo
— predictor's single freshness site (``_verify_arctic_fresh``) uses
trading-day arithmetic via ``last_closed_trading_day()`` and contains no
``.days`` access.
"""
from __future__ import annotations

import ast
import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FRESHNESS_NAME_PATTERN = re.compile(r"(fresh|stale|preflight|postflight)", re.IGNORECASE)

# Functions whose name matches the pattern but where calendar-day arithmetic
# is the correct semantic. Each entry is (filepath_glob, function_name).
ALLOWLIST: list[tuple[str, str]] = []


def _python_files() -> list[pathlib.Path]:
    skip_dirs = {".venv", "__pycache__", "tests", ".claude", ".git", "build"}
    out: list[pathlib.Path] = []
    for p in REPO_ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        out.append(p)
    return out


def _violations_in_file(path: pathlib.Path) -> list[tuple[str, int, str]]:
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    source_lines = source.splitlines()
    violations: list[tuple[str, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not FRESHNESS_NAME_PATTERN.search(node.name):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if any(rel.endswith(glob) and node.name == fname
               for glob, fname in ALLOWLIST):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Attribute) and inner.attr == "days"):
                lineno = inner.lineno
                source_line = (
                    source_lines[lineno - 1] if lineno - 1 < len(source_lines) else ""
                )
                if "# noqa: trading-day" in source_line:
                    continue
                violations.append((node.name, lineno, source_line.strip()))
    return violations


def test_no_calendar_days_in_freshness_functions():
    """Production freshness functions must not use ``.days`` calendar
    arithmetic. Use ``krepis.dates.{trading_days_stale,
    is_fresh_in_trading_days}`` instead, or add an inline
    ``# noqa: trading-day`` marker with a comment explaining why the
    calendar-day semantic is correct at that specific call site.
    """
    all_violations: list[str] = []
    for path in _python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for fname, lineno, src in _violations_in_file(path):
            all_violations.append(f"  {rel}:{lineno} (in {fname}): {src}")

    assert not all_violations, (
        "Calendar-day arithmetic found in freshness-named functions. Use "
        "krepis.dates.{trading_days_stale, is_fresh_in_trading_days} "
        "instead, or add `# noqa: trading-day` with a comment explaining "
        "why calendar days are correct at that site.\n"
        + "\n".join(all_violations)
    )


def test_allowlist_entries_are_real():
    """Every ALLOWLIST entry must point to an actual function."""
    import ast as _ast
    for glob, fname in ALLOWLIST:
        path = REPO_ROOT / glob
        assert path.exists(), f"Allowlist entry's file not found: {glob}"
        tree = _ast.parse(path.read_text(), filename=str(path))
        names = {
            n.name for n in _ast.walk(tree)
            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
        }
        assert fname in names, (
            f"Allowlist entry's function not found: {fname} in {glob}"
        )
