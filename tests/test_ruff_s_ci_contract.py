"""Discriminating configuration tests for issue #13."""

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _ruff_config() -> dict:
    return tomllib.loads(PYPROJECT_TOML.read_text(encoding="utf-8"))["tool"]["ruff"]["lint"]


def test_s_is_in_default_ruff_selection() -> None:
    assert "S" in _ruff_config()["select"]


def test_ci_has_only_the_main_blocking_ruff_check() -> None:
    commands = [
        line.strip().removeprefix("run:").strip()
        for line in CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
        if "ruff check" in line
    ]
    assert commands == ["ruff check glyph tests"]


def test_s101_remains_ignored_only_for_pytest_files() -> None:
    ignores = _ruff_config()["per-file-ignores"]
    assert "S101" in ignores["tests/**"]


def test_s603_is_ignored_for_tests_subprocess_calls() -> None:
    ignores = _ruff_config()["per-file-ignores"]
    assert "S603" in ignores["tests/**"]


def test_s101_comment_explains_pytest_assert_usage() -> None:
    lines = PYPROJECT_TOML.read_text(encoding="utf-8").splitlines()
    entry = next(i for i, line in enumerate(lines) if line.startswith('"tests/**"'))
    comment = " ".join(line.lower() for line in lines[:entry] if line.startswith("#"))
    assert "s101" in comment and "assert" in comment and "pytest" in comment


def test_default_ruff_selection_blocks_an_s_finding(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.py"
    unsafe.write_text('eval("1 + 1")\n', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            str(unsafe),
            "--config",
            str(PYPROJECT_TOML),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "S307" in result.stdout


def test_default_ruff_check_is_green() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "glyph", "tests"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
