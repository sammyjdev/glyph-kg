"""Tests for .pre-commit-config.yaml (ruff lint + gitleaks secret scan).

Acceptance criteria mapping:
- AC1 (config exists, runs ruff + gitleaks): test_pre_commit_config_exists,
  test_ruff_hook_configured, test_gitleaks_hook_configured.
- AC2 (one-time setup documented): test_pre_commit_install_is_documented.
- AC3 (tier/scope only) has no runtime behavior to assert; exempt.

No YAML parser is used: PyYAML is not an existing dependency of this repo
(dev extras are pytest, pytest-cov, ruff, mypy, testcontainers) and the
config is small and fixed, so substring checks on the raw file catch drift
without adding a dependency for one test file.

Actually running `pre-commit run --all-files` was considered but the
gitleaks binary is unavailable in this offline environment (not on PATH,
and its pre-commit hook env would need to fetch it) - falling back to the
structural assertions below per the maker instructions.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"


def test_pre_commit_config_exists() -> None:
    assert PRE_COMMIT_CONFIG.is_file()


def test_ruff_hook_configured() -> None:
    config = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    assert "repo: https://github.com/astral-sh/ruff-pre-commit" in config
    assert "id: ruff" in config


def test_gitleaks_hook_configured() -> None:
    config = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    assert "repo: https://github.com/gitleaks/gitleaks" in config
    assert "id: gitleaks" in config


def test_pre_commit_install_is_documented() -> None:
    lines: list[str] = []
    for name in ("README.md", "CONTRIBUTING.md"):
        path = REPO_ROOT / name
        if path.is_file():
            lines += path.read_text(encoding="utf-8").splitlines()
    assert any(line.strip() == "pre-commit install" for line in lines)
