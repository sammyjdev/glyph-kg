"""Tests for gitleaks full-history CI job (ecosystem security-gate rollout, phase 2).

Acceptance criteria mapping:
- gitleaks job exists in ci.yml: test_gitleaks_job_present
- job uses fetch-depth: 0 for full history scan: test_gitleaks_fetch_depth_zero
- job uses gitleaks/gitleaks-action: test_gitleaks_action_used
- gitleaks invocation is blocking (no continue-on-error or soft-fail):
  test_gitleaks_is_blocking
- gitleaks is not added as a project dependency in pyproject.toml (CI-only tool):
  test_gitleaks_not_a_project_dependency
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"


def _gitleaks_job_block(content: str) -> str | None:
    """Return the YAML block for the gitleaks/secret-scan job, or None."""
    lines = content.splitlines()
    job_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("gitleaks:", "secret-scan:"):
            job_start = i
            break
    if job_start is None:
        return None
    job_end = len(lines)
    for i in range(job_start + 1, len(lines)):
        if lines[i] and not lines[i].startswith(" ") and not lines[i].startswith("\t"):
            job_end = i
            break
    return "\n".join(lines[job_start:job_end])


def test_gitleaks_job_present() -> None:
    content = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "gitleaks" in content, "no gitleaks mention in ci.yml"
    block = _gitleaks_job_block(content)
    assert block is not None, "no gitleaks or secret-scan job definition in ci.yml"


def test_gitleaks_fetch_depth_zero() -> None:
    block = _gitleaks_job_block(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert block is not None, "gitleaks job not found"
    assert "fetch-depth: 0" in block, "gitleaks job must use fetch-depth: 0"


def test_gitleaks_action_used() -> None:
    block = _gitleaks_job_block(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert block is not None, "gitleaks job not found"
    assert "gitleaks/gitleaks-action" in block, "gitleaks job must use gitleaks/gitleaks-action"


def test_gitleaks_is_blocking() -> None:
    content = CI_WORKFLOW.read_text(encoding="utf-8")
    block = _gitleaks_job_block(content)
    assert block is not None, "gitleaks job not found"
    assert "continue-on-error" not in block, "gitleaks job must not use continue-on-error"
    assert "|| true" not in content, "gitleaks job must not use || true soft-fail"


def test_gitleaks_not_a_project_dependency() -> None:
    pyproject = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert "gitleaks" not in pyproject, (
        "gitleaks is a CI-only tool, must not be in pyproject.toml dependencies"
    )
