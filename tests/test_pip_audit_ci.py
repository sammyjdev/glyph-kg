"""Tests for pip-audit CI job (ecosystem security-gate rollout, phase 2, issue #14).

Acceptance criteria mapping:
- pip-audit step exists in ci.yml: test_pip_audit_step_present, test_pip_audit_installed
- pip/setuptools upgrade happens before pip-audit runs (avoids stale-bootstrap-pip
  false positives): test_pip_upgrade_before_pip_audit
- pip-audit invocation is blocking (no --exit-zero or other soft-fail flag):
  test_pip_audit_is_blocking
- pip-audit is not added as a project dependency in pyproject.toml (CI-only tool):
  folded into test_pip_audit_installed

Exempt: the job actually running green on GitHub Actions can't be verified by a
local unit test - verified manually instead (see maker report) by installing
deps + pip-audit in a scratch venv after the pip/setuptools upgrade and
confirming 0 findings.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"


def _pip_audit_invocation_lines(lines: list[str]) -> list[str]:
    """Lines that actually invoke pip-audit (not the `pip install pip-audit` step)."""
    return [
        line
        for line in lines
        if line.strip() == "pip-audit" or line.strip().startswith("pip-audit ")
    ]


def test_pip_audit_step_present() -> None:
    """The CI workflow must contain a step that runs pip-audit."""
    lines = CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
    assert _pip_audit_invocation_lines(lines), "no line invokes pip-audit"


def test_pip_upgrade_before_pip_audit() -> None:
    """pip/setuptools must be upgraded before pip-audit runs.

    A stale bootstrap pip/setuptools (as installed by `python -m venv`) is
    flagged by pip-audit with findings unrelated to this project's actual
    dependencies; upgrading first eliminates that noise.
    """
    lines = CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
    upgrade_idx = next(
        (i for i, line in enumerate(lines) if "pip install --upgrade pip setuptools" in line),
        None,
    )
    audit_lines = _pip_audit_invocation_lines(lines)
    assert upgrade_idx is not None, "no `pip install --upgrade pip setuptools` step in ci.yml"
    assert audit_lines, "no line invokes pip-audit"
    audit_idx = lines.index(audit_lines[0])
    assert audit_idx > upgrade_idx, (
        f"pip-audit (line {audit_idx}) must run after the pip/setuptools upgrade "
        f"(line {upgrade_idx})"
    )


def test_pip_audit_is_blocking() -> None:
    """pip-audit must be blocking: no --exit-zero or other soft-fail suffix."""
    lines = CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
    audit_lines = _pip_audit_invocation_lines(lines)
    assert audit_lines, "no line invokes pip-audit"
    for line in audit_lines:
        assert "--exit-zero" not in line, f"pip-audit must be blocking: {line.strip()}"
        assert "|| true" not in line, f"pip-audit must be blocking: {line.strip()}"


def test_pip_audit_installed() -> None:
    """pip-audit must be installed inline in CI, not added as a project dependency."""
    config = CI_WORKFLOW.read_text(encoding="utf-8")
    pyproject = PYPROJECT_TOML.read_text(encoding="utf-8")

    assert "pip install pip-audit" in config, "ci.yml must install pip-audit"
    assert "pip-audit" not in pyproject, (
        "pip-audit is a CI-only tool, it must not be added to pyproject.toml dependencies"
    )
