"""
TDD tests for the GitHub Actions workflow structure.
These validate the CI configuration before it ever runs in GitHub.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = (
    Path(__file__).parent.parent / ".github" / "workflows" / "python-app.yml"
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def test_workflow_file_exists():
    assert WORKFLOW_PATH.exists(), "python-app.yml must exist"


def test_workflow_triggers_on_push_and_pr(workflow):
    # PyYAML parses the bare `on:` key as boolean True
    on = workflow.get("on") or workflow.get(True)
    assert on is not None, "'on' trigger block missing"
    assert "push" in on
    assert "pull_request" in on


def test_workflow_has_lint_job(workflow):
    assert "lint" in workflow["jobs"], "A dedicated lint job is required"


def test_workflow_has_test_job(workflow):
    assert "test" in workflow["jobs"], "A dedicated test job is required"


def test_workflow_has_security_job(workflow):
    assert "security" in workflow["jobs"], "A security scanning job is required"


def test_lint_job_uses_ruff(workflow):
    steps = workflow["jobs"]["lint"]["steps"]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert "ruff" in run_cmds, "Lint job must use ruff, not flake8"
    assert "flake8" not in run_cmds, "flake8 must not appear in lint job"


def test_lint_job_runs_mypy(workflow):
    steps = workflow["jobs"]["lint"]["steps"]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert "mypy" in run_cmds, "Lint job must run mypy type checking"


def test_test_job_uses_matrix(workflow):
    strategy = workflow["jobs"]["test"].get("strategy", {})
    matrix = strategy.get("matrix", {})
    assert "python-version" in matrix, "Test job must use a Python version matrix"
    versions = matrix["python-version"]
    assert len(versions) >= 2, "Matrix must cover at least 2 Python versions"


def test_test_job_uses_uv(workflow):
    steps = workflow["jobs"]["test"]["steps"]
    uses_uv = any("setup-uv" in s.get("uses", "") for s in steps)
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert uses_uv or "uv" in run_cmds, "Test job must use uv for package management"


def test_test_job_uploads_coverage_artifact(workflow):
    steps = workflow["jobs"]["test"]["steps"]
    uploads = [s for s in steps if "upload-artifact" in s.get("uses", "")]
    assert uploads, "Test job must upload coverage as an artifact"


def test_test_job_fails_below_coverage_threshold(workflow):
    steps = workflow["jobs"]["test"]["steps"]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert "cov-fail-under" in run_cmds or "fail-under" in run_cmds, (
        "Test job must enforce a minimum coverage threshold"
    )


def test_security_job_uses_bandit(workflow):
    steps = workflow["jobs"]["security"]["steps"]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert "bandit" in run_cmds, "Security job must run bandit"


def test_python_version_is_at_least_311(workflow):
    versions = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    parsed = [tuple(int(x) for x in str(v).split(".")) for v in versions]
    assert all(v >= (3, 11) for v in parsed), (
        "All matrix versions must be >= 3.11 (project requirement)"
    )


def test_lint_job_checks_formatting(workflow):
    steps = workflow["jobs"]["lint"]["steps"]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert "ruff format" in run_cmds or "ruff check" in run_cmds, (
        "Lint job must check code formatting with ruff"
    )
