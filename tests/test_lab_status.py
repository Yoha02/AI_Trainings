"""Prevent misleading validation claims from GitHub run metadata."""
import importlib.util
from pathlib import Path
from urllib.error import HTTPError

import pytest

SPEC = importlib.util.spec_from_file_location("lab_status", Path(__file__).resolve().parents[1] / "scripts/lab_status.py")
status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status)
SHA = "a" * 40


def run(number=1, **overrides):
    return dict(id=number, head_sha=SHA, event="push", head_branch="main",
                run_number=number, run_attempt=1, status="completed", conclusion="success", **overrides)


def fetch_runs(runs):
    def fetch(path):
        return {"sha": SHA} if path.startswith("/commits/") else {"workflow_runs": runs}
    return fetch


def test_clean_exact_revision_and_dirty_checkout():
    fetch = fetch_runs([run()])
    clean = status.inspect_revision(SHA, False, fetch)
    assert clean["checks_passed_for_this_checkout"] is True
    assert SHA in clean["download_url"]
    assert status.inspect_revision(SHA, True, fetch)["checks_passed_for_this_checkout"] is False
    assert status.inspect_revision("main", None, fetch)["checks_passed_for_this_checkout"] is False


def test_latest_run_takes_precedence_over_old_success():
    queued = {**run(2), "status": "queued", "conclusion": None}
    report = status.inspect_revision(SHA, False, fetch_runs([run(), queued]))
    assert report["check_status"] == "queued"
    assert report["checks_passed_for_this_checkout"] is False


def test_unrelated_commits_and_pr_runs_do_not_count():
    wrong_commit = {**run(), "head_sha": "b" * 40}
    pr = {**run(), "event": "pull_request"}
    branch = {**run(), "head_branch": "feature"}
    report = status.inspect_revision(SHA, False, fetch_runs([wrong_commit, pr, branch]))
    assert report["check_status"] == "no_run"
    assert report["checks_passed_for_this_checkout"] is False


def test_rate_limit_has_actionable_error(monkeypatch):
    def denied(*args, **kwargs):
        raise HTTPError("https://api.github.com", 403, "rate limited", {}, None)
    monkeypatch.setattr(status, "urlopen", denied)
    with pytest.raises(RuntimeError, match="Try later"):
        status.read_json("/commits/main")


def test_require_pass_rejects_local_edits(monkeypatch):
    monkeypatch.setattr(status, "local_revision", lambda: (SHA, True))
    monkeypatch.setattr(status, "inspect_revision", lambda *args: {
        "check_status": "success", "local_changes": True,
    })
    assert status.main(["--json", "--require-pass"]) == 1
