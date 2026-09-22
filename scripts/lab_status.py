"""AgenticWorks lab companion: inspect public GitHub checks for an exact revision.

Uses only Python's standard library. No GitHub token or Google request is needed.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

REPOSITORY = "Yoha02/AI_Trainings"
ROOT = Path(__file__).resolve().parents[1]
API = f"https://api.github.com/repos/{REPOSITORY}"


def read_json(path):
    request = Request(API + path, headers={
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
        "User-Agent": "AgenticWorks-lab-companion",
    })
    try:
        with urlopen(request, timeout=15) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code in (403, 429):
            raise RuntimeError("GitHub denied or rate-limited this public request. Try later; no token is required.") from exc
        if exc.code == 404:
            raise RuntimeError("This revision or workflow is not available in the public lab repository.") from exc
        raise RuntimeError(f"GitHub returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError("Could not read GitHub's response. Check connectivity and try later.") from exc


def local_revision():
    try:
        def git(*args):
            return subprocess.run(
                ["git", *args], cwd=ROOT, check=True, capture_output=True,
                text=True, timeout=10,
            ).stdout.strip()
        # Prevent accidentally inspecting a parent repository when using a ZIP.
        if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT.resolve():
            raise RuntimeError("This directory is not a Git checkout. Use --ref main to inspect the published labs.")
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot read the local Git revision. Use --ref main to inspect the published labs.") from exc


def inspect_revision(ref, dirty=None, fetch=read_json):
    commit = fetch("/commits/" + quote(ref, safe=""))
    sha = commit.get("sha", "")
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise RuntimeError("GitHub did not return a full commit identifier.")
    query = urlencode({"head_sha": sha, "event": "push", "branch": "main", "per_page": 100})
    payload = fetch("/actions/workflows/labs.yml/runs?" + query)
    runs = [run for run in payload.get("workflow_runs", [])
            if run.get("head_sha") == sha and run.get("event") == "push"
            and run.get("head_branch") == "main"]
    # A new/in-progress run must supersede an older successful run.
    run = max(runs, key=lambda item: (item.get("run_number", 0), item.get("run_attempt", 0)), default=None)
    status = "no_run"
    if run:
        status = run.get("conclusion") if run.get("status") == "completed" else run.get("status")
        status = status or "unknown"
    return {
        "repository": REPOSITORY,
        "revision": sha,
        "check_status": status,
        "local_changes": dirty,
        "checks_passed_for_this_checkout": status == "success" and dirty is False,
        "source_url": f"https://github.com/{REPOSITORY}/tree/{sha}",
        "download_url": f"https://github.com/{REPOSITORY}/archive/{sha}.zip",
        "run_url": f"https://github.com/{REPOSITORY}/actions/runs/{int(run['id'])}" if run else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Inspect a published branch/tag/SHA instead of the local checkout (for example: main).")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    parser.add_argument("--require-pass", action="store_true", help="Exit 1 unless the revision passed and there are no known local changes.")
    args = parser.parse_args(argv)
    try:
        ref, dirty = (args.ref, None) if args.ref else local_revision()
        report = inspect_revision(ref, dirty)
    except (RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"Lab status unavailable: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"AgenticWorks lab companion: {REPOSITORY}")
        print(f"Revision: {report['revision']}")
        print(f"GitHub lab checks: {report['check_status']}")
        if dirty:
            print("Local changes detected: remote checks do not validate these edits.")
        elif dirty is None:
            print("Inspecting a published revision; local files were not assessed.")
        else:
            print("Local checkout is clean.")
        print(f"Source: {report['source_url']}")
        print(f"Download this revision: {report['download_url']}")
        print(f"Check details: {report['run_url'] or 'No main-branch push run found for this revision.'}")
        print("Checks use mocked Google responses: a pass does not establish model quality.")
    if args.require_pass and (report["check_status"] != "success" or dirty is True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
