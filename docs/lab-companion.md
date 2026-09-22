# AgenticWorks lab companion

Before debugging a lesson, check which source revision you are using and open the automated check results for that same revision.

From a clone of this repository:

```sh
python scripts/lab_status.py
```

This small GitHub API integration uses only Python's standard library, so you can run it **before installing the lab dependencies**. It reads the local Git revision and asks GitHub for public commit and workflow-run metadata. It prints an exact source link, a ZIP download for that revision, and the lab-check result.

If you downloaded a ZIP, or want to inspect the latest published version:

```sh
python scripts/lab_status.py --ref main
```

This mode inspects the published commit; it does not assess your local files. For a machine-readable report, add `--json`. Add `--require-pass` to return exit code 1 when checks have not passed or the inspected checkout has local changes. Normal inspection returns 0; an API or Git error returns 2.

## Read the result

| Result | Meaning and next step |
|---|---|
| `success` and a clean local checkout | The main-branch lab checks passed for this exact source revision. Open the run link to see the environments and checks. |
| Local changes detected | Remote checks describe the committed version. Run the local tests to check your edits. |
| `queued` or `in_progress` | GitHub is still running the checks. Follow the run link. |
| `failure`, `cancelled`, or another conclusion | Inspect the run before relying on that revision. A cancelled run is not a pass. |
| `no_run` | No matching main-branch push run was found. Older commits and unmerged branches may have none; this does not establish failure or success. |
| Status unavailable | Connectivity, API limits, or an unavailable revision/workflow prevented a result. Retry later or inspect the repository directly. |

## What the checks establish

The [Lab checks workflow](../.github/workflows/labs.yml) executes the notebook and regression suite on Windows and Ubuntu with Python 3.13. It installs dependencies and downloads the public embedding model. Google responses are mocked; the workflow has no Google credentials and does not invoke live generation. A green result establishes that those automated checks passed. It does not establish model quality, production readiness, or independent learner validation.

The companion selects the latest run for the **exact commit**, rather than borrowing a passing result from a different version. It does not treat edited local files as remotely validated.

## Data and API usage

Requests go only to the public `Yoha02/AI_Trainings` GitHub API endpoints. The queried revision and ordinary request metadata reach GitHub. Local file contents, credentials, notebook outputs, and the changed-file list are not uploaded. No token, OAuth grant, paid service, or Google API call is needed. Unauthenticated GitHub rate limits apply; the tool makes two requests per successful inspection and does not poll.

Implementation: [scripts/lab_status.py](../scripts/lab_status.py). API reference: [GitHub workflow runs](https://docs.github.com/en/rest/actions/workflow-runs#list-workflow-runs-for-a-workflow). For help, [open an issue](https://github.com/Yoha02/AI_Trainings/issues/new) with the revision and error text, after removing credentials.
