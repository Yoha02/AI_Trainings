"""Exercise actual PDF retrieval and prevent accidental remote backend use."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lab_support as lab
from scripts import retrieval_walkthrough as walkthrough


def test_walkthrough_retrieves_evidence_without_google_or_hosted_qdrant(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This lesson must not use a remote backend or generate an answer.")

    monkeypatch.setenv("QDRANT_URL", "https://must-not-contact.invalid")
    monkeypatch.setattr(lab, "get_qdrant_client", forbidden)
    monkeypatch.setattr(lab.genai, "Client", forbidden)
    monkeypatch.setattr(lab.RagSession, "answer", forbidden)
    report = walkthrough.collect_evidence(walkthrough.QUESTIONS, k=1)
    returns, delivery, unsupported = report["cases"]
    assert returns["passages"][0]["page"] == 1
    assert "30 days" in returns["passages"][0]["text"]
    assert delivery["passages"][0]["page"] == 2
    assert "3 to 5 business days" in delivery["passages"][0]["text"]
    # Search still returns a passage even though this handbook gives no policy.
    assert unsupported["passages"]
    assert "no international shipping" in unsupported["passages"][0]["text"]
    assert all("answer" not in case for case in report["cases"])


@pytest.mark.parametrize("args", [
    ["--k", "0"], ["--chunk-size", "80", "--overlap", "80"],
    ["--overlap", "-1"], ["--question", "   "],
])
def test_bad_arguments_fail_before_loading_models(args, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid arguments should fail before loading dependencies.")
    monkeypatch.setattr(walkthrough, "collect_evidence", forbidden)
    with pytest.raises(SystemExit) as error:
        walkthrough.main(args)
    assert error.value.code == 2
