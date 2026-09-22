# Validation record: September 22, 2026

This record describes what was actually exercised during the lab refresh. It does not establish model quality, production readiness, or learner outcomes.

## Environment and reproducibility

- Windows, Python 3.13, a newly created virtual environment.
- Dependencies installed from `requirements.txt`; `python -m pip check` reported no broken requirements.
- Top-level package versions are pinned. Transitive dependencies and the embedding-model revision are not fully locked, so later installations can differ.

## Local checks: 11 passed

Command: `python -m pytest tests -q` from the repository root. Result: **11 passed in 16.43 seconds**.

- Every code cell in all three exercise notebooks and all three solution notebooks executed in order. This runs the Python cells, not the full Jupyter browser interface.
- Real CPU embeddings, product-vector insertion/filtering, local Qdrant queries, and sample-PDF extraction/retrieval ran.
- Google requests used the actual SDK with HTTP responses mocked. This checked system-instruction and schema serialization without billing or treating mock text as model output.
- Empty-PDF handling, chunk boundaries, invalid configuration, and grounded-request construction passed.
- The Streamlit test confirmed that the initial interface loads and disables indexing/chat until documents are supplied. It did not simulate the full browser file-upload interaction.

There were two dependency deprecation warnings (Google SDK/Pydantic and pypdf/cryptography). They did not fail these checks; dependency upgrades should be followed by the same validation.

## Live checks: four successful Gemini requests

Command: `python scripts/validate_live.py --confirm-live` with the Vertex AI backend, global location, and `gemini-3.5-flash-lite`.

| Check | Observed result |
|---|---|
| Basic generation | Returned product copy |
| Structured output | Parsed JSON with string `title` and `description` fields |
| Few-shot example | Returned `carrot -> vegetable` |
| PDF RAG | Returned a 30-day return window with a citation; retrieved evidence was page 1 of the sample handbook |

The actual text and retrieved source are in [the smoke-test record](validation/live-smoke-2026-09-22.json). Requests used a user-approved test project; no project ID or credentials are included in this public record. Initial attempts stopped at expired credentials, an invalid project identifier, and a disabled API before generation. After authentication and service setup were corrected, the four-request script completed successfully.

### A useful failure to discuss with learners

The first prompt supplied only "unsweetened sparkling water, 330 ml." The response added "completely calorie-free" and assumed a can. Those details were not explicitly supplied. The API smoke check passed because it received text, but this does **not** mean the copy passed a factual-support check.

Before publishing marketing content, separate supplied product facts from inferred or invented claims. Structured output constrains shape; it does not prove the contents. This is a useful next evaluation exercise for Lab 1.

## Limits and next checks

- Four requests are a smoke test, not a full live execution of all notebooks, an accuracy benchmark, or a cost/latency study.
- The Gemini Developer API backend and hosted Qdrant were not exercised live.
- PDF OCR, prompt-injection resistance, sensitive-data use, production session isolation, and citation correctness across a larger dataset have not been established.
- An independent learner walkthrough is still needed. Record confusing steps and incorporate corrections before describing the material as validated with learners.
