# AI Trainings: build a Gemini application, step by step

Three Python labs connect prompting, vector search, and a PDF question-answering app. Start with a small example, inspect what the system sends and retrieves, then change one thing and compare the result.

| Lab | Build and inspect | Expected checkpoint |
|---|---|---|
| [1: Marketing content](Lab1_Marketing_content_generation.ipynb) | Google Gen AI SDK, system instructions, JSON schemas, few-shot prompts | Generate text and parse JSON with the requested fields. Check every product claim yourself. |
| [2: Vector search](Lab2_Qdrant_Vector_Database.ipynb) | Local embeddings, Qdrant, product metadata filters | Retrieve similar products and verify the returned brand/category. No Google request is needed. |
| [3: PDF RAG](Lab3_RAG_PDF_Chatbot.ipynb) | Extract pages, chunk text, retrieve evidence, ask Gemini, inspect citations | The included handbook supports a 30-day return window on page 1. |

Each lab has a [reference solution](Solutions/). The notebooks share [readable building blocks](lab_support.py) with the [Streamlit app](rag_chatbot_app.py).

## Start locally

Basic Python knowledge is enough. The initial local validation used **Python 3.13 on Windows**. Automated checks now target Windows and Ubuntu with Python 3.13; inspect the [workflow results](https://github.com/Yoha02/AI_Trainings/actions/workflows/labs.yml) for your revision. macOS has not been validated. Allow several GB for dependencies, including PyTorch. The first embedding run downloads the public `all-MiniLM-L6-v2` model and caches it. A GPU is not required.

```sh
git clone https://github.com/Yoha02/AI_Trainings.git
cd AI_Trainings
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` in place of `python`; changing your execution policy is unnecessary.

```sh
python -m pip install -r requirements.txt
python -m pip check
```

## Choose one Google backend

Lab 2 runs without Google credentials. Labs 1 and 3 send prompts to Google and can incur API charges. Use practice data, review billing and quota, and avoid repeatedly running all solution experiments without checking usage.

The default model is `gemini-3.5-flash-lite`. Availability depends on backend and project; override it with `GEMINI_MODEL` if needed. See [Google's model documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-flash-lite).

### Option A: Vertex AI with a Google Cloud project

Use a project with billing and the Vertex AI API enabled, and an account authorized to invoke models. [Google's local authentication guide](https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment) explains Application Default Credentials (ADC).

```sh
gcloud auth application-default login
```

Set these in the **same terminal** that will start Jupyter or Streamlit:

```powershell
$env:LAB_BACKEND = "vertex"
$env:GOOGLE_CLOUD_PROJECT = "YOUR_PROJECT_ID"
$env:GOOGLE_CLOUD_LOCATION = "global"
$env:GEMINI_MODEL = "gemini-3.5-flash-lite"
```

```sh
export LAB_BACKEND=vertex
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
export GOOGLE_CLOUD_LOCATION=global
export GEMINI_MODEL=gemini-3.5-flash-lite
```

`gcloud auth login` and ADC are separate credentials. For a reauthentication error, refresh ADC. A 403 needs a check of project permissions/API access; a 404 can mean the model or location is unavailable. Do not switch billing projects by guesswork.

### Option B: Gemini Developer API

Create a key in [Google AI Studio](https://aistudio.google.com/apikey), then set `LAB_BACKEND=gemini` and `GEMINI_API_KEY` in your terminal environment. Use PowerShell `$env:NAME = "value"` or shell `export NAME=value`, as above. Do not paste a real key into a notebook or commit it.

[.env.example](.env.example) documents variable names. These labs **do not automatically load .env files**. Restart Jupyter after changing environment variables so kernels inherit them.

## Run the lessons and app

From the repository root:

```sh
python -m notebook
```

Open a lab and run code cells in order. Notebooks find the repository root even from `Solutions/`. Hosted notebooks need equivalent file, dependency, and credential setup; this revision validates a local checkout.

Lab 3 includes [a two-page fictional shop handbook](data/sample_handbook.pdf) and [plain-text source](data/sample_handbook.txt). Try:

- **How long is the return window?** Check for 30 days and page 1.
- **How long does standard delivery take?** Check for 3–5 business days and page 2.
- **What is the international shipping policy?** The handbook does not specify one. A grounded answer should say so.

Inspect retrieved passages before judging an answer. Similarity scores and citations do not guarantee factual correctness. Scanned PDFs need OCR, which is outside this lab.

For the interface, open another terminal with the same environment:

```sh
python -m streamlit run rag_chatbot_app.py
```

Upload the sample PDF, select **Index documents**, then ask a question. Changing documents clears the old index and chat. Each question is independent: displayed history is not sent as conversational memory.

## Storage and cleanup

Qdrant runs **in memory by default**: no account or server needed. Embeddings run locally on CPU. Each lab run/session uses a unique collection. Call `rag.close()` when finished with Lab 3, or **Clear session** in the app. Restart the notebook kernel to release its other local resources.

Optional hosted Qdrant: set `QDRANT_URL` and `QDRANT_API_KEY`. This sends indexed text, metadata, and vectors to that service. Clean up collections you create, including those left after interrupted runs. Never use someone else's collection for these exercises.

## Validation

The [AgenticWorks lab companion](docs/lab-companion.md) checks your source revision against public GitHub Actions results and provides an exact source/download link. It works before installing dependencies and needs no account or token:

```sh
python scripts/lab_status.py
```

The [automated lab checks](https://github.com/Yoha02/AI_Trainings/actions/workflows/labs.yml) run on pull requests and updates to `main`. Inspect the run for your revision to see the actual result; an edited checkout is not covered by a remote pass.

To check your local files:

```sh
python -m pytest tests -q
```

Tests execute the code cells of all six notebooks in sequence with real local embeddings, PDF extraction, and Qdrant. Google HTTP responses are mocked through the actual SDK: this checks serialization and notebook flow, **not live model quality**. The initial model download requires network access.

An opt-in script sends four small requests to your configured Google backend:

```sh
python scripts/validate_live.py --confirm-live
```

It checks generation, JSON parsing, a few-shot response, and an answer grounded in the sample PDF. It incurs API usage and is a smoke check, not a benchmark. See [the dated validation record](VALIDATION.md) for actual coverage and limitations.

## Extend one experiment

Change chunk size or retrieved passage count in Lab 3. Record the question, retrieved pages, answer, and whether the evidence supports it. Include an unanswerable question. If retrieval misses the relevant page, repair retrieval before changing the generation prompt.

Open an issue with Python version, lab/cell, expected result, and the error with credentials removed. Improvements to instructions and examples are welcome.

Contributors include [Yoha02](https://github.com/Yoha02) and [renoschubert](https://github.com/renoschubert). See repository history for authorship. Licensed under [MIT](LICENSE).

References: [Google Gen AI SDK](https://googleapis.github.io/python-genai/), [Qdrant](https://qdrant.tech/documentation/), [Sentence Transformers](https://www.sbert.net/), [Streamlit](https://docs.streamlit.io/).
