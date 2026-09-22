"""Real local retrieval and SDK serialization; Google HTTP responses are mocked."""
from pathlib import Path
import json
import sys
from unittest.mock import patch
import httpx
import nbformat
import pytest
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lab_support as lab

def schema_example(schema):
    kind = schema.get("type", "STRING").upper()
    if kind == "ARRAY":
        return [schema_example(schema["items"])]
    if kind == "OBJECT":
        return {k: schema_example(v) for k, v in schema.get("properties", {}).items()}
    if kind in ("NUMBER", "INTEGER"):
        return 1
    if kind == "BOOLEAN":
        return True
    return "MOCK: inspect the real model output in live mode."

@pytest.fixture
def mock_client():
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert "contents" in body
        schema = body.get("generationConfig", {}).get("responseSchema")
        text = json.dumps(schema_example(schema)) if schema else "MOCK Google response [1]."
        return httpx.Response(200, json={"candidates": [{"content": {
            "role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]})
    client = genai.Client(api_key="unit-test-placeholder", http_options=types.HttpOptions(
        client_args={"transport": httpx.MockTransport(respond)}))
    yield client, requests
    client.close()

def test_explicit_auth_configuration(monkeypatch):
    monkeypatch.setenv("LAB_BACKEND", "vertex")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        lab.get_client()
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "unit-test-project")
    with patch.object(lab.genai, "Client") as constructor:
        lab.get_client()
        assert constructor.call_args.kwargs["vertexai"] is True
        assert constructor.call_args.kwargs["project"] == "unit-test-project"
    monkeypatch.setenv("LAB_BACKEND", "invalid")
    with pytest.raises(ValueError, match="LAB_BACKEND"):
        lab.get_client()

def test_chunk_boundaries_and_invalid_overlap():
    assert lab.split_text("abcdefghij", 6, 2) == ["abcdef", "efghij"]
    assert lab.split_text("", 6, 2) == []
    for size, overlap in [(0, 0), (5, 5), (5, -1)]:
        with pytest.raises(ValueError):
            lab.split_text("abc", size, overlap)

def test_pdf_retrieval_and_grounded_request(mock_client):
    client, requests = mock_client
    rag = lab.RagSession(client=client)
    try:
        assert rag.answer("Return window?")["sources"] == []
        assert requests == []
        assert rag.ingest([ROOT / "data/sample_handbook.pdf"]) > 0
        result = rag.answer("How many days do I have to return an unused item?", k=1)
        assert result["sources"][0]["page"] == 1
        assert "30 days" in result["sources"][0]["text"]
        body = requests[-1]
        assert "30 days" in json.dumps(body["contents"])
        assert "say you do not know" in json.dumps(body["systemInstruction"])
        assert "sample_handbook.pdf" in json.dumps(body["contents"])
    finally:
        rag.close()

def test_empty_pdf_fails_without_generation(tmp_path, mock_client):
    from pypdf import PdfWriter
    p = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with p.open("wb") as f:
        writer.write(f)
    client, requests = mock_client
    rag = lab.RagSession(client=client)
    try:
        with pytest.raises(ValueError, match="No extractable text"):
            rag.ingest([p])
        assert requests == []
    finally:
        rag.close()

NOTEBOOKS = sorted(ROOT.glob("*.ipynb")) + sorted((ROOT / "Solutions").glob("*.ipynb"))

@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_complete_notebook_code(notebook, mock_client, monkeypatch):
    client, requests = mock_client
    monkeypatch.chdir(notebook.parent)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setattr(lab, "get_client", lambda: client)
    document = nbformat.read(notebook, as_version=4)
    nbformat.validate(document)
    namespace = {"__name__": "__main__"}
    try:
        for index, cell in enumerate(document.cells):
            if cell.cell_type == "code":
                exec(compile(cell.source, f"{notebook.name}:cell{index}", "exec"), namespace)
        if notebook.name.startswith("Lab1"):
            assert len(requests) >= 3
            assert all("systemInstruction" in r for r in requests)
        elif notebook.name.startswith("Lab2"):
            assert namespace["hits"]
            assert requests == []
        else:
            assert requests
            assert namespace["result"]["sources"]
    finally:
        if "rag" in namespace:
            namespace["rag"].close()
        if "qdrant_client" in namespace:
            namespace["qdrant_client"].delete_collection(namespace["collection_name"])
            namespace["qdrant_client"].close()
