"""Opt-in validation: four small Google requests using the configured backend."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from google.genai import types
from lab_support import ROOT, DEFAULT_MODEL, get_client, RagSession

def main():
    if "--confirm-live" not in sys.argv:
        raise SystemExit("This sends four requests to your configured Google backend and may incur charges. "
                         "Read the README, then add --confirm-live.")
    client = get_client()
    reports = []
    try:
        prompts = [
            ("Write one factual sentence advertising this fictional product: unsweetened sparkling water, 330 ml.", None),
            ("Return the title and description of this fictional product: unsweetened sparkling water, 330 ml.",
             {"type": "OBJECT", "properties": {"title": {"type": "STRING"},
                "description": {"type": "STRING"}}, "required": ["title", "description"]}),
            ("Example: apple -> fruit. Classify carrot using the same format.", None)]
        for prompt, schema in prompts:
            response = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=512,
                    response_mime_type="application/json" if schema else None, response_schema=schema))
            assert response.text, "No model text returned"
            if schema:
                parsed = json.loads(response.text)
                assert isinstance(parsed.get("title"), str) and isinstance(parsed.get("description"), str)
            reports.append({"check": "structured" if schema else "generation", "response": response.text})
        rag = RagSession(client=client)
        try:
            rag.ingest([ROOT / "data/sample_handbook.pdf"])
            result = rag.answer("How long is the return window?", k=1)
            assert "30" in result["answer"] and result["sources"][0]["page"] == 1
            reports.append({"check": "rag", **result})
        finally:
            rag.close()
    finally:
        client.close()
    print(json.dumps({"model": DEFAULT_MODEL, "checks": reports}, indent=2))

if __name__ == "__main__":
    main()
