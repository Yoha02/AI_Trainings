"""Check the initial learner interface without credentials or API calls."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_starts_with_generation_disabled():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "rag_chatbot_app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Ask a PDF"
    assert app.chat_input[0].disabled
    assert next(b for b in app.button if b.label == "Index documents").disabled
