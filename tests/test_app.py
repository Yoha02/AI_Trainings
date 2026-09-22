"""Check the learner interface without credentials or API calls."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_starts_with_generation_disabled():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "rag_chatbot_app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Ask a PDF"
    assert app.chat_input[0].disabled
    assert next(b for b in app.button if b.label == "Index documents").disabled


def test_earlier_answers_keep_their_sources_after_another_question():
    class ExampleRag:
        def answer(self, question):
            page = 2 if "delivery" in question else 1
            return {
                "answer": f"Example answer from page {page} [1].",
                "sources": [{"source": "handbook.pdf", "page": page,
                             "text": "Delivery takes 3 to 5 days." if page == 2 else "Returns within 30 days."}]
            }

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "rag_chatbot_app.py"))
    app.run(timeout=30)
    app.session_state["rag"] = ExampleRag()
    app.run()
    app.chat_input[0].set_value("What is the return window?").run()
    assert not app.exception
    assert len(app.expander) == 1

    app.chat_input[0].set_value("How long does delivery take?").run()
    assert not app.exception
    assert len(app.expander) == 2
    assert "Returns within 30 days." in [m.value for m in app.expander[0].markdown]
    assert "Delivery takes 3 to 5 days." in [m.value for m in app.expander[1].markdown]

    app.run()
    assert len(app.expander) == 2
    assert [message["sources"][0]["page"] for message in
            app.session_state["messages"] if message["role"] == "assistant"] == [1, 2]
