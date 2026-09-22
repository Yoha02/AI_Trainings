"""Run with: python -m streamlit run rag_chatbot_app.py"""
import hashlib
import streamlit as st
from lab_support import DEFAULT_MODEL, RagSession


def render_sources(sources):
    if sources:
        with st.expander("Retrieved sources: check that they support the answer"):
            for i, source in enumerate(sources, 1):
                st.write(f"[{i}] {source['source']}, page {source['page']}")
                st.write(source["text"])


st.set_page_config(page_title="Learn RAG: ask a PDF", layout="wide")
st.title("Ask a PDF")
st.caption("Lab 3: retrieve document passages, then ask Gemini for a grounded answer.")
st.info("PDF text stays local during indexing with the default configuration. Retrieved passages and your question are sent to your configured Google API when you ask. Use non-sensitive practice documents.")
st.caption(f"Generation model: {DEFAULT_MODEL}. Each question is answered independently.")
if "rag" not in st.session_state:
    st.session_state.rag = None
    st.session_state.messages = []
    st.session_state.document_signature = None
files = st.file_uploader("Choose text PDFs", type="pdf", accept_multiple_files=True)
signature = tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files)
if signature != st.session_state.document_signature:
    if st.session_state.rag:
        st.session_state.rag.close()
    st.session_state.rag = None
    st.session_state.messages = []
    st.session_state.document_signature = signature
if st.button("Index documents", disabled=not files):
    try:
        if st.session_state.rag:
            st.session_state.rag.close()
        st.session_state.rag = None
        st.session_state.messages = []
        session = RagSession()
        try:
            with st.spinner("Extracting text and building the search index..."):
                count = session.ingest(files)
        except Exception:
            session.close()
            raise
        st.session_state.rag = session
        st.success(f"Indexed {count} passages from {len(files)} document(s).")
    except Exception:
        st.error("Indexing failed. Check that the PDF contains selectable text and that the embedding model can download.")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))
question = st.chat_input("Ask about the indexed documents", disabled=st.session_state.rag is None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            result = st.session_state.rag.answer(question)
            st.markdown(result["answer"])
            render_sources(result["sources"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"],
                                              "sources": result["sources"]})
        except Exception:
            st.error("Generation failed. Check the backend, credentials, model availability, and quota using the README.")
if st.button("Clear session", disabled=st.session_state.rag is None):
    st.session_state.rag.close()
    st.session_state.rag = None
    st.session_state.messages = []
    st.rerun()
