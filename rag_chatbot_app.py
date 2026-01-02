# RAG PDF Chatbot - Streamlit Application
# Generated from Lab 3 Notebook

import os
import uuid
import streamlit as st
import google.generativeai as genai
import vertexai
from vertexai.language_models import TextEmbeddingModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from PyPDF2 import PdfReader

# ============================================
# CONFIGURATION (from notebook)
# ============================================
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.B0RB3KPvC_HSZ4wxZnSymZKrt6DZ_za45eKViBpdvIw"
QDRANT_URL = "https://d78b1147-cde0-4b94-aa1e-9b6b2278050c.us-east4-0.gcp.cloud.qdrant.io:6333"
MODEL_NAME = "gemini-2.0-flash-001"
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_OUTPUT = 8192
DEFAULT_COLLECTION = "pdfs_collection"

# Initialize Google AI
GOOGLE_API_KEY = "AIzaSyAQuY3ZgJuzUVHrDQFpAMeW7oNThvgug-U"
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    genai.configure()

# Initialize Vertex AI
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "agents4good")
vertexai.init(project=PROJECT_ID, location="us-central1")

# ============================================
# RAG FUNCTIONS
# ============================================
def init_qdrant(qdrant_url, qdrant_api_key):
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)

def create_qdrant_collection(collection_name, qdrant_url, qdrant_api_key, vector_size=768, distance="Cosine"):
    client = init_qdrant(qdrant_url, qdrant_api_key)
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance)
        )
    except:
        try:
            client.delete_collection(collection_name=collection_name)
        except:
            pass
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance)
        )

def ingest_pdfs_to_qdrant(pdf_files, collection_name, qdrant_url, qdrant_api_key, chunk_size=500):
    client = init_qdrant(qdrant_url, qdrant_api_key)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
        embeddings = embedding_model.get_embeddings(chunks)

        points = []
        for emb, chunk in zip(embeddings, chunks):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.values,
                payload={"text": chunk}
            ))
        client.upsert(collection_name=collection_name, points=points)

def get_bot_response(messages, model_name, temperature, top_p, max_output,
                     collection_name, qdrant_url, qdrant_api_key, k=3):
    history = ""
    for msg in messages:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        history += f"{speaker}: {msg['content']}\n"

    context = ""
    if collection_name and qdrant_url and qdrant_api_key:
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        query_embedding = embedding_model.get_embeddings([messages[-1]["content"]])[0].values
        client = init_qdrant(qdrant_url, qdrant_api_key)
        search_result = client.query_points(collection_name=collection_name, query=query_embedding, limit=k)
        hits = search_result.points
        context = "\n\n".join(hit.payload.get("text", "") for hit in hits)

    if context:
        prompt = f"""You are a helpful assistant. Use the following context to answer questions.

Context from documents:
{context}

Conversation:
{history}
Assistant:"""
    else:
        prompt = f"""You are a helpful assistant.

Conversation:
{history}
Assistant:"""

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={"temperature": temperature, "max_output_tokens": max_output, "top_p": top_p}
    )
    response = model.generate_content(prompt)
    return response.text

# ============================================
# STREAMLIT UI
# ============================================
st.set_page_config(page_title="My PDF RAG Chatbot", layout="wide", page_icon="")
st.title(" Chat with My PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qdrant_collection" not in st.session_state:
    st.session_state.qdrant_collection = None

uploaded_files = st.file_uploader(
    " Upload PDFs (optional — chat works even without PDFs)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and not st.session_state.qdrant_collection:
    with st.spinner("Creating Qdrant collection & ingesting PDFs…"):
        create_qdrant_collection(
            collection_name=DEFAULT_COLLECTION,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
        )
        ingest_pdfs_to_qdrant(
            pdf_files=uploaded_files,
            collection_name=DEFAULT_COLLECTION,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
        )
        st.session_state.qdrant_collection = DEFAULT_COLLECTION
        st.success(" PDFs ingested! Future replies will include RAG context.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = get_bot_response(
                messages=st.session_state.messages,
                model_name=MODEL_NAME,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output=MAX_OUTPUT,
                collection_name=st.session_state.qdrant_collection,
                qdrant_url=QDRANT_URL,
                qdrant_api_key=QDRANT_API_KEY,
            )
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
