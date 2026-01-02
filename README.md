# 🤖 AI Trainings: Introduction to LLM-Based Applications

Welcome to this beginner-friendly training series! These hands-on labs will teach you how to build real-world applications using Large Language Models (LLMs), Vector Databases, and Retrieval Augmented Generation (RAG).

##  What You'll Learn

By completing these labs, you will:
- Understand how to work with LLMs (Google Gemini)
- Master prompt engineering techniques
- Build and query vector databases (Qdrant)
- Create embeddings for semantic search
- Build a complete RAG application with a chat UI

##  Lab Overview

| Lab | Topic | Key Concepts |
|-----|-------|--------------|
| **Lab 1** | Marketing Content Generation | LLM basics, Prompt Engineering, Structured Output, Few-Shot Prompting |
| **Lab 2** | Vector Database & Similarity Search | Embeddings, Qdrant, Semantic Search, Metadata Filtering |
| **Lab 3** | RAG PDF Chatbot | Full RAG Pipeline, PDF Processing, Streamlit UI |

---

## 🧪 Lab 1: Marketing Content Generation

**Notebook:** `Lab1_Marketing_content_generation.ipynb`

Learn the fundamentals of working with Large Language Models through a marketing use case.

### Topics Covered:
-  Understanding LLMs and how they work
-  Basic prompting techniques
-  System instructions for persona control
-  Structured JSON output
-  Few-shot prompting with examples

### Prerequisites:
- Google Cloud account with Vertex AI enabled
- Basic Python knowledge

---

## 🧪 Lab 2: Vector Database & Similarity Search

**Notebook:** `Lab2_Qdrant_Vector_Database.ipynb`

Explore vector databases and semantic search using Qdrant Cloud.

### Topics Covered:
-  What are embeddings and why they matter
-  Creating and managing Qdrant collections
-  Uploading documents with embeddings
-  Similarity search (finding related content)
-  Metadata filtering for precise retrieval

### Prerequisites:
- Completed Lab 1
- Qdrant Cloud account (free tier available)

---

## 🧪 Lab 3: RAG PDF Chatbot

**Notebook:** `Lab3_RAG_PDF_Chatbot.ipynb`

Build a complete Retrieval Augmented Generation (RAG) application with a web interface.

### Topics Covered:
-  PDF text extraction and chunking
-  The RAG pipeline (Ingestion → Retrieval → Generation)
-  Building a chat interface with Streamlit
-  Tuning RAG parameters (chunk size, top-k)

### Prerequisites:
- Completed Labs 1 & 2
- All previous account requirements

---

##  Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AI_Trainings.git
cd AI_Trainings
```

### 2. Install Dependencies
```bash
pip install google-cloud-aiplatform qdrant-client sentence-transformers pandas PyPDF2 streamlit
```

### 3. Set Up Accounts

#### Google Cloud (for Gemini LLM)
1. Create a Google Cloud account
2. Enable Vertex AI API
3. Set up authentication (ADC or API key)

#### Qdrant Cloud (for Vector Database)
1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free cluster
3. Get your API key and cluster URL

### 4. Open the Labs
Open the notebooks in Jupyter, VS Code, or Google Colab and follow along!

---

##  Project Structure

```
AI_Trainings/
├── Lab1_Marketing_content_generation.ipynb   # LLM & Prompt Engineering
├── Lab2_Qdrant_Vector_Database.ipynb         # Vector DB & Embeddings
├── Lab3_RAG_PDF_Chatbot.ipynb                # Complete RAG Application
├── data/                                      # Sample data files
│   ├── softdrinks.csv                        # Product data for Lab 2
│   └── *.json                                # Product JSON files
├── images/                                    # Screenshots and diagrams
│   └── streamlit_demo.png                    # RAG chatbot demo
├── Solutions/                                 # Reference solutions
├── README.md                                  # This file
└── LICENSE                                    # License information
```

---

##  Technologies Used

| Technology | Purpose |
|------------|---------|
| **Google Gemini** | Large Language Model for text generation |
| **Vertex AI** | Google Cloud AI platform |
| **Qdrant** | Vector database for similarity search |
| **SentenceTransformers** | Creating text embeddings |
| **Streamlit** | Web UI framework |
| **PyPDF2** | PDF text extraction |
| **Pandas** | Data manipulation |

---

## 📖 Additional Resources

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

##  Contributing

Found an issue or have a suggestion? Feel free to open an issue or submit a pull request!

---

## 📄 License

This project is licensed under MIT LICENSE.

---

**Happy Learning! **
