📘 Multi-Modal RAG Question-Answering System

A Retrieval-Augmented Generation (RAG) system built for querying complex PDF documents containing text, tables, figures, and metadata.
Live Demo → https://multimodelrag-fdwwe2ev2afwkrlbdwwny5.streamlit.app/

🚀 Overview

This project implements a multi-modal Retrieval-Augmented Generation (RAG) pipeline capable of answering user questions strictly using information extracted from a complex PDF, such as IMF country reports.

The system performs:

PDF extraction (text + tables + structured content)

Chunking into semantically meaningful segments

Embedding using Sentence-BERT (local) or TF-IDF fallback

Vector search using FAISS when available

Reranking with TF-IDF cross-similarity

Strict evidence-based prompting to prevent hallucinations

LLM answer generation using Gemini Flash

Verification step to block unsupported answers

Streamlit UI for an interactive search experience

The entire pipeline is optimized for high accuracy retrieval, low hallucination, and cloud-friendly execution.

🧩 System Architecture
1️⃣ PDF Extraction

Located in src/extract.py

Extracts text, tables, and structural elements

Saves chunk metadata into data/meta.jsonl

Saves raw chunks into data/pages.jsonl

2️⃣ Chunking

Located in src/chunks.py

Splits long PDF segments into short, overlapping, context-preserving chunks

Handles tables, paragraphs, and captions

3️⃣ Embeddings

Located in src/embeddings.py

Generates embeddings using:

SentenceTransformer (all-MiniLM-L6-v2) when available

TF-IDF vectorizer fallback when heavy ML models cannot load (Streamlit Cloud friendly)

4️⃣ Vector Index

Located in src/vector_index.py

FAISS-like nearest-neighbors retrieval

Automatically falls back if FAISS unavailable

5️⃣ RAG Pipeline

🚩 Most important file: src/rag.py

Features:

Multi-strategy retrieval (vector → SBERT → TF-IDF)

TF-IDF reranking

Strict evidence-only prompt construction

Evidence-verification heuristic

Modular design for easy debugging

6️⃣ Streamlit Interface

Located in src/app.py

Handles user input

Displays retrieved evidence

Streams generated answer

Runs seamlessly on Streamlit Cloud

🧠 Example Questions You Can Ask

Try these on the live app:

“What is the projected GDP growth for Qatar in 2024-25?”

“What risks are highlighted in the IMF report?”

“What reforms are included in Qatar’s Third National Development Strategy (NDS3)?”

“How did Qatar's banking sector perform in 2023?”

📁 Project Structure
multi_modal_rag/
│
├── data/
│   ├── qatar.pdf
│   ├── meta.jsonl
│   ├── pages.jsonl
│   └── embeddings.npy
│
├── src/
│   ├── app.py               # Streamlit interface
│   ├── extract.py           # PDF extraction
│   ├── chunks.py            # Chunk generation
│   ├── embeddings.py        # Embedding generation
│   ├── indexing.py          # Vector index builder
│   ├── vector_index.py      # Lightweight FAISS-like search
│   └── rag.py               # 🔥 Core retrieval+prompt pipeline
│
├── requirements.txt
└── README.md

🧪 How to Run Locally
1. Clone repository
git clone https://github.com/HarshilUndhad/Multi_Model_Rag.git
cd Multi_Model_Rag

2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3. Install requirements
pip install -r requirements.txt

4. Add your Gemini API key

Create a .env file:

GOOGLE_API_KEY=your_key_here

5. Run Streamlit app
streamlit run src/app.py

🌐 Deployment

The project is deployed on Streamlit Cloud, using the same structure as local execution.

Live App Link:

👉 https://multimodelrag-fdwwe2ev2afwkrlbdwwny5.streamlit.app/
🛠️ Technologies Used

Python 3

Streamlit

SentenceTransformers

Scikit-Learn (TF-IDF)

NumPy

FAISS / custom vector index

Google Gemini API

🧩 Key Features / Highlights
✔ Multi-modal PDF understanding

Extracts text, tables, captions, and structured metadata.

✔ Reliable retrieval stack

Try vector index

Try SBERT embeddings

Fall back to TF-IDF

Then rerank for best results

✔ Strict anti-hallucination design

Evidence-only prompt

Limited chunk size

Reranking

Verification step

✔ Cloud-optimized

Runs even on Streamlit Cloud without GPU.

📜 Limitations

Images in PDF are not semantically interpreted (OCR/Tesseract can be added later).

No cross-chunk global reasoning (can be added with long-context LLM).

Table extraction depends on PDF structure quality.
