
->   Qatar IMF Document — Multi-Modal RAG QA System

A Retrieval-Augmented Generation (RAG) application that performs multi-modal question answering (text + tables) over the IMF Qatar 2024 Article IV Consultation document using:

SentenceTransformers (all-MiniLM-L6-v2) for embeddings

FAISS (L2 index) for similarity search

Gemini Flash (gemini-flash-latest) for answer generation

Streamlit for an interactive UI

->   Features
->   Retrieval Pipeline

Extracts text, tables, and structured blocks from PDF
Chunks intelligently with metadata
Embeds all chunks using MiniLM
Searches with FAISS (fast, lightweight, high recall)

->   LLM Answering

Uses Gemini Flash for grounded answers
Strict prompt ensures responses use only retrieved evidence
Automatic rejection of irrelevant/off-document questions

->   Table Understanding

Automatic extraction of table-like structures
Heuristic parser converts numeric sequences into structured tables
Displays parsed tables inside Streamlit

->   Streamlit App

Auto-run on pressing Enter
Highlighted retrieved chunks
Concise + detailed answers
Source citations
Table visualization if present

->   Project Structure
multi_modal_rag/
│
├── data/
│   ├── qatar.pdf
│   ├── pages.jsonl
│   ├── chunks.jsonl
│   ├── meta.jsonl
│   ├── embeddings.npy
│   ├── faiss.index
│
├── src/
│   ├── extract.py
│   ├── embeddings.py
│   ├── indexing.py
│   ├── rag.py
│   ├── gen_client.py
│   ├── app.py
│
├── requirements.txt
├── .env.example
└── README.md

->   Installation
1. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

2. Install dependencies
pip install -r requirements.txt

3. Add API key

Create a .env file:

GOOGLE_API_KEY=YOUR_KEY_HERE

->   Data Preparation Pipeline
1. Extract PDF pages → pages.jsonl
python src/extract.py --pdf data/qatar.pdf --out data/pages.jsonl

2. Chunk pages → chunks.jsonl
python src/chunker.py

3. Generate embeddings
python src/embeddings.py

4. Build FAISS index
python src/indexing.py

->   Running the App
streamlit run src/app.py

->  Example Queries

"What is the projected GDP growth for Qatar in 2024–25?"
"What are the main risks to Qatar’s economic outlook?"
"Show fiscal revenue levels over 2020–2024."
"What does the report say about tourism?"
"Who is the President of the United States?" → Should respond: "I don't know."


->   Limitations

Table extraction is heuristic, not 100% perfect
No multilingual support
Limited to one PDF document
No image embeddings (can be added via CLIP)


->  Future Work

CLIP-based image embeddings
Hybrid BM25 + FAISS retrieval
Cross-modal re-ranking
Evaluation dashboard for precision/recall