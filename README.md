# 📚 Academic Text Simplifier

An AI-powered tool that automatically extracts, chunks, and simplifies dense academic papers (PDFs and TXTs) using T5 Transformers and SciBERT. It features a modern Streamlit frontend and a robust FastAPI backend.

## 🌟 Features
- **PDF Extraction:** Uses `PyMuPDF4LLM` to perfectly extract text and serialize data tables into Markdown natively on Apple Silicon.
- **Table Linearization:** Automatically detects Markdown tables and translates them into plain English sentences, allowing the NLP model to "read" and summarize tabular data.
- **Text Simplification:** Uses the `t5-base` transformer model to summarize complex paragraphs while preserving the core academic thesis.
- **Smart Formatting:** Employs `nltk` for sentence casing and intelligently strings summarized chunks into cohesive, flowing paragraphs.
- **Multi-Format Support:** Upload `.pdf` or `.txt` files, or paste confusing paragraphs directly into the UI.
- **Interactive UI:** Built with Streamlit for a drag-and-drop, tabbed user experience.

## 📂 Repository Structure
```
├── app.py                         # FastAPI backend server and endpoints
├── frontend.py                    # Streamlit interactive UI
├── Transformers.py        # T5 chunking, summarization, and table linearization
├── PDF_Extraction.py      # PDF to Markdown extraction using PyMuPDF4LLM
├── test_api.py                    # Script to test the API programmatically
├── requirements.txt               # Python dependencies
├── sample.pdf                     # Sample academic paper for testing
├── input.txt                      # Sample text for testing
├── output.txt                     # Sample generated output
└── README.md                      # Project documentation
```

## 🚀 Getting Started

### 1. Install Dependencies
Make sure you are using Python 3.11+.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Backend (FastAPI)
In your first terminal, start the API engine:
```bash
python -m uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### 3. Run the Frontend (Streamlit)
In a second terminal, start the UI:
```bash
streamlit run frontend.py
```
The web app will automatically open at `http://localhost:8501`.

## 🧠 How it Works (Under the Hood)
1. **Extraction:** `module5_pdf_extraction` converts the PDF into Markdown.
2. **Linearization:** `module4_transformers` intercepts markdown tables (e.g. `| column |`) and rewrites them as English sentences (e.g. "Table row: column is X").
3. **Chunking & Summarization:** The text is safely chunked into 350-word blocks and fed to the `t5-base` transformer.
4. **Post-Processing:** Extraneous Markdown characters (`*`, `#`, `_`) are stripped, and `nltk` ensures perfect sentence casing and paragraph flow.
