import os
# pyrefly: ignore [missing-import]
import pymupdf4llm

def extract_pdf_to_markdown(pdf_path: str) -> str:
    """
    Extracts text, tables, and formatting from an academic PDF into Markdown
    using PyMuPDF4LLM. Native to Apple Silicon and incredibly fast.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Running PyMuPDF4LLM extraction on {pdf_path}...")
    
    # This directly parses the PDF (including tables) into a Markdown string
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    return md_text

if __name__ == "__main__":
    # Quick test function (will fail if sample.pdf doesn't exist)
    test_pdf = "sample.pdf"
    if os.path.exists(test_pdf):
        print("Testing PDF extraction...")
        md = extract_pdf_to_markdown(test_pdf)
        print("\n--- Extracted Markdown (first 500 chars) ---")
        print(md[:500])
    else:
        print(f"Please place a '{test_pdf}' file in this directory to run standalone tests.")
