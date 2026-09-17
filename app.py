from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import shutil
import os
import tempfile
from module1_preprocessing import preprocess_text
from module4_transformers import simplify_text, get_scibert_embeddings
from module5_pdf_extraction import extract_pdf_to_markdown

app = FastAPI(title="Academic Text Simplification API", version="1.0")

class TextRequest(BaseModel):
    text: str

class PreprocessResponse(BaseModel):
    filtered_tokens: list
    lemmas: list
    pos_tags: list
    entities: list

class SimplifyResponse(BaseModel):
    original_text: str
    simplified_text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Academic Text Simplification API"}

@app.post("/preprocess", response_model=PreprocessResponse)
def api_preprocess(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = preprocess_text(request.text)
        return result
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/simplify", response_model=SimplifyResponse)
def api_simplify(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    simplified = simplify_text(request.text)
    
    return {
        "original_text": request.text,
        "simplified_text": simplified
    }

class PDFSimplifyResponse(BaseModel):
    extracted_markdown: str
    simplified_text: str

@app.post("/upload_file", response_model=PDFSimplifyResponse)
def api_upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".txt"):
        try:
            text = file.file.read().decode("utf-8")
            simplified = simplify_text(text)
            return {
                "extracted_markdown": text,
                "simplified_text": simplified
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    elif not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
        
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        try:
            # 1. Extract markdown from PDF using PyMuPDF4LLM
            markdown_text = extract_pdf_to_markdown(tmp_path)
            
            # 2. Simplify the extracted markdown text
            simplified = simplify_text(markdown_text)
            
            return {
                "extracted_markdown": markdown_text,
                "simplified_text": simplified
            }
        finally:
            # Clean up the temporary PDF file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn app:app --reload
