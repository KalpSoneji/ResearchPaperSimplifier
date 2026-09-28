import torch
import nltk
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# 1. Insight Extraction with SciBERT
scibert_name = "allenai/scibert_scivocab_uncased"
try:
    tokenizer_sci = AutoTokenizer.from_pretrained(scibert_name)
    model_sci = AutoModel.from_pretrained(scibert_name)
except Exception as e:
    print(f"Failed to load SciBERT (maybe no internet connection): {e}")
    tokenizer_sci = None
    model_sci = None

def get_scibert_embeddings(text):
    if tokenizer_sci is None or model_sci is None:
        return None
    inputs = tokenizer_sci(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_sci(**inputs)
    # Return the pooled output (CLS token representation)
    return outputs.pooler_output

# 2. Text Simplification with T5
t5_name = "t5-base"
try:
    tokenizer_t5 = AutoTokenizer.from_pretrained(t5_name)
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_name)
except Exception as e:
    print(f"Failed to load T5: {e}")
    tokenizer_t5 = None
    model_t5 = None

def chunk_text(text, max_words=350):
    """Splits text into chunks of roughly max_words, preserving sentence boundaries."""
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt_tab')
        sentences = nltk.sent_tokenize(text)
        
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
            
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks if chunks else [text]

def to_sentence_case(text):
    """Converts a block of text into proper Sentence case."""
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt_tab')
        sentences = nltk.sent_tokenize(text)
    
    capitalized = []
    for s in sentences:
        s = s.strip()
        if not s: continue
        # Capitalize first letter but keep rest intact
        capitalized.append(s[0].upper() + s[1:])
    return " ".join(capitalized)

def preprocess_tables_in_text(text):
    """
    Detects markdown tables in text and linearizes them into English sentences
    so small seq2seq models like T5 can read the tabular data naturally.
    """
    lines = text.split('\n')
    new_lines = []
    in_table = False
    headers = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            # It's part of a table
            cells = [cell.strip() for cell in stripped.split('|')[1:-1]]
            
            if not in_table:
                # First line of table is usually headers
                headers = cells
                in_table = True
                new_lines.append(f"The following table contains columns: {', '.join(headers)}.")
            elif all(c == '-' or c == ':' or c == '' for c in cells[0]):
                # This is the |---|---| separator row, skip it
                continue
            else:
                # Data row
                row_desc = []
                for i, cell in enumerate(cells):
                    if i < len(headers) and cell:
                        row_desc.append(f"{headers[i]} is {cell}")
                if row_desc:
                    new_lines.append("Table row: " + ", ".join(row_desc) + ".")
        else:
            in_table = False
            new_lines.append(line)
            
    return "\n".join(new_lines)

def simplify_text(text):
    if tokenizer_t5 is None or model_t5 is None:
        return "Model not loaded."
    # Pre-process raw text to linearize tables before chunking
    text = preprocess_tables_in_text(text)
    
    # Strip markdown formatting that confuses the T5 tokenizer
    # (e.g. **B**idirectional becomes Bidirectional)
    import re
    # Strip HTML tags like <br> that PyMuPDF4LLM leaves in tables
    text = re.sub(r'<[^>]+>', ' ', text)
    # Strip malformed tags or leftovers
    text = re.sub(r'br>', ' ', text)
    # Strip markdown formatting that confuses the tokenizer
    text = re.sub(r'[*#_]', '', text)
    
    chunks = chunk_text(text)
    summaries = []
    
    for chunk in chunks:
        # T5 often uses task-specific prefixes
        input_text = "summarize: " + chunk
        inputs = tokenizer_t5(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            summary_ids = model_t5.generate(
                inputs["input_ids"], 
                num_beams=4, 
                min_length=10, 
                max_length=150, 
                early_stopping=True
            )
        
        summary = tokenizer_t5.decode(summary_ids[0], skip_special_tokens=True)
        summary = to_sentence_case(summary)
        summaries.append(summary)
        
    # Group chunk summaries into cohesive paragraphs to improve flow
    # E.g., combine every 3 chunk summaries into 1 paragraph
    paragraphs = []
    for i in range(0, len(summaries), 3):
        paragraph = " ".join(summaries[i:i+3])
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)

if __name__ == "__main__":
    complex_text = "We propose a novel neural network architecture based on self-attention mechanisms that entirely dispenses with recurrences and convolutions, yielding superior parallelization capabilities and reduced training times."
    
    if model_sci:
        emb = get_scibert_embeddings(complex_text)
        print("SciBERT Embedding shape:", emb.shape)
        
    if model_t5:
        print("Original:", complex_text)
        print("Simplified:", simplify_text(complex_text))
