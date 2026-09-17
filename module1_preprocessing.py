import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# Load spacy en_core_web_sm model (in practice, use a sci-specific model if available)
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Basic tokenization and stop-word removal using NLTK
    stop_words = set(stopwords.words('english'))
    scientific_stops = {'et', 'al', 'fig', 'table', 'equation'}
    stop_words.update(scientific_stops)
    
    tokens = word_tokenize(text.lower())
    filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    
    # Advanced Linguistic Analysis using spaCy
    doc = nlp(text)
    
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return {
        "filtered_tokens": filtered_tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "entities": entities
    }

if __name__ == "__main__":
    # Example Usage
    sample_text = "The proposed Transformer model achieves state-of-the-art results on the dataset, as shown in Table 1."
    print("Preprocessing Sample:")
    print(preprocess_text(sample_text))
