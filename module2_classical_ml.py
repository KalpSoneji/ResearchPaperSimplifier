from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import numpy as np

corpus = [
    "Transformers rely entirely on self-attention mechanisms.",
    "Attention is all you need for sequence modeling.",
    "The model achieves high accuracy on the benchmark."
]

def vectorization_comparison(corpus):
    # Bag of Words
    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(corpus)
    
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    return bow_matrix, tfidf_matrix

def train_ngram_lm(corpus, n=3):
    tokenized_text = [word_tokenize(sentence.lower()) for sentence in corpus]
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    
    model = MLE(n)
    model.fit(train_data, padded_sents)
    return model

if __name__ == "__main__":
    bow, tfidf = vectorization_comparison(corpus)
    print(f"BoW shape: {bow.shape}")
    print(f"TF-IDF shape: {tfidf.shape}")

    # Calculate Perplexity for a test sentence
    test_sentence = word_tokenize("Transformers rely on attention mechanisms.".lower())
    lm_model = train_ngram_lm(corpus, n=3)
    
    # Creating trigrams for testing
    test_trigrams = list(ngrams(test_sentence, 3))
    
    # Perplexity (handle unseen ngrams appropriately in production with smoothing)
    # Note: MLE will give inf perplexity for unseen ngrams. Using Laplace smoothing in practice.
    perplexity = lm_model.perplexity(test_trigrams)
    print(f"Trigram Perplexity: {perplexity}")
