from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

corpus = [
    "Transformers rely entirely on self-attention mechanisms.",
    "Attention is all you need for sequence modeling.",
    "The model achieves high accuracy on the benchmark."
]

def train_word2vec(corpus):
    tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]
    
    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    return model

if __name__ == "__main__":
    w2v_model = train_word2vec(corpus)
    # Access vector for a word
    if "attention" in w2v_model.wv:
        vector = w2v_model.wv["attention"]
        print(f"Vector shape for 'attention': {vector.shape}")
        print("Vector for 'attention':", vector[:5], "...")
