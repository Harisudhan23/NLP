import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

documents = [
    "Spacy is a great library for NLP tasks.",
    "Machine learning models improve decision-making.",
    "I enjoy solving problems using machine learning and AI."
]

preprocessed_docs = [preprocess_text(doc) for doc in documents]
print("Preprocessed Texts:", preprocessed_docs)

#Bag of Words(BoW) implementation
vectorizer = CountVectorizer()

bow_matrix = vectorizer.fit_transform(preprocessed_docs)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Matrix:\n", bow_matrix.toarray())

#TF-IDF Implementation
tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)

print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())