import spacy
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load IMDb dataset
dataset = load_dataset("imdb")

# Split dataset into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    dataset['train']['text'], dataset['train']['label'], test_size=0.2, random_state=42
)

nlp = spacy.load("en_core_web_sm")

def preprocess_texts(texts):
    return [
        " ".join([token.lemma_.lower() for token in nlp(text) if not token.is_stop and token.is_alpha])
        for text in texts
    ]

train_texts = preprocess_texts(train_texts)
test_texts = preprocess_texts(test_texts)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# BoW vectorization
vectorizer = CountVectorizer(max_features=1000)
train_bow = vectorizer.fit_transform(train_texts)
test_bow = vectorizer.transform(test_texts)

# Logistic Regression for classification
clf_bow = LogisticRegression(max_iter=1000)
clf_bow.fit(train_bow, train_labels)
bow_predictions = clf_bow.predict(test_bow)

print("BoW Accuracy:", accuracy_score(test_labels, bow_predictions))

import gensim.downloader as api
import numpy as np

# Load pre-trained GloVe embeddings
glove_vectors = api.load("glove-wiki-gigaword-50")

def embed_with_glove(texts):
    embeddings = []
    for text in texts:
        words = text.split()
        word_embeddings = [glove_vectors[word] for word in words if word in glove_vectors]
        if word_embeddings:
            embeddings.append(np.mean(word_embeddings, axis=0))
        else:
            embeddings.append(np.zeros(50))  # Handle texts with no valid words
    return np.array(embeddings)

train_glove = embed_with_glove(train_texts)
test_glove = embed_with_glove(test_texts)

# Logistic Regression for classification
clf_glove = LogisticRegression(max_iter=1000)
clf_glove.fit(train_glove, train_labels)
glove_predictions = clf_glove.predict(test_glove)

print("GloVe Accuracy:", accuracy_score(test_labels, glove_predictions))

nlp_trf = spacy.load("en_core_web_trf")

def embed_with_transformer(texts):
    return [nlp_trf(text).vector for text in texts]

train_transformer = embed_with_transformer(train_texts)
test_transformer = embed_with_transformer(test_texts)

# Logistic Regression for classification
clf_transformer = LogisticRegression(max_iter=1000)
clf_transformer.fit(train_transformer, train_labels)
transformer_predictions = clf_transformer.predict(test_transformer)

print("Transformer-Based Accuracy:", accuracy_score(test_labels, transformer_predictions))

from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Summarize a sample text
sample_text = (
    "SpaceX designs, develops, and manufactures space launch vehicles, spacecraft, and satellite systems. "
    "Led by Elon Musk, SpaceX has launched a number of historic missions, including the first privately-funded "
    "craft to reach orbit and the first manned mission by a private company to the International Space Station."
)
summary = summarizer(sample_text, max_length=50, min_length=25, do_sample=False)
print("Summary:", summary[0]["summary_text"])

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# Reference and generated summaries
reference_summary = "SpaceX develops space launch vehicles and spacecraft. It is led by Elon Musk and achieved historic milestones."
generated_summary = summary[0]["summary_text"]

# ROUGE Evaluation
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = scorer.score(reference_summary, generated_summary)
print("ROUGE Scores:", rouge_scores)

# BLEU Evaluation
bleu_score = sentence_bleu([reference_summary.split()], generated_summary.split())
print("BLEU Score:", bleu_score)

import matplotlib.pyplot as plt

# Accuracies
accuracies = [
    accuracy_score(test_labels, bow_predictions),
    accuracy_score(test_labels, glove_predictions),
    accuracy_score(test_labels, transformer_predictions)
]

approaches = ["BoW", "GloVe", "Transformer"]

# Bar Plot
plt.bar(approaches, accuracies, color=["blue", "green", "orange"])
plt.title("Accuracy Comparison")
plt.xlabel("Approach")
plt.ylabel("Accuracy")
plt.show()


