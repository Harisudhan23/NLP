# Importing necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import gensim.downloader as api
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu

# Step 1: Sample dataset
texts = [
    "I love programming in Python.",
    "The movie was a terrible waste of time.",
    "Artificial Intelligence is shaping the future.",
    "This book was boring and not worth reading.",
    "AI is making the world smarter and more efficient.",
    "I didn't enjoy the movie at all."
]
labels = [1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative

# Step 2: Bag of Words (BoW) Implementation
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X_bow, labels, test_size=0.3, random_state=42)

bow_model = SVC(kernel="linear")
bow_model.fit(X_train, y_train)
pred_bow = bow_model.predict(X_test)
print("BoW Accuracy:", accuracy_score(y_test, pred_bow))
print("Classification Report (BoW):\n", classification_report(y_test, pred_bow))

# Step 3: GloVe Embeddings Implementation
glove_vectors = api.load("glove-wiki-gigaword-50")
def text_to_glove_vector(text):
    words = text.split()
    vectors = [glove_vectors[word] for word in words if word in glove_vectors]
    return np.mean(vectors, axis=0) if vectors else np.zeros(50)
X_glove = np.array([text_to_glove_vector(text) for text in texts])
X_train_glove, X_test_glove, y_train_glove, y_test_glove = train_test_split(X_glove, labels, test_size=0.3, random_state=42)

glove_model = SVC(kernel="linear")
glove_model.fit(X_train_glove, y_train_glove)
pred_glove = glove_model.predict(X_test_glove)
print("GloVe Accuracy:", accuracy_score(y_test_glove, pred_glove))
print("Classification Report (GloVe):\n", classification_report(y_test_glove, pred_glove))

# Step 4: BERT Transformer Implementation
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {key: val.squeeze() for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label)
        return encoding

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = TextDataset(texts, labels, tokenizer)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    logging_dir="./logs",
)
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=dataset,
     eval_dataset=dataset,
)
trainer.train()

# Evaluate BERT
predictions = trainer.predict(dataset)
y_pred_bert = predictions.predictions.argmax(-1)
print("BERT Accuracy:", accuracy_score(labels, y_pred_bert))
print("Classification Report (BERT):\n", classification_report(labels, y_pred_bert))

# Step 5: Summarization with ROUGE and BLEU Scores
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = """
Artificial Intelligence (AI) is transforming industries by automating tasks, improving decision-making, 
and enabling innovative solutions to complex problems. From healthcare to finance, AI's impact is profound.
"""
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
reference_summary = "AI is transforming industries by automating tasks and enabling innovation."

# ROUGE Score
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, summary)
print("ROUGE Scores:", scores)

# BLEU Score
bleu_score = sentence_bleu(summary, [reference_summary])
print("BLEU Score:", bleu_score.score)
