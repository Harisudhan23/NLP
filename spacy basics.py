import spacy

nlp = spacy.load("en_core_web_sm")

text = "Hi!, This is Sudhan here."

doc = nlp(text)

#tokenization
tokens = [token.text for token in doc ]
print("tokens:",tokens)
print("count:",tokens.count())