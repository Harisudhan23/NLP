import spacy

nlp = spacy.load("en_core_web_sm")

text = "Hi!, This is Sudhan here."

doc = nlp(text)

#tokenization
tokens = [token.text for token in doc ]
print("tokens:",tokens)

#Lemmentization
lemmas = [(token.text, token.lemma_) for token in doc]
print("Lemmas:", lemmas)

#entity
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Entities:", entities)

#stopWords
stop_words = [token.text for token in doc if token.is_stop]
print("stop Words:", stop_words)