import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

text = "Hi!, This is Sudhan here."

doc = nlp(text)

#tokenization
tokens = [token.text for token in doc]
print("tokens:",tokens)

#Lemmentization
lemmas = [(token.text, token.lemma_) for token in doc]
print("Lemmas:", lemmas)

with doc.retokenize() as retokenizer:
    retokenizer.merge(doc[3:5], attrs={"LEMMA": "Sudhan"})
print("After:", [token.text for token in doc])

#entity
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Entities:", entities)

#stopWords
stop_words = [token.text for token in doc if token.is_stop]
print("stop Words:", stop_words)

#sentence segmentation
doc = nlp("Whats Up!. Nice to meet you.")
#assert doc.has_annotation("SENT_START")
for sent in doc.sents:
    print(sent.text)