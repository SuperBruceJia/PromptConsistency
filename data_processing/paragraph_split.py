# coding=utf-8

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")


def paragraph_splitter(paragraph):
    # Process the paragraph with spaCy
    doc = nlp(paragraph)

    # Extract sentences
    sentences = [sent.text for sent in doc.sents]

    return sentences
