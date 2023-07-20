import spacy
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm

nlp = en_core_web_sm.load()


def has_negation(token):
    """
    Check if a token is a negation word or part of a negation phrase.
    """
    return any([child.dep_ == 'neg' for child in token.children]) or token.dep_ == 'neg'


def has_phrase_negation(chunk):
    """
    Check if a noun phrase contains negation.
    """
    for token in chunk:
        if has_negation(token):
            return True
    return False


def get_synonyms(word):
    """
    Get synonyms for a given word using WordNet.
    """
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def add_words(sentence, phases):
    this_set = set()
    for token in sentence:
        if (
                not has_negation(token) or has_negation(token.head)
        ) and token.lemma_ not in STOP_WORDS:
            for chunk in phases:
                if token in chunk and not has_phrase_negation(chunk):
                    this_set.add(token.lemma_)
    expanded_set = add_synonyms(this_set)
    return expanded_set


def add_synonyms(this_set):
    for word in this_set:
        tokens = word_tokenize(word)
        for token in tokens:
            synonyms = get_synonyms(token.lemma_)
            this_set.update(synonyms)
    return this_set


def chunkify(document):
    phrases = set()
    for chunk in document.noun_chunks:
        phrases.add(chunk)
    return phrases


def jaccard_similarity(sentence1, sentence2):
    nlp = spacy.load("en_core_web_sm")
    doc1 = nlp(sentence1)
    print(type(doc1))
    doc2 = nlp(sentence2)

    # Extract noun phrases from the sentences
    noun_phrases1 = chunkify(doc1)
    noun_phrases2 = chunkify(doc2)

    # Add words and their synonyms to sets, excluding stop words and considering phrase-level negations
    set1 = add_words(doc1, noun_phrases1)
    set2 = add_words(doc2, noun_phrases2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    phrase_similarity = len(noun_phrases1.intersection(noun_phrases2)) / len(noun_phrases1.union(noun_phrases2))
    word_similarity = len(intersection) / len(union)

    jac_sim = 0 * phrase_similarity + 1 * word_similarity
    return jac_sim


# Example usage
sentence1 = "I dont like to work on Sunday."
sentence2 = "I like to work Monday - Saturday."

similarity = jaccard_similarity(sentence1, sentence2)
print(
    "Jaccard similarity (considering synonyms, negations, dual negations, removing stop words, phrase-level similarity, and phrase-level negations):",
    similarity)
