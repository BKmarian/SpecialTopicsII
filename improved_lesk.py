from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.corpus import stopwords
from functools import lru_cache

import numpy as np
import string
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('wordnet_ic')
STOPWORDS = set(stopwords.words('english'))
FORBIDEN = STOPWORDS.union(set(string.punctuation))

def pos_map(pos_):
    first_letter = pos_[0].lower()
    if first_letter == 'j':
        return 'a'
    return first_letter

def tokenize_and_clean(sentence):
    sentence_ = [word for word in word_tokenize(sentence) if word not in FORBIDEN]
    return sentence_

@lru_cache(maxsize=None)
def get_extended_concepts(concept):
    extended_concepts = [concept] + \
        concept.hyponyms() + \
        concept.hypernyms() + \
        concept.substance_meronyms() + \
        concept.part_meronyms() + \
        concept.member_meronyms() + \
        concept.part_holonyms() + \
        concept.member_holonyms() + \
        concept.topic_domains() + \
        concept.region_domains() + \
        concept.usage_domains() + \
        concept.entailments() + \
        concept.causes() + \
        concept.also_sees() + \
        concept.verb_groups() + \
        concept.attributes() + \
        concept.similar_tos()
    return extended_concepts

def extract_gloss_ngrams(subconcept, max_ngrams):
    gloss_ngrams = []
    gloss = [subconcept.definition()] + subconcept.examples()
    for example in gloss:
        example_set = tokenize_and_clean(example)
        gloss_ngrams += nltk.everygrams(example_set, 1, max_ngrams)
    return set(gloss_ngrams)

def get_overlap_score(subconcept_i, subconcept_j, max_ngrams=1):
    gloss_i = extract_gloss_ngrams(subconcept_i, max_ngrams)
    gloss_j = extract_gloss_ngrams(subconcept_j, max_ngrams)

    ngrams_intersection = gloss_i & gloss_j
    values, counts = np.unique(
        list(map(lambda ngram: len(ngram), ngrams_intersection))
    , return_counts=True)
    
    score = 0
    for value, count in zip(values, counts):
        score += count ** value
    return score

@lru_cache(maxsize=None)
def lesk_distance(concept_i, concept_j, max_ngrams=5):
    distance = 0

    concept_i_ext = get_extended_concepts(concept_i)
    concept_j_ext = get_extended_concepts(concept_j)

    for subconcept_i in concept_i_ext:
        for subconcept_j in concept_j_ext:
            distance += get_overlap_score(subconcept_i, subconcept_j, max_ngrams)
    return distance

@lru_cache(maxsize=None)    
def lesk_distance2(concept_i, concept_j):
    distance = 0
    #print(concept_i , concept_j)
    if wordnet.synsets(concept_i) == None or wordnet.synsets(concept_j) == None:
        return distance

    for syn1 in wordnet.synsets(concept_i):
        for syn2 in wordnet.synsets(concept_j):
            distance += lesk_distance(syn1, syn2, max_ngrams=1)
    return distance

def lesk_distance_full(odour_node, odour_ant):
    sum = 0
    for word in odour_node:
        for word2 in odour_ant:
            sum+= lesk_distance2(word,word2)
    return sum

if __name__ == "__main__":
    concept_1 = wordnet.synset('dog.n.01')
    concept_2 = wordnet.synset('cat.n.01')
    print(lesk_distance(concept_1, concept_2))