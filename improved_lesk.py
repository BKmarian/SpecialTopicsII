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
lemmatizer = nltk.stem.WordNetLemmatizer()

def pos_map(pos_):
    first_letter = pos_[0].lower()
    if first_letter == 'j':
        return 'a'
    return first_letter

def tokenize_and_clean(sentence):
    sentence_ = [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word not in FORBIDEN]
    return sentence_

def get_subconcept_ngrams(sub_concepts, max_ngrams):
    all_ngrams = []
    for sub_concept in sub_concepts:
        sent = sub_concept.name() + ' ' + sub_concept.definition()
        sent_clean = tokenize_and_clean(sent)
        all_ngrams += nltk.everygrams(sent_clean, 1, max_ngrams)
    return all_ngrams

@lru_cache(maxsize=None)
def get_ngrams_by_scope(concept, scope, max_ngrams):
    if scope == "gloss":
        sent = concept.name() + ' ' + concept.definition()
        sent_clean = tokenize_and_clean(sent)
        ngrams = nltk.everygrams(sent_clean, 1, max_ngrams)
        return ngrams

    elif scope == "example":
        all_ngrams = []
        for example in concept.examples():
            sent_clean = tokenize_and_clean(example)
            all_ngrams += nltk.everygrams(sent_clean, 1, max_ngrams)
        return all_ngrams

    elif scope == "hypo":
        all_ngrams = []
        for sub_concept in concept.hyponyms():
            sent = sub_concept.name() + ' ' + sub_concept.definition()
            sent_clean = tokenize_and_clean(sent)
            all_ngrams += nltk.everygrams(sent_clean, 1, max_ngrams)
        return all_ngrams

    elif scope == "mero":
        meronyms = \
            concept.substance_meronyms() + \
            concept.part_meronyms() + \
            concept.member_meronyms()
        return get_subconcept_ngrams(meronyms, max_ngrams)

    elif scope == "also":
        return get_subconcept_ngrams(concept.also_sees(), max_ngrams)

    elif scope == "attr":
        return get_subconcept_ngrams(concept.attributes(), max_ngrams)

    elif scope == "hype":
        return get_subconcept_ngrams(concept.hypernyms(), max_ngrams)


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

def get_overlap_score(concept_i, concept_j, relation, max_ngrams=1):
    scope_i, scope_j = relation
    ngrams_i = get_ngrams_by_scope(concept_i, scope_i, max_ngrams)
    ngrams_j = get_ngrams_by_scope(concept_j, scope_j, max_ngrams)

    ngrams_intersection = set(ngrams_i) & set(ngrams_j)
    values, counts = np.unique(
        list(map(lambda ngram: len(ngram), ngrams_intersection))
    , return_counts=True)
    
    score = 0
    for value, count in zip(values, counts):
        score += count ** value
    return score

# @lru_cache(maxsize=None)
# def lesk_distance_ant(concept_i, concept_j):
#     distance = 0

#     concept_i_ext = get_extended_concepts(concept_i)
#     concept_j_ext = get_extended_concepts(concept_j)

#     return sum([concept_j_ext.count(element) for element in concept_i_ext])

def get_pos_best_relations(pos_i, pos_j):
    if pos_i == 'n' and pos_j == 'n':
        return [
            ("hypo", "mero"),
            ("mero", "hypo"),
            ("gloss", "mero"),
            ("mero", "gloss"),
            ("gloss", "gloss"),
            ("example", "mero"),
            ("mero", "example")
        ]
    elif pos_i == 'a' and pos_j == 'a':
        return [
            ("also", "gloss"),
            ("gloss", "also"),
            ("gloss", "gloss"),
            ("example", "gloss"),
            ("gloss", "example"),
            ("hype", "gloss"),
            ("gloss", "hype")
        ]
    elif pos_i == 'v' and pos_j == 'v':
        return [
            ("example", "example"),
            ("example", "hype"),
            ("hype", "example"),
            ("hypo", "hypo"),
            ("gloss", "hypo"),
            ("hypo", "gloss"),
            ("example", "gloss"),
            ("gloss", "example"),
        ]
    return []

@lru_cache(maxsize=None)
def lesk_distance_ant(concept_i, concept_j):
    concept_i_ext = get_extended_concepts(concept_i)
    concept_j_ext = get_extended_concepts(concept_j)

    return len(np.intersect1d(concept_i_ext , concept_j_ext))
    #return sum([concept_j_ext.count(element) for element in concept_i_ext])
def lesk_distance(concept_i, concept_j, max_ngrams):
    distance = 0

    relations = get_pos_best_relations(concept_i.pos(), concept_j.pos())
    for relation in relations:
        distance += get_overlap_score(concept_i, concept_j, relation, max_ngrams)
    return distance

@lru_cache(maxsize=None)
def lesk_distance_legacy(concept_i, concept_j, max_ngrams):
    distance = 0

    concept_i_ext = get_extended_concepts(concept_i)
    concept_j_ext = get_extended_concepts(concept_j)

    for subconcept_i in concept_i_ext:
        for subconcept_j in concept_j_ext:
            distance += get_overlap_score(subconcept_i, subconcept_j, max_ngrams)
    return distance

@lru_cache(maxsize=None)
def lesk_distance_X(word1,word2):
    sum = 0
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    for s1 in synsets1:
        for s2 in synsets2:
            sum+=lesk_distance_ant(s1,s2)
    return sum
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
            sum+= lesk_distance_X(word,word2)
    return sum

if __name__ == "__main__":
    concept_1 = wordnet.synset('dog.n.01')
    concept_2 = wordnet.synset('cat.n.01')
    print(lesk_distance_X("dog", "cat"))