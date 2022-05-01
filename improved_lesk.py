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
wordnet_poss = [wordnet.NOUN, wordnet.ADJ, wordnet.ADV, wordnet.VERB]

import pdb

def pos_map(pos_):
    first_letter = pos_[0].lower()
    if first_letter == 'j':
        return 'a'
    return first_letter

def tokenize_and_clean(sentence):
    sentence_ = [word for word in word_tokenize(sentence) if word not in FORBIDEN]
    return sentence_

def tokenize_and_pos(sentence):
    token_pos_pair = nltk.pos_tag(word_tokenize(sentence))
    token_pos_map = {token: pos_map(pos) for token, pos in token_pos_pair}
    return token_pos_map

def overlapcontext(synset, sentence, max_ngrams):
    # TODO: also add 2-3 gram overlap only on firefly algorithm
    # firefly    - suma polinomiala (p cuvinte care apar impreuna + (overlap-urile) ** p, p >=2 )
    # ant colony - suma liniara
    
    gloss_ngrams = []
    gloss = [synset.definition()] + synset.examples()
    for example in gloss:
        example_set = tokenize_and_clean(example)
        gloss_ngrams += nltk.everygrams(example_set, 1, max_ngrams)

    sentence = tokenize_and_clean(sentence)
    sentece_ngrams = nltk.everygrams(sentence, 1, max_ngrams)

    ngrams_intersection = set(gloss_ngrams).intersection(sentece_ngrams)
    values, counts = np.unique(
        list(map(lambda ngram: len(ngram), ngrams_intersection))
    , return_counts=True)
    
    score = 0
    for value, count in zip(values, counts):
        score += count ** value
    return score

# def get_wordnet_form(word):
#     return wordnet.morphy(word) if wordnet.morphy(word) is not None else word
    
# def lesk(word, sentence):
#     bestsense = None
#     maxoverlap = 0
#     word = get_wordnet_form(word)
#     for sense in wordnet.synsets(word):
#         overlap = overlapcontext(sense, sentence)
#         extended_gloss = sense.hyponyms() + sense.hypernyms()
#         for hyp in extended_gloss:
#             overlap += overlapcontext(hyp, sentence)
#         if overlap > maxoverlap:
#             maxoverlap = overlap
#             bestsense = sense
#     return (maxoverlap,bestsense)

@lru_cache(maxsize=None)
def get_overlaping(concept, sentence, max_ngrams=1):
    overlap = overlapcontext(concept, sentence, max_ngrams)
    extended_gloss = concept.hyponyms() + \
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

    for hyp in extended_gloss:
        overlap += overlapcontext(hyp, sentence, max_ngrams)
    return overlap

def lesk_unidirect_distance(concept_i, concept_j):
    sentence = concept_i.definition()
    tokens_ = tokenize_and_clean(sentence)
    token_pos_map = tokenize_and_pos(sentence)
    
    distance = 0
    for token_ in tokens_:
        pos = token_pos_map[token_]
        if pos in wordnet_poss:
            synsets = wordnet.synsets(token_, pos=token_pos_map[token_])
            for synset in synsets:
                distance += get_overlaping(synset, concept_j.definition(), max_ngrams=5)
    return distance

@lru_cache(maxsize=None)
def lesk_distance(concept_i, concept_j):
    distance = 0

    distance += lesk_unidirect_distance(concept_i, concept_j)
    distance += lesk_unidirect_distance(concept_j, concept_i)
    return distance

@lru_cache(maxsize=None)    
def lesk_distance2(concept_i, concept_j):
    distance = 0
    #print(concept_i , concept_j)
    if wordnet.synsets(concept_i) == None or wordnet.synsets(concept_j) == None:
        return distance

    for syn1 in wordnet.synsets(concept_i):
        for syn2 in wordnet.synsets(concept_j):
            distance += get_overlaping(syn1, syn2.definition())
            distance += get_overlaping(syn2, syn1.definition())
    return distance

def lesk_distance_full(odour_node,odour_ant):
    sum = 0
    for word in odour_node:
        for word2 in odour_ant:
            sum+= lesk_distance2(word,word2)
    return sum


# concept_1 = wordnet.synset('friday.n.01')
# concept_2 = wordnet.synset('state.n.01')
# print(lesk_distance(concept_1, concept_2))


#sentence = "I want to go to watch football game in the street"
#word = "game"

# a = lesk(word, sentence)
# print("\n\nSynset:", a)
# if a is not None:
#     print("Meaning:", a.definition())
#     num = 0
#     print("\nExamples:")
#     for i in a.examples():
#         num = num + 1
#         print(str(num) + '.' + ')', i)

#synset1.path_similarity(synset2)