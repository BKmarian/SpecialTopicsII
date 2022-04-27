from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
from functools import lru_cache
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('wordnet_ic')
STOPWORDS = set(stopwords.words('english'))

def overlapcontext(synset, sentence):
    # TODO: also add 2-3 gram overlap only on firefly algorithm
    # firefly    - suma polinomiala (p cuvinte care apar impreuna + (overlap-urile) ** p, p >=2 )
    # ant colony - suma liniara
    gloss = set(word_tokenize(synset.definition()))
    for i in synset.examples():
        gloss.union(i)
    gloss = gloss.difference(STOPWORDS)
    
    sentence = set(sentence.split(" "))
    sentence = sentence.difference(STOPWORDS)
    return len(gloss.intersection(sentence))

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
def get_overlaping(concept, sentence):
    overlap = overlapcontext(concept, sentence)
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
        overlap += overlapcontext(hyp, sentence)
    return overlap

def lesk_distance(concept_i, concept_j):
    distance = 0
    distance += get_overlaping(concept_i, concept_j.definition())
    distance += get_overlaping(concept_j, concept_i.definition())
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