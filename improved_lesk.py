from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
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


def get_overlaping(concept, sentence):
    overlap = overlapcontext(concept, sentence)
    extended_gloss = concept.hyponyms() + concept.hypernyms()
    for hyp in extended_gloss:
        overlap += overlapcontext(hyp, sentence)
    return overlap

def lesk_distance(concept_i, concept_j):
    distance = 0
    distance += get_overlaping(concept_i, concept_j.definition())
    distance += get_overlaping(concept_j, concept_i.definition())
    return distance
    


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
