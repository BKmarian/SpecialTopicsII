from nltk.corpus import wordnet
from nltk import word_tokenize
import sys
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def overlapcontext(synset, sentence):
    gloss = set(word_tokenize(synset.definition()))
    for i in synset.examples():
        gloss.union(i)
    gloss = gloss.difference(STOPWORDS)
    if isinstance(sentence, str):
        sentence = set(sentence.split(" "))
    elif isinstance(sentence, list):
        sentence = set(sentence)
    elif isinstance(sentence, set):
        pass
    else:
        return
    sentence = sentence.difference(STOPWORDS)
    return len(gloss.intersection(sentence))

def get_wordnet_form(word):
    return wordnet.morphy(word) if wordnet.morphy(word) is not None else word
    
def lesk(word, sentence):
    bestsense = None
    maxoverlap = 0
    word = get_wordnet_form(word)
    for sense in wordnet.synsets(word):
        overlap = overlapcontext(sense, sentence)
        extended_gloss = sense.hyponyms() + sense.hypernyms()
        for h in extended_gloss:
            overlap += overlapcontext(h, sentence)
        if overlap > maxoverlap:
            maxoverlap = overlap
            bestsense = sense
    return (maxoverlap,bestsense)


# sentence = "Enter the Sentence (or) Context :"
# word = "Enter the word :"

# a = lesk(word, sentence)
# print("\n\nSynset:", a)
# if a is not None:
#     print("Meaning:", a.definition())
#     num = 0
#     print("\nExamples:")
#     for i in a.examples():
#         num = num + 1
#         print(str(num) + '.' + ')', i)
