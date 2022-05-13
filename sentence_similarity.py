
import spacy
#from sentence_transformers import SentenceTransformer, util
from functools import lru_cache

#model = SentenceTransformer('distilbert-base-nli-mean-tokens')
nlp = spacy.load("en_core_web_md")

@lru_cache(maxsize=None)
def setence_sim_spacy(sent1 , sent2):
    doc1 = nlp(" ".join(sent1))
    doc2 = nlp(" ".join(sent2))
    return doc1.similarity(doc2)

@lru_cache(maxsize=None)
def sentence_sim_transformers(sent1, sent2):
    sentence_embeddings = model.encode([sent1,sent2])
    return util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1])