from nltk.corpus import wordnet, wordnet_ic

x = wordnet.synset('sunday.n.01')
y = wordnet.synset('friday.n.01')


ic = wordnet_ic.ic('ic-brown.dat')
print(x.res_similarity(y, ic=ic))