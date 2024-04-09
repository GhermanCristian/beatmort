from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import nltk

nltk.download("wordnet", download_dir="Corpora")
nltk.data.path.append("Corpora")

brown_ic = wordnet_ic.ic("ic-brown.dat")


def get_synset(word: str):
    try:
        synset = wn.synsets(word)[0]
    except KeyError:
        synset = wn.morphy(word)
    return synset


dog = get_synset("dog")
cat = get_synset("cat")

print(dog.path_similarity(cat))
print(dog.lch_similarity(cat))
print(dog.wup_similarity(cat))
print(dog.res_similarity(cat, brown_ic))
print(dog.jcn_similarity(cat, brown_ic))
print(dog.lin_similarity(cat, brown_ic))
