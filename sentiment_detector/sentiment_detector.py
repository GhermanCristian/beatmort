from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import nltk

from sentiment import Sentiment

nltk.download("wordnet", download_dir="Corpora")
nltk.data.path.append("Corpora")

brown_ic = wordnet_ic.ic("ic-brown.dat")


def get_synset(word: str):
    try:
        synset = wn.synsets(word)[0]
        # TODO - force everything to be noun ? or find adj/verb similar to the sentiments
    except KeyError:
        synset = wn.morphy(word)
    return synset


def similarity_score(s1, s2) -> float:
    return (
        s1.path_similarity(s2)
        # + s1.lch_similarity(s2) * 0.1  # TODO - find better coeff for all these
        + s1.wup_similarity(s2)
        # + s1.res_similarity(s2, brown_ic) * 0.025
        # + s1.jcn_similarity(s2, brown_ic) * 1.0
        + s1.lin_similarity(s2, brown_ic)
    ) / 3.0


def get_closest_sentiment(word: str) -> Sentiment:
    word_synset = get_synset(word)
    scores = {s: similarity_score(word_synset, wn.synset(f"{s.value}.n.01")) for s in Sentiment}
    # TODO - remove neutral, select it if no value is > threshold || too many values close to the max one
    for k, v in scores.items():
        print(k, v)
    return max(scores, key=lambda k: scores[k])


w = "regret"
print(f"Closest sentiment to '{w}' is {get_closest_sentiment(w)}")
