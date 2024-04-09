from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus.reader import Synset
import nltk

from sentiment import Sentiment


class SentimentDetector:
    CORPORA_DIR = "Corpora"
    NEUTRAL_THRESHOLD = 0.3

    def __init__(self) -> None:
        nltk.download("wordnet", download_dir=self.CORPORA_DIR)
        nltk.data.path.append(self.CORPORA_DIR)
        self._brown_ic = wordnet_ic.ic("ic-brown.dat")

    def _get_synset(self, word: str) -> Synset:
        try:
            synset = wn.synsets(word)[0]
            # TODO - force everything to be noun ? or find adj/verb similar to the sentiments
        except KeyError:
            synset = wn.synset(f"{wn.morphy(word, wn.NOUN)}.n.01")
        return synset

    def _similarity_score(self, s1: Synset, s2: Synset) -> float:
        return (
            s1.path_similarity(s2)
            # + s1.lch_similarity(s2) * 0.1  # TODO - find better coeff for all these
            + s1.wup_similarity(s2)
            # + s1.res_similarity(s2, brown_ic) * 0.025
            # + s1.jcn_similarity(s2, brown_ic) * 1.0
            + s1.lin_similarity(s2, self._brown_ic)
        ) / 3.0

    def get_closest_sentiment(self, words: list[str]) -> Sentiment:
        scores_list: list[dict[Sentiment, float]] = []
        for word in words:
            word_synset = self._get_synset(word)
            scores = {
                s: self._similarity_score(word_synset, wn.synset(f"{s.value}.n.01"))
                for s in Sentiment
                if s != Sentiment.NEUTRAL
            }
            scores_list.append(scores)
            for k, v in scores.items():
                print(k, v)

        overall_scores = {
            s: sum(d[s] for d in scores_list) / len(scores_list)
            for s in Sentiment
            if s != Sentiment.NEUTRAL
        }
        for k, v in overall_scores.items():
            print(k, v)
        # TODO - select neutral if values are very close to the max
        if max(overall_scores.values()) < self.NEUTRAL_THRESHOLD:
            return Sentiment.NEUTRAL
        return max(overall_scores, key=lambda k: overall_scores[k])


words = ["happiness", "satisfaction"]
print(f"Closest sentiment to '{words}' is {SentimentDetector().get_closest_sentiment(words)}")
