import re
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic, stopwords
from nltk.corpus.reader import Synset
import nltk
from typing import Optional

from sentiment import Sentiment


class SentimentDetector:
    CORPORA_DIR = "Corpora"
    NEUTRAL_THRESHOLD = 0.3

    def __init__(self) -> None:
        nltk.download("wordnet", download_dir=self.CORPORA_DIR, quiet=True)
        nltk.data.path.append(self.CORPORA_DIR)
        self._brown_ic = wordnet_ic.ic("ic-brown.dat")
        self._stopwords = stopwords.words("english")
        self._sentiment_equivalents = {
            Sentiment.ANGER: {
                wn.ADJ: "angry",
                wn.ADJ_SAT: "angered",
                wn.ADV: "angrily",
                wn.NOUN: "anger",
                wn.VERB: "anger",
            },
            Sentiment.ANTICIPATION: {
                wn.ADJ: "expected",
                wn.ADJ_SAT: "anticipant",
                wn.ADV: "expectantly",
                wn.NOUN: "expectation",
                wn.VERB: "expect",
            },
            Sentiment.DISGUST: {
                wn.ADJ: "unpleasant",
                wn.ADJ_SAT: "disgusting",
                wn.ADV: "disgustedly",
                wn.NOUN: "disgust",
                wn.VERB: "disgust",
            },
            Sentiment.FEAR: {
                wn.ADJ: "alarming",
                wn.ADJ_SAT: "fearful",
                wn.ADV: "fearfully",
                wn.NOUN: "fear",
                wn.VERB: "frighten",
            },
            Sentiment.JOY: {
                wn.ADJ: "happy",
                wn.ADJ_SAT: "delighted",
                wn.ADV: "happily",
                wn.NOUN: "happiness",
                wn.VERB: "rejoice",
            },
            Sentiment.SADNESS: {
                wn.ADJ: "sad",
                wn.ADJ_SAT: "depressed",
                wn.ADV: "sadly",
                wn.NOUN: "sadness",
                wn.VERB: "sadden",
            },
            Sentiment.SURPRISE: {
                wn.ADJ: "surprised",
                wn.ADJ_SAT: "startled",
                wn.ADV: "surprisingly",
                wn.NOUN: "surprise",
                wn.VERB: "surprise",
            },
            Sentiment.TRUST: {
                wn.ADJ: "peaceful",
                wn.ADJ_SAT: "calm",
                wn.ADV: "peacefully",
                wn.NOUN: "trust",
                wn.VERB: "trust",
            },
        }

    def _get_synset_for_pos(self, word: str, pos: str) -> Optional[Synset]:
        try:
            synset = wn.synsets(word, pos)[0]
        except KeyError:
            synset = wn.synset(f"{wn.morphy(word, pos)}.{pos}.01")
        except IndexError:
            return None
        return synset

    def _get_synset(self, word: str) -> Optional[Synset]:
        for pos in [wn.ADJ_SAT, wn.ADJ, wn.ADV, wn.NOUN, wn.VERB]:
            possible_synset = self._get_synset_for_pos(word, pos)
            if possible_synset:
                return possible_synset
        return None

    def _compute_res_similarity(self, s1: Synset, s2: Synset) -> float:
        MAX_VALUE = 8.0

        sim = s1.res_similarity(s2, self._brown_ic)
        if sim >= MAX_VALUE:
            return 1
        return sim / MAX_VALUE

    def _compute_lch_similarity(self, s1: Synset, s2: Synset) -> float:
        MAX_VALUE = 3.64

        sim = s1.lch_similarity(s2)
        if sim >= MAX_VALUE:
            return 1
        return sim / MAX_VALUE

    def _compute_jcn_similarity(self, s1: Synset, s2: Synset) -> float:
        MAX_VALUE = 0.5
        MULTIPLIER = 2.0

        sim = s1.jcn_similarity(s2, self._brown_ic)
        if sim >= MAX_VALUE:
            return 1
        return sim * MULTIPLIER

    def _similarity_score(self, s1: Synset, s2: Synset) -> float:
        sims = [
            s1.path_similarity(s2),
            self._compute_lch_similarity(s1, s2),
            s1.wup_similarity(s2),
        ]
        if s1.pos() in (wn.NOUN, wn.VERB):
            sims.extend(
                [
                    self._compute_res_similarity(s1, s2),
                    self._compute_jcn_similarity(s1, s2),
                    s1.lin_similarity(s2, self._brown_ic),
                ]
            )
        return float(sum(sims) / len(sims))

    def tokenize_prompt(self, prompt: str) -> list[str]:
        prompt = re.sub(r"[!@#\$%\^&\*-_=\+,\.;\'\[\]\(\)\\|]", "", prompt)
        words = [w for w in prompt.split() if w.isalpha() and w not in self._stopwords]
        return words

    def get_closest_sentiment(self, words: list[str]) -> Sentiment:
        scores_list: list[dict[Sentiment, float]] = []
        for word in words:
            word_synset = self._get_synset(word)
            if not word_synset:
                continue

            pos = word_synset.pos()
            synsets_to_compare_against = {
                s: wn.synset(f"{self._sentiment_equivalents[s][pos]}.{pos}.01")
                for s in Sentiment
                if s != Sentiment.NEUTRAL
            }
            scores = {
                s: self._similarity_score(word_synset, other_synset)
                for s, other_synset in synsets_to_compare_against.items()
            }
            scores_list.append(scores)

        # TODO - ignore words if their values are smaller than some (other) threshold
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


sentiment_detector = SentimentDetector()
sentence = "happiness and satisfaction should not really give out fear, you know ?"
words = sentiment_detector.tokenize_prompt(sentence)
print(f"Closest sentiment to '{words}' is {sentiment_detector.get_closest_sentiment(words)}")
