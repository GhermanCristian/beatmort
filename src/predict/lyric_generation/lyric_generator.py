import re
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import pandas as pd
from sentiment import Sentiment


class LyricGenerator:
    def __init__(self, max_sequence_len: int, tokenizer: Tokenizer, model: Model) -> None:
        self._max_sequence_len = max_sequence_len
        self._tokenizer = tokenizer
        self._model = model

    def _get_next_word(self, current_seed: str) -> str:
        """Given a seed, determines the next word. Some degree of randomness is used to ensure
        non-determinism.

        Args:
            current_seed (str): Current seed word

        Returns:
            str: Next word to be used
        """
        token_list = self._tokenizer.texts_to_sequences([current_seed])[0]
        token_list = pad_sequences([token_list], maxlen=self._max_sequence_len, padding="pre")
        prediction = self._model.predict(token_list, verbose=0)[0]

        # choose a random prediction which is at least half of the most probable one
        pred_with_pos = [(pred, idx) for idx, pred in enumerate(prediction)]
        sorted_pred = sorted(pred_with_pos, key=lambda pair: pair[0], reverse=True)
        max_position = 1
        while (
            max_position < self._max_sequence_len - 1
            and sorted_pred[0][0] <= 1.75 * sorted_pred[max_position + 1][0]
        ):
            max_position += 1
        random_idx = random.randint(0, max_position - 1) if max_position > 1 else 0
        pos = sorted_pred[random_idx][1]

        for word, index in self._tokenizer.word_index.items():
            if index == pos:
                return word
        return ""

    def _get_seeds_for_sentiment(self, sentiment: Sentiment) -> list[str]:
        """Reads list of seeds for a given sentiment and parses them

        Args:
            sentiment (Sentiment): The sentiment whose seeds are retrieved

        Returns:
            list[str]: List of parsed seeds
        """
        seeds = pd.read_csv(f"../data/Seeds/lyrics_seeds.csv", encoding="utf-8")
        filtered_seeds = seeds[seeds["Emotion"] == sentiment.value]
        seed_lines = []
        for line in filtered_seeds["Text"]:
            line = re.sub(r"\(.*\)", "", line)
            line = re.sub(r"[^ a-zA-Z]", "", line)
            seed_lines.append(line)
        return seed_lines

    def _beautify_verse(self, verse: str) -> str:
        """Improves the content of a verse in order to make it more suitable

        Args:
            verse (str): Verse which may be modified

        Returns:
            str: Modified verse
        """
        for pair in [
            ("i", "I"),
            ("im", "I'm"),
            ("id", "I'd"),
            ("aint", "ain't"),
            ("dont", "don't"),
            ("cant", "can't"),
            ("dick", "d**k"),
            ("fuck", "f**k"),
            ("shit", "s**t"),
            ("bitch", "b***h"),
        ]:
            verse = re.sub(rf"\b{pair[0]}\b", pair[1], verse)
        return verse

    def run(self, n_verses: int, sentiment: Sentiment) -> list[str]:
        """Creates a number of verses following a particular sentiment

        Args:
            n_verses (int): Number of output verses
            sentiment (Sentiment): Sentiment which the verses follow

        Returns:
            list[str]: List of verses
        """
        seeds = self._get_seeds_for_sentiment(sentiment)
        current_seed = random.choice(seeds)
        verse_length = 8
        lyrics = []

        i = 0
        while i < n_verses:
            for _ in range(verse_length):
                output_word = self._get_next_word(current_seed)
                current_seed += " " + output_word

            # verse has less than 3 unique words,
            # or previous verse identical to current one
            if len(set(current_seed.split()[-verse_length:])) <= 2 or (
                i > 1 and current_seed == lyrics[i - 1]
            ):
                current_seed = random.choice(seeds)
            else:
                new_seed = " ".join(current_seed.split()[-verse_length:])
                verse = self._beautify_verse(new_seed)
                # capitalize first letter, without modifying the others
                verse = verse[:1].upper() + verse[1:]
                lyrics.append(verse)
                current_seed = new_seed
                i += 1

        return lyrics
