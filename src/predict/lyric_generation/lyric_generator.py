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
        seeds = pd.read_csv(f"../data/Datasets/lyrics/seeds.csv", encoding="utf-8")
        filtered_seeds = seeds[seeds["Emotion"] == sentiment.value]
        seed_lines = []
        for line in filtered_seeds["Text"]:
            line = re.sub(r"\(.*\)", "", line)
            line = re.sub(r"[^ a-zA-Z]", "", line)
            seed_lines.append(line)
        return seed_lines

    def run(self, n_verses: int, sentiment: Sentiment) -> list[str]:
        seeds = self._get_seeds_for_sentiment(sentiment)
        current_seed = random.choice(seeds)
        verse_length = 8
        lyrics = []

        i = 0
        while i < n_verses:
            for _ in range(verse_length):
                output_word = self._get_next_word(current_seed)
                current_seed += " " + output_word

            # verse is only compose of one repeated word,
            # or previous verse identical to current one
            if len(set(current_seed.split()[-verse_length:])) <= 2 or (
                i > 1 and current_seed == lyrics[i - 1]
            ):
                current_seed = random.choice(seeds)
            else:
                new_seed = " ".join(current_seed.split()[-verse_length:])
                # TODO - make first char in verse uppercase
                lyrics.append(new_seed)
                current_seed = new_seed
                i += 1

        return lyrics
