from dataclasses import dataclass
import pickle
import numpy as np
import tensorflow.keras.utils as ku
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


@dataclass
class DataContainer:
    x_train: np.array
    x_test: np.array
    y_train: np.array
    y_test: np.array


class DataLoader:
    TOKENIZER_PATH = "Tokenizers/tokenizer_sentiment.pkl"

    def __init__(self) -> None:
        pass

    def filter_lines(self) -> list[str]:
        all_lines = set()
        with open("all_lyrics.txt", "r") as f:
            lines = f.readlines()
            all_lines.update(lines)

        print(len(all_lines))
        count = {}
        for line in all_lines:
            for word in line.split():
                if word not in count:
                    count[word] = 0
                count[word] += 1
        filtered_lines = [l for l in all_lines if all(count[w] >= 50 for w in l.split())]
        filtered_lines = [l for l in filtered_lines if len(l.split()) > 2]
        print(len(filtered_lines))
        print(filtered_lines[:5])
        return filtered_lines

    def _get_new_tokenizer(self, filtered_lines: list[str]) -> Tokenizer:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(filtered_lines)
        with open(self.TOKENIZER_PATH, "wb") as tokenizer_path:
            pickle.dump(tokenizer, tokenizer_path)
        return tokenizer

    @staticmethod
    def load_tokenizer() -> Tokenizer:
        with open(DataLoader.TOKENIZER_PATH, "rb") as tokenizer_path:
            return pickle.load(tokenizer_path)

    def run(self) -> tuple[DataContainer, Tokenizer]:
        features, targets = [], []
        filtered_lines = self.filter_lines()

        for line in filtered_lines:
            token_list = line.split()
            for i in range(1, len(token_list)):
                feature = token_list[:i]
                target = token_list[i]

                features.append(feature)
                targets.append([target])

        x_train, x_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )

        tokenizer = self._get_new_tokenizer(filtered_lines)
        total_words = len(tokenizer.word_index) + 1
        max_sequence_len = max(len(line.split()) for line in filtered_lines)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)
        y_train = tokenizer.texts_to_sequences(y_train)
        y_test = tokenizer.texts_to_sequences(y_test)

        x_train = pad_sequences(x_train, maxlen=max_sequence_len)
        x_test = pad_sequences(x_test, maxlen=max_sequence_len)
        y_train = ku.to_categorical(y_train, num_classes=total_words)
        y_test = ku.to_categorical(y_test, num_classes=total_words)

        return DataContainer(x_train, x_test, y_train, y_test), tokenizer
