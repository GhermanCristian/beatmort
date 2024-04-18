import pickle
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from dataclasses import dataclass

from sentiment_classifier.sentiment import Sentiment


@dataclass
class DataContainer:
    x_train_pad: np.array
    x_test_pad: np.array
    y_train: np.array
    y_test: np.array


class DataLoader:
    TOKENIZER_PATH = "Tokenizers/tokenizer_sentiment.pkl"

    def __init__(self, max_seq_len: int) -> None:
        self._max_seq_len = max_seq_len

    def _load_csv_files(self) -> pd.DataFrame:
        return pd.concat(
            [
                pd.read_csv(f"Datasets/sentiment/{csv_name}", encoding="utf-8")
                for csv_name in [
                    "isear.csv",
                    "emotion-stimulus.csv",
                    "dailydialog.csv",
                    "anticipation.csv",
                    "trust.csv",
                ]
            ],
            ignore_index=True,
        )

    def _normalize_neutral(self, initial_df: pd.DataFrame) -> pd.DataFrame:
        neutral_mask = initial_df["Emotion"] == "neutral"
        only_neutral = initial_df[neutral_mask]
        data = initial_df[~neutral_mask]
        only_neutral = only_neutral.sample(n=15000, random_state=42)
        return pd.concat([data, only_neutral], ignore_index=True)

    def _remove_unused_sentiments(
        self, data: pd.DataFrame, unused_sentiments: tuple[str] = ("shame", "guilt")
    ) -> pd.DataFrame:
        for s in unused_sentiments:
            mask = data["Emotion"] == s
            data = data[~mask]
        return data

    @staticmethod
    def _clean_text(data: str) -> list[str]:
        data = re.sub(r"(#[\d\w\.]+)", "", data)
        data = re.sub(r"(@[\d\w\.]+)", "", data)

        return data.split()

    def _get_new_tokenizer(self, all_sentences: list[str]) -> Tokenizer:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_sentences)
        with open(self.TOKENIZER_PATH, "wb") as tokenizer_path:
            pickle.dump(tokenizer, tokenizer_path)
        return tokenizer

    @staticmethod
    def load_tokenizer() -> Tokenizer:
        with open(DataLoader.TOKENIZER_PATH, "rb") as tokenizer_path:
            return pickle.load(tokenizer_path)

    def run(self) -> tuple[DataContainer, Tokenizer]:
        data = self._remove_unused_sentiments(self._normalize_neutral(self._load_csv_files()))

        x_train, x_test, y_train, y_test = train_test_split(
            data.Text, data.Emotion, test_size=0.25, random_state=42
        )

        all_sentences = [" ".join(self._clean_text(text)) for text in data.Text]
        sentences_train = [" ".join(self._clean_text(text)) for text in x_train]
        sentences_test = [" ".join(self._clean_text(text)) for text in x_test]

        tokenizer = self._get_new_tokenizer(all_sentences)
        sequence_train = tokenizer.texts_to_sequences(sentences_train)
        sequence_test = tokenizer.texts_to_sequences(sentences_test)

        x_train_pad = pad_sequences(sequence_train, maxlen=self._max_seq_len)
        x_test_pad = pad_sequences(sequence_test, maxlen=self._max_seq_len)

        encoding = {sentiment: idx for idx, sentiment in enumerate(Sentiment.class_names())}
        y_train = to_categorical([encoding[x] for x in y_train])
        y_test = to_categorical([encoding[x] for x in y_test])

        return DataContainer(x_train_pad, x_test_pad, y_train, y_test), tokenizer
