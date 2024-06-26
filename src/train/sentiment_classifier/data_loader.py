import pickle
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from constants import Constants
from sentiment import Sentiment
from train.utils import DataContainer


class DataLoader:
    def _load_csv_files(self) -> pd.DataFrame:
        """Loads sentence dataset from CSV files

        Returns:
            pd.DataFrame: Combined dataframe of all CSV files
        """
        return pd.concat(
            [
                pd.read_csv(f"../data/Datasets/sentiment/{csv_name}", encoding="utf-8")
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
        """Randomly samples only 15000 neutral sentences, in order to normalize
        the dataset

        Args:
            initial_df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Dataframe with only 15000 neutral sentences
        """
        neutral_mask = initial_df["Emotion"] == "neutral"
        only_neutral = initial_df[neutral_mask]
        data = initial_df[~neutral_mask]
        only_neutral = only_neutral.sample(n=15000, random_state=42)
        return pd.concat([data, only_neutral], ignore_index=True)

    def _remove_unused_sentiments(
        self, data: pd.DataFrame, unused_sentiments: tuple[str] = ("shame", "guilt")
    ) -> pd.DataFrame:
        """Removes unused sentiments from the input data

        Args:
            data (pd.DataFrame): Original dataframe
            unused_sentiments (tuple[str], optional): Sentiments whose sentences have to be removed.
            Defaults to ("shame", "guilt").

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        for s in unused_sentiments:
            mask = data["Emotion"] == s
            data = data[~mask]
        return data

    @staticmethod
    def _clean_text(data: str) -> list[str]:
        """Removes words with unnecessary characters from input text

        Args:
            data (str): Input text

        Returns:
            list[str]: Words from cleaned text
        """
        data = re.sub(r"(#[\d\w\.]+)", "", data)
        data = re.sub(r"(@[\d\w\.]+)", "", data)

        return data.split()

    def _get_new_tokenizer(self, all_sentences: list[str]) -> Tokenizer:
        """Creates a new tokenizer, fits it on the provided verses, and saves it to disk

        Args:
            all_sentences (list[str]): Verses with which the tokenizers works

        Returns:
            Tokenizer: Fitted tokenizer instance
        """
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_sentences)
        with open(Constants.SENTIMENT_TOKENIZER_PATH, "wb") as tokenizer_path:
            pickle.dump(tokenizer, tokenizer_path)
        return tokenizer

    def run(self) -> tuple[DataContainer, Tokenizer]:
        """Loads data, cleans and normalizes it, splits it into feature/targets,
        then into training/test. This is then converted to numeric form using a tokenizer.

        Returns:
            tuple[DataContainer, Tokenizer]: The resulting data splits and the new tokenizer
        """
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

        x_train_pad = pad_sequences(sequence_train, maxlen=Constants.SENTIMENT_MAX_SEQ_LEN)
        x_test_pad = pad_sequences(sequence_test, maxlen=Constants.SENTIMENT_MAX_SEQ_LEN)

        encoding = {sentiment: idx for idx, sentiment in enumerate(Sentiment.class_names())}
        y_train = to_categorical([encoding[x] for x in y_train])
        y_test = to_categorical([encoding[x] for x in y_test])

        return DataContainer(x_train_pad, x_test_pad, y_train, y_test), tokenizer
