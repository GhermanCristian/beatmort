import numpy as np
from sentiment_detector.sentiment import Sentiment
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model


class SentimentDetector:
    def __init__(
        self, tokenizer: Tokenizer, model: Model, max_seq_len: int, class_names: list[str]
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._max_seq_len = max_seq_len
        self._class_names = class_names
        # TODO - replace class names with iteration over Sentiment (order might be different => maybe retrain)

    def run(self, prompt: str) -> Sentiment:
        seq = self._tokenizer.texts_to_sequences(prompt)
        padded = pad_sequences(seq, maxlen=self._max_seq_len)

        prediction = self._model.predict(padded)

        sentiment = self._class_names[np.argmax(prediction)]
        print(f"Prompt = {prompt}; sentiment = {sentiment}")
        return Sentiment[sentiment]
