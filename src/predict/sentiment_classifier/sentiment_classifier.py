import numpy as np
from sentiment import Sentiment
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model


class SentimentClassifier:
    CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, tokenizer: Tokenizer, model: Model, max_seq_len: int) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._max_seq_len = max_seq_len

    def run(self, prompt: str) -> Sentiment:
        """Given a prompt, classifies the sentiment expressed in it. If the prediction
        confidence is not higher than a given threshold, NEUTRAL is used by default

        Args:
            prompt (str): Prompt that is analysed

        Returns:
            Sentiment: Sentiment expressed in the prompt
        """
        seq = self._tokenizer.texts_to_sequences([prompt])
        padded = pad_sequences(seq, maxlen=self._max_seq_len)

        prediction = self._model.predict(padded)

        max_index = np.argmax(prediction)
        prediction_confidence = prediction[0][max_index]
        if prediction_confidence >= self.CONFIDENCE_THRESHOLD:
            sentiment = Sentiment.class_names()[max_index]
        else:
            sentiment = Sentiment.NEUTRAL
        return Sentiment[sentiment.upper()]
