from pathlib import Path
import numpy as np

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History

import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from pathlib import Path
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import RMSprop

from lyric_generation.data_loader import DataContainer


class ModelCreator:
    EMBEDDING_MATRIX_NAME = "wiki-news-300d-1M.vec"
    EMBEDDINGS_DIR = "embeddings"
    EMBEDDING_MATRIX_PATH = Path(f"{EMBEDDINGS_DIR}/{EMBEDDING_MATRIX_NAME}")

    def __init__(
        self,
        n_dims_embedding: int,
        max_seq_len: int,
        num_classes: int,
        word_index,
        batch_size: int,
        data_container: DataContainer,
    ) -> None:
        self._n_dims_embedding = n_dims_embedding
        self._max_seq_len = max_seq_len
        self._num_classes = num_classes
        self._word_index = word_index
        self._vocabulary_size = len(word_index) + 1
        self._batch_size = batch_size
        self._data_container = data_container

    def _load_embedding_matrix(self) -> list[np.ndarray]:
        embedding_matrix = np.zeros((self._vocabulary_size, self._n_dims_embedding))
        with open(
            self.EMBEDDING_MATRIX_PATH, "r", encoding="utf-8", newline="\n", errors="ignore"
        ) as f:
            for line in f:
                word, *vector = line.split()
                if word in self._word_index:
                    idx = self._word_index[word]
                    embedding_matrix[idx] = np.array(vector, dtype=np.float32)[
                        : self._n_dims_embedding
                    ]
        return embedding_matrix

    def _create_model(
        self, embedding_matrix: list[np.ndarray], n_units: int, dropout_rate: float
    ) -> Model:
        model = Sequential()
        model.add(
            Embedding(
                self._vocabulary_size,
                self._n_dims_embedding,
                input_length=self._max_seq_len,
                weights=[embedding_matrix],
                trainable=False,
            )
        )
        model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(n_units // 2))
        model.add(Dense(self._vocabulary_size, activation="softmax"))
        optimizer = RMSprop(learning_rate=0.005)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy", CategoricalAccuracy()],
        )
        return model

    @staticmethod
    def _model_name() -> str:
        return f"Models/model_lyrics.keras"

    @staticmethod
    def load_model() -> Model:
        return load_model(ModelCreator._model_name())

    def get_new_model(self, n_units: int = 128, dropout_rate: float = 0.35) -> Model:
        if not self.EMBEDDING_MATRIX_PATH.exists():
            self._download_embedding_matrix()
        embedding_matrix = self._load_embedding_matrix()
        return self._create_model(embedding_matrix, n_units, dropout_rate)

    def train_model(self, model: Model, n_epochs: int) -> History:
        checkpoint = ModelCheckpoint(
            self._model_name(),
            monitor="val_categorical_accuracy",
            save_best_only=True,
            save_freq="epoch",
            verbose=2,
            initial_value_threshold=0,
        )
        stop_early = EarlyStopping(monitor="val_categorical_accuracy", patience=90)

        history = model.fit(
            self._data_container.x_train_pad,
            self._data_container.y_train,
            batch_size=self._batch_size,
            epochs=n_epochs,
            callbacks=[checkpoint, stop_early],
            validation_data=(self._data_container.x_test, self._data_container.y_test),
        )

        return history

    def evaluate_model(self, model: Model) -> list[str]:
        result = model.evaluate(
            self._data_container.x_test,
            self._data_container.y_test,
            batch_size=self._batch_size,
        )
        return result
