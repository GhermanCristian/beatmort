import numpy as np

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History

import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import RMSprop

from lyric_generation.data_loader import DataContainer
from utils import Utils


class ModelCreator:
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
        if not Utils.EMBEDDING_MATRIX_PATH.exists():
            Utils.download_embedding_matrix()
        embedding_matrix = Utils.load_embedding_matrix(
            self._vocabulary_size, self._n_dims_embedding, self._word_index
        )
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
