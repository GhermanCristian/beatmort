import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History

import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import RMSprop

from constants import Constants
from train.utils import DataContainer, Utils


class ModelCreator:
    def __init__(
        self,
        word_index: dict[str, int],
        batch_size: int,
        data_container: DataContainer,
    ) -> None:
        self._word_index = word_index
        self._vocabulary_size = len(word_index) + 1
        self._batch_size = batch_size
        self._data_container = data_container

    def _create_model(
        self, embedding_matrix: list[np.ndarray], n_units: int, dropout_rate: float
    ) -> Model:
        """Creates a new model

        Args:
            embedding_matrix (list[np.ndarray]): Pre-trained matrix with word embeddings
            n_units (int): Number of units in the first LSTM layer
            dropout_rate (float): Dropout rate

        Returns:
            Model: Compiled model (not yet trained)
        """
        model = Sequential()
        model.add(
            Embedding(
                self._vocabulary_size,
                Utils.N_DIMS_EMBEDDING,
                input_length=Constants.LYRICS_MAX_SEQ_LEN,
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

    def get_new_model(self, n_units: int = 128, dropout_rate: float = 0.35) -> Model:
        """Creates new model. If necessary, the embedding matrix is downloaded and loaded into memory

        Args:
            n_units (int, optional): Number of units in the first LSTM layer. Defaults to 128.
            dropout_rate (float, optional): Dropout layer. Defaults to 0.35.

        Returns:
            Model: Compiled model (not yet trained)
        """
        if not Utils.EMBEDDING_MATRIX_PATH.exists():
            Utils.download_embedding_matrix()
        embedding_matrix = Utils.load_embedding_matrix(self._vocabulary_size, self._word_index)
        return self._create_model(embedding_matrix, n_units, dropout_rate)

    def train_model(self, model: Model, n_epochs: int) -> History:
        """Trains a model for a number of epochs. The best model (defined by its validation
        accuracy) is saved after every epoch. Moreover, the process stops if no progress is
        made for 100 epochs. Training starts in epoch 0.

        Args:
            model (Model): Model that is trained
            n_epochs (int): Number of epochs that the model is trained for.

        Returns:
            History: Contains the progression of the main training metrics (loss, accuracy, ..)
        """
        checkpoint = ModelCheckpoint(
            Constants.LYRICS_MODEL_PATH,
            monitor="val_categorical_accuracy",
            save_best_only=True,
            save_freq="epoch",
            verbose=2,
            initial_value_threshold=0,
        )
        stop_early = EarlyStopping(monitor="val_categorical_accuracy", patience=100)

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
        """Evaluates the model on the existing testing dataset split

        Args:
            model (Model): Model that is evaluated

        Returns:
            list[str]: Validation loss and accuracy
        """
        result = model.evaluate(
            self._data_container.x_test,
            self._data_container.y_test,
            batch_size=self._batch_size,
        )
        return result
