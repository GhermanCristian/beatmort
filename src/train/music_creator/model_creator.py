from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import CategoricalAccuracy

from constants import Constants
from music_creator.data_loader import DataContainer


class ModelCreator:
    def __init__(
        self,
        validation_size: float,
        batch_size: int,
        learning_rate: float,
        data_container: DataContainer,
    ) -> None:
        self._validation_size = validation_size
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._data_container = data_container

    def create_model(self, first_layer_units: int = 512, dropout_rate: float = 0.25) -> Model:
        model = Sequential()
        model.add(
            LSTM(
                first_layer_units,
                input_shape=(self._data_container.x.shape[1], self._data_container.x.shape[2]),
                return_sequences=True,
            )
        )
        model.add(Dropout(dropout_rate))
        model.add(LSTM(first_layer_units // 2))
        model.add(Dense(first_layer_units // 2))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self._data_container.y.shape[1], activation="softmax"))
        optimizer = Adamax(learning_rate=self._learning_rate)
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=[CategoricalAccuracy()]
        )

        return model

    def train_model(self, model: Model, n_epochs: int) -> History:
        checkpoint = ModelCheckpoint(
            Constants.MUSIC_MODEL_PATH,
            monitor="val_categorical_accuracy",
            save_best_only=True,
            save_freq="epoch",
            verbose=2,
            initial_value_threshold=0,
        )
        stop_early = EarlyStopping(monitor="val_categorical_accuracy", patience=n_epochs // 2)
        history = model.fit(
            self._data_container.x_train,
            self._data_container.y_train,
            batch_size=self._batch_size,
            epochs=n_epochs,
            callbacks=[checkpoint, stop_early],
            validation_split=self._validation_size,
        )
        return history

    def evaluate_model(self, model: Model) -> list[str]:
        result = model.evaluate(
            self._data_container.x_seed, self._data_container.y_seed, batch_size=self._batch_size
        )
        return result
