from dataclasses import dataclass
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow

from constants import Constants


@dataclass
class DataContainer:
    x: np.array
    y: np.array
    x_train: np.array
    y_train: np.array
    x_seed: np.array
    y_seed: np.array


class DataLoader:
    def __init__(self, lim: int) -> None:
        self._lim = lim

    def get_notes_from_txt(self, txt_path: str) -> list[str]:
        all_notes = []
        with open(txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                all_notes.extend(line.split())
        groups = []
        group_size = 8
        for i in range(0, len(all_notes), group_size):
            groups.append("/".join(all_notes[i : i + group_size]))
        return groups

    def filter_notes(self, all_notes: list[str]) -> list[str]:
        count = {}
        for n in all_notes:
            if n not in count:
                count[n] = 0
            count[n] += 1
        filtered_notes = [n for n in all_notes if count[n] >= self._lim]
        return filtered_notes

    def create_indices(self, filtered_notes: list[str]) -> dict[int, str]:
        unique_notes = sorted(list(set(filtered_notes)))

        index = {note: ind for ind, note in enumerate(unique_notes)}
        reverse_index = dict(enumerate(unique_notes))
        with open(Constants.MUSIC_REVERSE_INDEX_PATH, "wb") as reverse_index_path:
            pickle.dump(reverse_index, reverse_index_path)
        return index

    @staticmethod
    def save_seed_to_disk(data_container: DataContainer) -> None:
        with open(Constants.MUSIC_SEED_PATH, "wb") as music_seed_path:
            np.save(music_seed_path, data_container.x_seed)

    def run(
        self, filtered_notes: list[str], seed_size: float, index: dict[int, str]
    ) -> DataContainer:
        features = []
        targets = []
        for i in range(0, len(filtered_notes) - Constants.MUSIC_FEATURE_LENGTH):
            feature = filtered_notes[i : i + Constants.MUSIC_FEATURE_LENGTH]
            target = filtered_notes[i + Constants.MUSIC_FEATURE_LENGTH]
            features.append([index[j] for j in feature])
            targets.append(index[target])

        n_datapoints = len(targets)

        vocab_size = len(index)
        x = (
            np.reshape(features, (n_datapoints, Constants.MUSIC_FEATURE_LENGTH, 1), order="C")
        ) / float(vocab_size)
        y = tensorflow.keras.utils.to_categorical(targets)
        x_train, x_seed, y_train, y_seed = train_test_split(
            x, y, test_size=seed_size, random_state=42
        )
        return DataContainer(x, y, x_train, y_train, x_seed, y_seed)
