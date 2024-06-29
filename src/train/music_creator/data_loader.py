import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow

from constants import Constants
from train.utils import DataContainer


class DataLoader:
    def __init__(self, lim: int) -> None:
        self._lim = lim

    def get_notes_from_txt(self, txt_path: str) -> list[str]:
        """Extracts notes and chords from txt file, in groups of 8

        Args:
            txt_path (str): File which contains notes and chords

        Returns:
            list[str]: List of note and chord groups
        """
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
        """Filters note groups based on their frequency

        Args:
            all_notes (list[str]): All note groups

        Returns:
            list[str]: List of filtered notes
        """
        count = {}
        for n in all_notes:
            if n not in count:
                count[n] = 0
            count[n] += 1
        filtered_notes = [n for n in all_notes if count[n] >= self._lim]
        return filtered_notes

    def create_indices(self, filtered_notes: list[str]) -> dict[str, int]:
        """Creates indices of the note groups (similar to a tokenizer).
        A reverse index is also saved to disk

        Args:
            filtered_notes (list[str]): Note groups

        Returns:
            dict[str, int]: Index from note to its numerical value
        """
        unique_notes = sorted(list(set(filtered_notes)))

        index = {note: ind for ind, note in enumerate(unique_notes)}
        reverse_index = dict(enumerate(unique_notes))
        with open(Constants.MUSIC_REVERSE_INDEX_PATH, "wb") as reverse_index_path:
            pickle.dump(reverse_index, reverse_index_path)
        return index

    @staticmethod
    def save_seed_to_disk(data_container: DataContainer) -> None:
        """Saves seed to disk

        Args:
            data_container (DataContainer): Path where seed is saved
        """
        with open(Constants.MUSIC_SEED_PATH, "wb") as music_seed_path:
            np.save(music_seed_path, data_container.x_seed)

    def run(
        self, filtered_notes: list[str], seed_size: float, index: dict[str, int]
    ) -> DataContainer:
        """Computes feature/targets, splits data into test/train, and converts it
        to numerical form

        Args:
            filtered_notes (list[str]): All data
            seed_size (float): Data percentage which will be used as seed
            index (dict[str, int]): Maps note groups to a numerical value

        Returns:
            DataContainer: All necessary data sections
        """
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
        return DataContainer(x_train, y_train, x_seed, y_seed)
