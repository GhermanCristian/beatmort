from dataclasses import dataclass
from pathlib import Path
import numpy as np
import urllib.request
import zipfile


@dataclass
class DataContainer:
    x_train_pad: np.array
    x_test_pad: np.array
    y_train: np.array
    y_test: np.array


class Utils:
    EMBEDDING_MATRIX_NAME = "wiki-news-300d-1M.vec"
    EMBEDDINGS_DIR = "../data/embeddings"
    EMBEDDING_MATRIX_PATH = Path(f"{EMBEDDINGS_DIR}/{EMBEDDING_MATRIX_NAME}")
    N_DIMS_EMBEDDING = 300

    @staticmethod
    def download_embedding_matrix() -> None:
        """Downloads embedding matrix and extracts the zip archive, which is then deleted"""
        urllib.request.urlretrieve(
            f"https://dl.fbaipublicfiles.com/fasttext/vectors-english/{Utils.EMBEDDING_MATRIX_NAME}.zip",
            f"{Utils.EMBEDDING_MATRIX_PATH}.zip",
        )
        with zipfile.ZipFile(f"{Utils.EMBEDDING_MATRIX_PATH}.zip", "r") as zip_ref:
            zip_ref.extractall(Utils.EMBEDDINGS_DIR)

        Path(f"{Utils.EMBEDDING_MATRIX_PATH}.zip").unlink()

    @staticmethod
    def load_embedding_matrix(vocabulary_size: int, word_index: dict[str, int]) -> list[np.ndarray]:
        """Loads the embeddings of the words available in the index

        Args:
            vocabulary_size (int): Number of words in the resulting matrix
            word_index (dict[str, int]): Mapping of words to numerical indices

        Returns:
            list[np.ndarray]: Word embedding matrix
        """
        embedding_matrix = np.zeros((vocabulary_size, Utils.N_DIMS_EMBEDDING))
        with open(
            Utils.EMBEDDING_MATRIX_PATH, "r", encoding="utf-8", newline="\n", errors="ignore"
        ) as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word]
                    embedding_matrix[idx] = np.array(vector, dtype=np.float32)[
                        : Utils.N_DIMS_EMBEDDING
                    ]
        return embedding_matrix
