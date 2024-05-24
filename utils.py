from pathlib import Path
import numpy as np
import urllib.request
import zipfile


class Utils:
    EMBEDDING_MATRIX_NAME = "wiki-news-300d-1M.vec"
    EMBEDDINGS_DIR = "embeddings"
    EMBEDDING_MATRIX_PATH = Path(f"{EMBEDDINGS_DIR}/{EMBEDDING_MATRIX_NAME}")

    def download_embedding_matrix(self) -> None:
        urllib.request.urlretrieve(
            f"https://dl.fbaipublicfiles.com/fasttext/vectors-english/{self.EMBEDDING_MATRIX_NAME}.zip",
            f"{self.EMBEDDING_MATRIX_PATH}.zip",
        )
        with zipfile.ZipFile(f"{self.EMBEDDING_MATRIX_PATH}.zip", "r") as zip_ref:
            zip_ref.extractall(self.EMBEDDINGS_DIR)

        Path(f"{self.EMBEDDING_MATRIX_PATH}.zip").unlink()

    def load_embedding_matrix(
        self, vocabulary_size: int, n_dims_embedding: int, word_index: dict[str, int]
    ) -> list[np.ndarray]:
        embedding_matrix = np.zeros((vocabulary_size, n_dims_embedding))
        with open(
            self.EMBEDDING_MATRIX_PATH, "r", encoding="utf-8", newline="\n", errors="ignore"
        ) as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word]
                    embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:n_dims_embedding]
        return embedding_matrix
