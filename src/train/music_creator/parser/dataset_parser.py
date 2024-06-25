from abc import ABC, abstractmethod


class DatasetParser(ABC):
    def _save_to_txt_file(self, notes: list[str], file_name: str) -> None:
        """Saves notes and chords to a txt file

        Args:
            notes (list[str]): List of notes and chords
            file_name (str): File where sounds are saved
        """
        with open(file_name, "a+") as f:
            line = " ".join(notes)
            f.write(line + "\n")

    @abstractmethod
    def run(self) -> None:
        pass
