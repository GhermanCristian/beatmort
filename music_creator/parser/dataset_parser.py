from abc import ABC, abstractmethod


class DatasetParser(ABC):
    def _save_to_txt_file(self, notes: list[str], file_name: str) -> None:
        with open(file_name, "a+") as f:
            line = " ".join(notes)
            f.write(line + "\n")

    @abstractmethod
    def run(self) -> None:
        pass
