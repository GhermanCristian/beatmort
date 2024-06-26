from music21 import chord
from pathlib import Path

from parser.dataset_parser import DatasetParser


class JSBParser(DatasetParser):
    def _piano_key_to_midi(self, piano_key: int) -> int:
        """Converts the note of a piano key to its MIDI value

        Args:
            piano_key (int): Piano key note value (from 0-88)

        Returns:
            int: MIDI value
        """
        return piano_key + 20

    def _convert_piano_key_numbers_to_chord(self, chord_as_piano_keys: str) -> str:
        """Creates a chord, given the piano keys

        Args:
            chord_as_piano_keys (str): Piano key note values

        Returns:
            str: Chord in C scale notation
        """
        chord_notes = [self._piano_key_to_midi(int(n)) for n in chord_as_piano_keys.split(",")]
        if 20 in chord_notes:
            chord_notes.remove(20)
        return ".".join(str(n) for n in chord.Chord(chord_notes).normalOrder)

    def _convert_piano_key_folder_to_chords(self, dir: Path, output_file: str) -> None:
        """Creates chords for all piano keys in the given folder

        Args:
            dir (Path): Root directory
            output_file (str): File where the chords are outputted
        """
        BATCH_SIZE = 10

        files = dir.rglob("*.csv")
        for i in range(0, len(files), BATCH_SIZE):
            chords = []
            for file in files[i : i + BATCH_SIZE]:
                try:
                    with open(str(file), "r") as f:
                        lines = f.readlines()[1:]
                        chords.extend(
                            [self._convert_piano_key_numbers_to_chord(line[:-1]) for line in lines]
                        )
                except Exception as e:
                    print(e)
            self._save_to_txt_file(chords, output_file)

    def run(self) -> None:
        for purpose in ["train", "test", "valid"]:
            self._convert_piano_key_folder_to_chords(
                Path(f"Datasets\\JSB\\{purpose}"), f"jsb_{purpose}"
            )
