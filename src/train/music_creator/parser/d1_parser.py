from pathlib import Path
from music21 import chord, converter, instrument, note, stream

from parser.dataset_parser import DatasetParser


class D1Parser(DatasetParser):
    def _extract_notes(self, converted_midi_files: list[stream.Stream]) -> list[str]:
        """Extracts notes and chords from a list of converted MIDI files 

        Args:
            converted_midi_files (list[stream.Stream]): List of MIDI files that have
                been parsed with `converter.parse()`

        Returns:
            list[str]: List of notes and chords from all input files
        """
        notes = []
        for file in converted_midi_files:
            songs = instrument.partitionByInstrument(file)
            for part in songs.parts:
                pick = part.recurse()
                for element in pick:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append(".".join(str(n) for n in element.normalOrder))
        return notes

    def _get_all_files_in_directory(self, root_dir: Path) -> list[Path]:
        """Recursively gets all MIDI files in the directory. These can have either
        the .mid or the .midi extension.

        Args:
            root_dir (Path): Root directory

        Returns:
            list[Path]: List of all MIDI file paths
        """
        return list(root_dir.rglob("*.midi")) + list(root_dir.rglob("*.mid"))

    def run(self) -> None:
        """Finds all MIDI files in the dataset directory, converts them to objects so that
        their notes and chords can be extracted, then outputs those notes to a file on disk"""
        BATCH_SIZE = 10
        ROOT_DIR = Path("Datasets")

        midi_files = self._get_all_files_in_directory(ROOT_DIR)
        for i in range(0, len(midi_files), BATCH_SIZE):
            converted_midi_files = [
                converter.parse(midi) for midi in midi_files[i : i + BATCH_SIZE]
            ]
            all_notes = self._extract_notes(converted_midi_files)
            self._save_to_txt_file(all_notes, "d1_parsed.txt")
