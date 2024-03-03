from pathlib import Path
from music21 import chord, converter, instrument, note, stream


class D1Parser:
    def _save_to_txt_file(self, notes: list[str], file_name: str) -> None:
        with open(file_name, "a+") as f:
            line = " ".join(notes)
            f.write(line + "\n")

    def _extract_notes(self, converted_midi_files: list[stream.Stream]) -> list[str]:
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
        return list(root_dir.rglob("*.midi")) + list(root_dir.rglob("*.mid"))

    def run(self) -> None:
        BATCH_SIZE = 10
        ROOT_DIR = Path("Datasets")

        midi_files = self._get_all_files_in_directory(ROOT_DIR)
        for i in range(0, len(midi_files), BATCH_SIZE):
            converted_midi_files = [
                converter.parse(midi) for midi in midi_files[i : i + BATCH_SIZE]
            ]
            all_notes = self._extract_notes(converted_midi_files)
            self._save_to_txt_file(all_notes, "d1_parsed.txt")
            print(f"Finished batch {i} - {i + BATCH_SIZE}")
