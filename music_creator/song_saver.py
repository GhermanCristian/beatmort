from pathlib import Path
from typing import Optional
from music21 import stream
import subprocess


class SongSaver:
    @staticmethod
    def _save_midi_to_disk(
        main_score: stream.Score,
        output_name: str,
    ) -> None:
        main_score.write("midi", f"{output_name}.mid")

    @staticmethod
    def _save_score_to_disk(
        main_score: stream.Score,
        output_name: str,
        musescore_exe: Optional[Path],
    ) -> None:
        if musescore_exe:
            try:
                xml_path = main_score.write("musicxml", f"{output_name}.xml")
                subprocess.run(
                    [str(musescore_exe), str(xml_path), "-o", f"{output_name}.png"], check=True
                )
            except Exception:
                print("Could not write score")

    @staticmethod
    def _save_audio_to_disk(
        output_name: str,
        fluidsynth_exe: Optional[Path],
        soundfont: Optional[Path],
    ) -> None:
        if fluidsynth_exe and soundfont:
            try:
                subprocess.run(
                    [
                        str(fluidsynth_exe),
                        str(soundfont),
                        f"{output_name}.mid",
                        "-F",
                        f"{output_name}.wav",
                        "-r",
                        "44100",
                        "-q",
                    ],
                    check=True,
                )
            except:
                print("Could not convert to audio")

    @staticmethod
    def save_song_to_disk(
        main_score: stream.Score,
        output_name: str,
        musescore_exe: Optional[Path],
        fluidsynth_exe: Optional[Path],
        soundfont: Optional[Path],
    ) -> None:
        SongSaver._save_midi_to_disk(main_score, output_name)
        SongSaver._save_score_to_disk(main_score, output_name, musescore_exe)
        SongSaver._save_audio_to_disk(output_name, fluidsynth_exe, soundfont)
