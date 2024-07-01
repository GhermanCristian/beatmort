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
        fluidsynth_exe: Optional[Path],
        soundfont: Optional[Path],
    ) -> None:
        SongSaver._save_midi_to_disk(main_score, output_name)
        SongSaver._save_audio_to_disk(output_name, fluidsynth_exe, soundfont)
