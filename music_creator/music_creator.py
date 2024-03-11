import copy
from pathlib import Path
from typing import Optional
from music21 import chord, note, stream, duration
import numpy as np
from tensorflow.keras.models import Model
import subprocess

from music_creator.sentiment_to_melodies import MelodyInfo


class MusicCreator:
    def __init__(
        self,
        model: Model,
        x_seed: np.array,
        feature_length: int,
        vocab_size: int,
        reverse_index: dict[int, str],
    ) -> None:
        self._model = model
        self._x_seed = x_seed
        self._feature_length = feature_length
        self._vocab_size = vocab_size
        self._reverse_index = reverse_index

    def _chords_n_notes(
        self,
        measures: list[str],
        duration_quarter_length: float,
        offset: float,
        vol: float,
        pause_duration: float,
    ):
        melody = []
        offset: float = offset
        dur = duration.Duration(duration_quarter_length)
        for b in measures:
            sounds = b.split("/")
            for s in sounds:
                if "." in s or s.isdigit():
                    chord_notes = s.split(".")
                    notes = []
                    for j in chord_notes:
                        inst_note = int(j)
                        note_snip = note.Note(inst_note)
                        note_snip.volume.velocity = 127.0 * vol
                        notes.append(note_snip)
                    chord_snip = chord.Chord(notes, duration=dur)
                    chord_snip.offset = offset
                    melody.append(chord_snip)
                else:
                    note_snip = note.Note(s)
                    note_snip.offset = offset
                    melody.append(note_snip)
                if pause_duration:
                    p = note.Rest(duration=duration.Duration(pause_duration))
                    p.offset = offset
                    melody.append(p)
                offset += duration_quarter_length + pause_duration
        final_note_index = -1
        if isinstance(melody[final_note_index], note.Rest):
            final_note_index -= 1
        final_note = copy.deepcopy(melody[final_note_index])
        final_note.duration = duration.Duration(4)
        final_note.offset = offset
        final_note.volume.velocity = 127.0
        melody.append(final_note)
        return melody

    def _measure_generator(self, measure_length: int = 8):
        seed = self._x_seed[np.random.randint(0, len(self._x_seed) - 1)][:]
        notes = []
        for _ in range(measure_length):
            seed = seed.reshape(1, self._feature_length, 1)
            prediction = self._model.predict(seed, verbose=0)[0]
            prediction = np.log(prediction) / 0.25
            exp_preds = np.exp(prediction)
            prediction = exp_preds / np.sum(exp_preds)
            pos = np.argmax(prediction)
            pos_n = pos / float(self._vocab_size)
            notes.append(pos)
            seed = np.insert(seed[0], len(seed[0]), pos_n)
            seed = seed[1:]
        measure = [self._reverse_index[char] for char in notes]
        return measure

    def _melody_generator(self, song_length: int, dur: float, measure_length: int = 8):
        measure = self._measure_generator(measure_length)
        n_measures_in_song = int(song_length // dur // measure_length // 8)
        music = measure * n_measures_in_song
        # TODO - think of cases when song_length % (dur * measure_length) != 0
        return music

    def _compose_melody(self, song_length: int, melody_info: MelodyInfo) -> stream.Part:
        notes = self._melody_generator(song_length, melody_info.dur, melody_info.measure_length)
        print(notes)
        melody = self._chords_n_notes(
            notes,
            melody_info.dur,
            melody_info.offset,
            melody_info.vol,
            melody_info.pause_duration,
        )
        melody_midi = stream.Part(melody)
        melody_midi.insert(0, melody_info.instr)
        melody_midi.transpose(12 * melody_info.octave_offset, inPlace=True)
        return melody_midi

    def _compose_entire_song(self, song_length: int, melodies: list[MelodyInfo]) -> stream.Score:
        main_score = stream.Score()
        for melody_info in melodies:
            melody_midi = self._compose_melody(song_length, melody_info)
            main_score.insert(0, melody_midi)
        return main_score

    def run(
        self,
        melodies: list[MelodyInfo],
        song_length: int,
        output_name: str,
        musescore_exe: Optional[Path],
    ) -> None:
        # measure = multiple bars; bar = 8 notes/chords
        # TODO - pauzele afecteaza durata totala a piesei
        main_score = self._compose_entire_song(song_length, melodies)
        output_name = f"Outputs/{output_name}"
        main_score.write("midi", f"{output_name}.mid")
        xml_path = main_score.write("musicxml", f"{output_name}.xml")
        if musescore_exe:
            subprocess.run([str(musescore_exe), str(xml_path), "-o", f"{output_name}.png"])
