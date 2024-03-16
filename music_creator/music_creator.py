import copy
from pathlib import Path
from typing import Optional
from music21 import chord, note, stream, duration, pitch, interval, key
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

    def _chords_n_notes(self, measures: list[str], melody_info: MelodyInfo):
        melody = []
        offset: float = melody_info.offset
        dur = duration.Duration(melody_info.dur)
        for b in measures:
            sounds = b.split("/")
            for s in sounds:
                if "." in s or s.isdigit():
                    chord_notes = s.split(".")
                    notes = []
                    for j in chord_notes:
                        inst_note = int(j)
                        note_snip = note.Note(inst_note)
                        note_snip.volume.velocity = 127.0 * melody_info.vol
                        note_snip.articulations.append(melody_info.articulation)
                        notes.append(note_snip)
                    chord_snip = chord.Chord(notes, duration=dur)
                    chord_snip.offset = offset
                    melody.append(chord_snip)
                else:
                    note_snip = note.Note(s)
                    note_snip.volume.velocity = 127.0 * melody_info.vol
                    note_snip.articulations.append(melody_info.articulation)
                    note_snip.offset = offset
                    melody.append(note_snip)
                if melody_info.pause_duration:
                    p = note.Rest(duration=duration.Duration(melody_info.pause_duration))
                    p.offset = offset
                    melody.append(p)
                offset += melody_info.dur + melody_info.pause_duration
        final_note_index = -1
        while isinstance(melody[final_note_index], note.Rest):
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
        # TODO - ensure a measure has at least 2 different notes / chords
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
        assert (
            song_length >= dur * measure_length * 8
        ), "Song is too short for the given note durations and measure lengths"
        measure = self._measure_generator(measure_length)
        n_measures_in_song = int(song_length // dur // measure_length // 8)
        music = measure * n_measures_in_song
        # TODO - think of cases when song_length % (dur * measure_length) != 0
        return music

    def _compose_melody(self, song_length: int, melody_info: MelodyInfo) -> stream.Part:
        notes = self._melody_generator(song_length, melody_info.dur, melody_info.measure_length)
        # TODO - generate melody until the correct key mode is found ?
        print(notes)
        melody = self._chords_n_notes(notes, melody_info)
        melody_midi = stream.Part(melody)

        k: key.Key = melody_midi.analyze("key")
        print(f"Initial key = {k}; target = {melody_info.key}")
        melody_midi.insert(0, melody_info.instr)

        # attempts to transpose melody to another key only if their modes are equal
        # TODO - scris la partea teoretica despre toate astea
        # TODO - scris la partea teoretica despre articulations
        major_target_key = any(c.isupper() for c in melody_info.key)
        if k.mode == "major" and major_target_key or k.mode == "minor" and not major_target_key:
            i = interval.Interval(k.tonic, pitch.Pitch(melody_info.key))
            melody_midi.transpose(i, inPlace=True)
            print(f"Transposed to {melody_midi.analyze('key')}")
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
        fluidsynth_exe: Optional[Path],
        soundfont: Optional[Path],
    ) -> None:
        # measure = multiple bars; bar = 8 notes/chords
        # TODO - pauzele afecteaza durata totala a piesei
        main_score = self._compose_entire_song(song_length, melodies)
        output_name = f"Outputs/{output_name}"
        main_score.write("midi", f"{output_name}.mid")
        # TODO - separate module for these outputs, maybe for the midi write part as well
        if musescore_exe:
            try:
                xml_path = main_score.write("musicxml", f"{output_name}.xml")
                subprocess.run([str(musescore_exe), str(xml_path), "-o", f"{output_name}.png"])
            except:
                print("Could not write score")
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
                    ]
                )
            except:
                print("Could not convert to audio")
