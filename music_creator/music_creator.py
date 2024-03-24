import copy
from music21 import chord, note, stream, duration, pitch, interval, key
import numpy as np
from tensorflow.keras.models import Model

from music_creator.sentiment_to_melodies import MelodyInfo


class MusicCreator:
    Sound = note.Note | chord.Chord

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

    def _create_final_note(self, melody: list[Sound], offset: float) -> Sound:
        final_note_index = -1
        while isinstance(melody[final_note_index], note.Rest):
            final_note_index -= 1
        final_note = copy.deepcopy(melody[final_note_index])
        final_note.duration = duration.Duration(4)
        final_note.offset = offset
        final_note.volume.velocity = 127.0
        return final_note

    def _create_notes_and_chords(self, measure: list[str], melody_info: MelodyInfo) -> list[Sound]:
        melody = []
        offset: float = melody_info.offset
        durations: list[duration.Duration] = [
            duration.Duration(dur) for dur in melody_info.note_durations
        ]
        pause_durations: list[duration.Duration] = [
            duration.Duration(pause_duration) for pause_duration in melody_info.pause_durations
        ]

        for bar in measure:
            sounds = bar.split("/")
            for s_index, s in enumerate(sounds):
                if "." in s or s.isdigit():
                    chord_notes = s.split(".")
                    notes = []
                    for j in chord_notes:
                        inst_note = int(j)
                        note_snip = note.Note(inst_note)
                        note_snip.volume.velocity = 127.0 * melody_info.vol
                        if melody_info.articulation:
                            note_snip.articulations.append(melody_info.articulation)
                        notes.append(note_snip)
                    chord_snip = chord.Chord(notes, duration=durations[s_index])
                    chord_snip.offset = offset
                    melody.append(chord_snip)
                else:
                    note_snip = note.Note(s)
                    note_snip.volume.velocity = 127.0 * melody_info.vol
                    if melody_info.articulation:
                        note_snip.articulations.append(melody_info.articulation)
                    note_snip.offset = offset
                    melody.append(note_snip)
                if melody_info.pause_durations[s_index]:
                    p = note.Rest(duration=pause_durations[s_index])
                    p.offset = offset
                    melody.append(p)
                offset += melody_info.note_durations[s_index] + melody_info.pause_durations[s_index]
        melody.append(self._create_final_note(melody, offset))
        return melody

    def _measure_generator(self, n_bars: int) -> list[str]:
        seed = self._x_seed[np.random.randint(0, len(self._x_seed) - 1)][:]
        measure: list[str] = []
        while len(measure) < n_bars:
            seed = seed.reshape(1, self._feature_length, 1)
            prediction = self._model.predict(seed, verbose=0)[0]
            prediction = np.log(prediction) / 0.25
            exp_preds = np.exp(prediction)
            prediction = exp_preds / np.sum(exp_preds)
            pos = np.argmax(prediction)
            pos_n = pos / float(self._vocab_size)
            bar = self._reverse_index[pos]
            if len(set(bar.split("/"))) > 1:
                measure.append(bar)
            seed = np.insert(seed[0], len(seed[0]), pos_n)
            seed = seed[1:]

        return measure

    def _melody_generator(self, song_length: int, melody_info: MelodyInfo) -> list[str]:
        bar_duration: float = sum(melody_info.note_durations) + sum(melody_info.pause_durations)
        assert (
            song_length >= bar_duration * melody_info.n_bars
        ), "Song is too short for the given note durations and measure lengths"
        measure = self._measure_generator(melody_info.n_bars)
        n_measures_in_song = int(song_length // bar_duration // melody_info.n_bars)
        music = measure * n_measures_in_song
        return music

    def _compose_melody_correct_mode(
        self, song_length: int, melody_info: MelodyInfo
    ) -> stream.Part:
        major_target_key = any(c.isupper() for c in melody_info.key)
        attempts = 5
        while attempts:
            notes = self._melody_generator(song_length, melody_info)
            melody = self._create_notes_and_chords(notes, melody_info)
            melody_midi = stream.Part(melody)
            k: key.Key = melody_midi.analyze("key")
            if k.mode == "major" and major_target_key or k.mode == "minor" and not major_target_key:
                return melody_midi
            attempts -= 1
        return melody_midi

    def _transpose_melody(self, melody_midi: stream.Part, melody_info: MelodyInfo) -> stream.Part:
        # TODO - scris la partea teoretica despre toate astea
        k: key.Key = melody_midi.analyze("key")
        print(f"Initial key = {k}; target = {melody_info.key}")
        i = interval.Interval(k.tonic, pitch.Pitch(melody_info.key))
        melody_midi.transpose(i, inPlace=True)
        print(f"Transposed to {melody_midi.analyze('key')}")
        melody_midi.transpose(12 * melody_info.octave_offset, inPlace=True)

        return melody_midi

    def _compose_melody(self, song_length: int, melody_info: MelodyInfo) -> stream.Part:
        melody_midi = self._compose_melody_correct_mode(song_length, melody_info)
        melody_midi.insert(0, melody_info.instrument)

        # TODO - scris la partea teoretica despre articulations
        # TODO - mentionat in partea teoretica despre incercarea de a avea durate diferite in fiecare bar
        # TODO - scris la teoretic despre faptul ca e greu de scos corelatii bune cu sentimentele, pt ca seedul e neutru
        melody_midi = self._transpose_melody(melody_midi, melody_info)
        return melody_midi

    def run(self, song_length: int, melodies: list[MelodyInfo]) -> stream.Score:
        main_score = stream.Score()
        for melody_info in melodies:
            melody_midi = self._compose_melody(song_length, melody_info)
            main_score.insert(0, melody_midi)
        return main_score
