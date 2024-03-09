from music21 import chord, instrument, note, stream, duration
import numpy as np
from tensorflow.keras.models import Model


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
        octave_offset: int,
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
                    for j in chord_notes[1:]:
                        inst_note = int(j)
                        if 21 <= int(j) + (octave_offset * 12) <= 108:
                            inst_note += octave_offset * 12
                        note_snip = note.Note(inst_note, duration=dur)
                        note_snip.volume.velocity = 127.0 * vol
                        notes.append(note_snip)
                        chord_snip = chord.Chord(notes)
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

    def _compose_melody(
        self,
        song_length: int,
        instr: instrument.Instrument,
        measure_length: int,
        dur: float,
        offset: float = 0.0,
        octave_offset: int = 0,
        vol: float = 1,
        pause_duration: float = 0,
    ) -> stream.Part:
        notes = self._melody_generator(song_length, dur, measure_length)
        print(notes)
        melody = self._chords_n_notes(notes, dur, offset, octave_offset, vol, pause_duration)
        melody_midi = stream.Part(melody)
        melody_midi.insert(0, instr)
        return melody_midi

    def _compose_entire_song(self, song_length: int) -> stream.Score:
        main_score = stream.Score()
        for i, measure_length, dur, offset, octave_offset, vol, pause_duration in [
            (instrument.Piano(), 1, 0.5, 0, 0, 0.2, 0),
            # (instrument.Sampler(), 4, 1, 0, 2, 0.75, 1),
            (instrument.Xylophone(), 1, 1, 0, 2, 0.5, 1),
            (instrument.Vibraphone(), 1, 0.5, 0, 0, 1, 0),
        ]:
            melody_midi = self._compose_melody(
                song_length, i, measure_length, dur, offset, octave_offset, vol, pause_duration
            )
            main_score.insert(0, melody_midi)
        return main_score

    def run(self, song_length: int, output_name: str) -> None:
        # measure = multiple bars; bar = 8 notes/chords
        # TODO - pauzele afecteaza durata totala a piesei
        # TODO - last note should be very long / there should be a pause so that the song doesn't end before it
        main_score = self._compose_entire_song(song_length)
        main_score.write("midi", f"{output_name}.mid")
        main_score.write("musicxml", f"{output_name}.xml")
