import copy
from music21 import chord, note, stream, duration, pitch, interval, key
import numpy as np
from tensorflow.keras.models import Model

from constants import Constants
from predict.music_creator.sentiment_to_melodies import MelodyInfo


class MusicCreator:
    Sound = note.Note | chord.Chord

    def __init__(
        self,
        model: Model,
        x_seed: np.array,
        reverse_index: dict[int, str],
    ) -> None:
        self._model = model
        self._x_seed = x_seed
        self._vocab_size = len(reverse_index)
        self._reverse_index = reverse_index

    def _create_final_note(self, melody: list[Sound], offset: float) -> Sound:
        """Creates the final note of the melody, which is just a longer version
        of the last non-rest in it

        Args:
            melody (list[Sound]): Input melody
            offset (float): Position at which the note will be added in the melody

        Returns:
            Sound: The new note
        """
        final_note_index = -1
        while isinstance(melody[final_note_index], note.Rest):
            final_note_index -= 1
        final_note = copy.deepcopy(melody[final_note_index])
        final_note.duration = duration.Duration(4)
        final_note.offset = offset
        final_note.volume.velocity = 127.0
        return final_note

    def _create_notes_and_chords(self, measure: list[str], melody_info: MelodyInfo) -> list[Sound]:
        """Given a list of notes and information associated with the melody, creates the actual sound
        objects that are used in the melody

        Args:
            measure (list[str]): Notes and chords that are converted to sound
            melody_info (MelodyInfo): Information about the melody (volume, note duration)

        Returns:
            list[Sound]: Sound objects that will be used to create the melody
        """
        melody = []
        offset: float = melody_info.offset
        durations: list[duration.Duration] = [
            duration.Duration(dur) for dur in melody_info.note_durations
        ]
        pause_durations: list[duration.Duration] = [
            duration.Duration(pause_duration) for pause_duration in melody_info.pause_durations
        ]

        for group in measure:
            sounds = group.split("/")
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

    def _measure_generator(self, n_groups: int) -> list[str]:
        """Given the song length, expressed in number of 8-note groups, predicts
        the notes and chords that the melody will contain (string form), using the
        existing seeds.

        Args:
            n_groups (int): Number of 8-note groups in the melody

        Returns:
            list[str]: Notes and chords that will be used in the melody
        """
        seed = self._x_seed[np.random.randint(0, len(self._x_seed) - 1)][:]
        measure: list[str] = []
        while len(measure) < n_groups:
            seed = seed.reshape(1, Constants.MUSIC_FEATURE_LENGTH, 1)
            prediction = self._model.predict(seed, verbose=0)[0]
            pos = np.argmax(prediction)
            pos_n = pos / float(self._vocab_size)
            group = self._reverse_index[pos]
            if len(set(group.split("/"))) > 1:
                measure.append(group)
            seed = np.insert(seed[0], len(seed[0]), pos_n)
            seed = seed[1:]

        return measure

    def _melody_generator(self, song_length: int, melody_info: MelodyInfo) -> list[str]:
        """Wrapper over method that predicts the song notes and chords; responsible for
        determining the exact number of groups that will be used, based on the song length

        Args:
            song_length (int): Song length expressed in number of notes/chords
            melody_info (MelodyInfo): Information about the melody (ex. note durations)

        Returns:
            list[str]: Notes and chords that will be used in the melody
        """
        group_duration: float = sum(melody_info.note_durations) + sum(melody_info.pause_durations)
        assert (
            song_length >= group_duration * melody_info.n_groups
        ), "Song is too short for the given note durations and measure lengths"
        measure = self._measure_generator(melody_info.n_groups)
        n_measures_in_song = int(song_length // group_duration // melody_info.n_groups)
        music = measure * n_measures_in_song
        return music

    def _compose_melody_correct_mode(
        self, song_length: int, melody_info: MelodyInfo
    ) -> stream.Part:
        """Repeatedly attempts to create a melody that is in the same mode (major / minor)
        as the key provided in the melody info, so that it can easily be converted to it

        Args:
            song_length (int): Song length expressed in number of notes/chords
            melody_info (MelodyInfo): Information about the melody (ex. note durations)

        Returns:
            stream.Part: Melody object
        """
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
        """Transposes a melody to the key provided in the melody info. Both the source and the
        target keys have the same mode (major / minor)

        Args:
            melody_midi (stream.Part): Melody object
            melody_info (MelodyInfo): Information about the melody

        Returns:
            stream.Part: Melody object, converted to the correct key, if necessary
        """
        k: key.Key = melody_midi.analyze("key")
        i = interval.Interval(k.tonic, pitch.Pitch(melody_info.key))
        melody_midi.transpose(i, inPlace=True)
        melody_midi.transpose(12 * melody_info.octave_offset, inPlace=True)

        return melody_midi

    def _compose_melody(self, song_length: int, melody_info: MelodyInfo) -> stream.Part:
        """Composes a melody with the correct key and instrument

        Args:
            song_length (int): Song length expressed in number of notes/chords
            melody_info (MelodyInfo): Information about the melody

        Returns:
            stream.Part: Melody object
        """
        melody_midi = self._compose_melody_correct_mode(song_length, melody_info)
        melody_midi.insert(0, melody_info.instrument)
        melody_midi = self._transpose_melody(melody_midi, melody_info)
        return melody_midi

    def run(self, song_length: int, melodies: list[MelodyInfo]) -> stream.Score:
        """Composes an entire song (can have multiple melodies)

        Args:
            song_length (int): Song length expressed in number of notes/chords
            melodies (list[MelodyInfo]): Information about every melody

        Returns:
            stream.Score: Song object
        """
        main_score = stream.Score()
        for melody_info in melodies:
            melody_midi = self._compose_melody(song_length, melody_info)
            main_score.insert(0, melody_midi)
        return main_score
