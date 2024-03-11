from dataclasses import dataclass
from music21 import instrument
import random
from sentiment_detector.sentiment import Sentiment


@dataclass
class MelodyInfo:
    instr: instrument.Instrument
    measure_length: int
    dur: float
    offset: float
    octave_offset: int
    vol: float
    pause_duration: float


class SentimentToMelodies:
    INSTRUMENTS = {
        Sentiment.JOY: [
            instrument.Piano(),
            instrument.ElectricPiano(),
            instrument.StringInstrument(),
            instrument.Harp(),
            instrument.Mandolin(),
            instrument.Flute(),
            instrument.Recorder(),
            instrument.Whistle(),
            instrument.Ocarina(),
            instrument.Clarinet(),
            instrument.BrassInstrument(),
            instrument.Vibraphone(),
            instrument.Marimba(),
            instrument.Glockenspiel(),
            instrument.ChurchBells(),
            instrument.TubularBells(),
            instrument.SteelDrum(),
            instrument.Kalimba(),
            instrument.Vocalist(),
            instrument.Soprano(),
            instrument.MezzoSoprano(),
            instrument.Alto(),
            instrument.Tenor(),
            instrument.Baritone(),
            instrument.Bass(),
        ]
    }

    MEASURE_LENGTHS = {
        Sentiment.JOY: [2, 4]
    }

    DURATIONS = {
        Sentiment.JOY: [0.25, 0.5]
    }

    OCTAVE_OFFSETS = {
        Sentiment.JOY: [1, 1.5, 2]
    }

    PAUSE_DURATIONS = {
        Sentiment.JOY: [0.25]
    }

    def run(self, sentiment: Sentiment) -> list[MelodyInfo]:
        n_melodies = 2
        instruments: list[instrument.Instrument] = random.sample(self.INSTRUMENTS[sentiment], n_melodies)
        measure_lengths: list[int] = random.sample(self.MEASURE_LENGTHS[sentiment], n_melodies)
        durations: list[float] = random.sample(self.DURATIONS[sentiment], n_melodies)
        octave_offsets: list[float] = random.sample(self.OCTAVE_OFFSETS[sentiment], n_melodies)
        pause_durations: list[float] = random.sample(self.PAUSE_DURATIONS[sentiment], counts=[n_melodies], k=n_melodies)
        print(instruments)
        print(measure_lengths)
        print(durations)
        print(octave_offsets)
        print(pause_durations)
        return [
            MelodyInfo(instruments[0], measure_lengths[0], durations[0], 0, octave_offsets[0], 1, pause_durations[0])
        ] + [
            MelodyInfo(instruments[i], measure_lengths[i], durations[i], durations[i] / 2.0, octave_offsets[i], 0.5, pause_durations[i]) for i in range(1, n_melodies)
        ]
