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
        ],
        Sentiment.FEAR: [
            instrument.Piano(),
            instrument.Sampler(),
            instrument.ElectricPiano(),
            instrument.Organ(),
            instrument.PipeOrgan(),
            instrument.StringInstrument(),
            instrument.AcousticBass(),
            instrument.Ukulele(),
            instrument.Koto(),
            instrument.Tuba(),
        ],
        Sentiment.ANGER: [
            instrument.Sampler(),
            instrument.Tuba(),
            instrument.Dulcimer(),
        ],
        Sentiment.SADNESS: [ # TODO - find more instruments
            instrument.Piano(),
        ],
    }

    MEASURE_LENGTHS = {
        Sentiment.JOY: [2, 4],
        Sentiment.FEAR: [1, 2, 4],
        Sentiment.ANGER: [1, 1],
        Sentiment.SADNESS: [1, 1],
    }

    DURATIONS = {
        Sentiment.JOY: [0.25, 0.5],
        Sentiment.FEAR: [1.5, 2],
        Sentiment.ANGER: [0.25, 0.25],
        Sentiment.SADNESS: [2, 2],
    }
    # TODO - look into how to incorporate staccato and legato
    # TODO - look into equalizers - especially for violins & stuff

    OCTAVE_OFFSETS = {
        Sentiment.JOY: [1, 1.5, 2],
        Sentiment.FEAR: [-2.5, -2, -1.5],
        Sentiment.ANGER: [-1.5, -1],
        Sentiment.SADNESS: [-2.5, -2],
    }

    PAUSE_DURATIONS = {
        Sentiment.JOY: [0.25, 0.25],
        Sentiment.FEAR: [1.5, 2],
        Sentiment.ANGER: [0.5, 1, 1.5],
        Sentiment.SADNESS: [1.5, 2],
    }

    def run(self, sentiment: Sentiment) -> list[MelodyInfo]:
        n_melodies = 2  # TODO - n_melodies for every sentiment
        instruments: list[instrument.Instrument] = random.sample(self.INSTRUMENTS[sentiment], n_melodies)
        measure_lengths: list[int] = random.sample(self.MEASURE_LENGTHS[sentiment], n_melodies)
        durations: list[float] = random.sample(self.DURATIONS[sentiment], n_melodies)
        octave_offsets: list[float] = random.sample(self.OCTAVE_OFFSETS[sentiment], n_melodies)
        pause_durations: list[float] = random.sample(self.PAUSE_DURATIONS[sentiment], n_melodies)
        print("instruments", instruments)
        print("measure_lengths", measure_lengths)
        print("durations", durations)
        print("octave_offsets", octave_offsets)
        print("pause_durations", pause_durations)
        return [
            MelodyInfo(instruments[0], measure_lengths[0], durations[0], 0, octave_offsets[0], 1, pause_durations[0])
        ] + [
            MelodyInfo(instruments[i], measure_lengths[i], durations[i], 0, octave_offsets[i], 0.5, pause_durations[i]) for i in range(1, n_melodies)
        ]
