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
    key: str
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
        Sentiment.SADNESS: [  # TODO - find more instruments
            instrument.Piano(),
        ],
        Sentiment.NEUTRAL: [
            instrument.Piano(),
            instrument.Celesta(),
            instrument.ElectricPiano(),
            instrument.Guitar(),
            instrument.AcousticGuitar(),
            instrument.ElectricGuitar(),
            instrument.Banjo(),
            instrument.Lute(),
            instrument.Shamisen(),
            instrument.Recorder(),
            instrument.PanFlute(),
            instrument.Whistle(),
            instrument.Ocarina(),
            instrument.Clarinet(),
            instrument.BassClarinet(),
            instrument.BrassInstrument(),
            instrument.UnpitchedPercussion(),
            instrument.Woodblock(),
            instrument.Vibraphone(),
            instrument.Marimba(),
            instrument.Glockenspiel(),
            instrument.Xylophone(),
            instrument.Timpani(),
            instrument.Kalimba(),
            instrument.Vocalist(),
            instrument.Soprano(),
            instrument.MezzoSoprano(),
            instrument.Alto(),
            instrument.Tenor(),
            instrument.Baritone(),
            instrument.Bass(),
            instrument.Choir(),
        ],
    }

    MEASURE_LENGTHS = {
        Sentiment.JOY: [2, 4],
        Sentiment.FEAR: [1, 2, 4],
        Sentiment.ANGER: [1, 1, 1],
        Sentiment.SADNESS: [1, 1],
        Sentiment.NEUTRAL: [1, 1],
    }

    DURATIONS = {
        Sentiment.JOY: [0.25, 0.5],
        Sentiment.FEAR: [1.5, 2],
        Sentiment.ANGER: [0.25, 0.25, 0.25],
        Sentiment.SADNESS: [2, 2],
        Sentiment.NEUTRAL: [1, 1],
    }
    # TODO - look into how to incorporate staccato and legato
    # TODO - look into equalizers - especially for violins & stuff

    OCTAVE_OFFSETS = {
        Sentiment.JOY: [1, 2],
        Sentiment.FEAR: [-3, -2],
        Sentiment.ANGER: [-2, -1, -2],
        Sentiment.SADNESS: [-2, -1],
        Sentiment.NEUTRAL: [0, 0],
    }

    KEYS = {
        Sentiment.JOY: ["C", "G"],
        Sentiment.FEAR: ["f", "b-"],
        Sentiment.ANGER: ["d", "e", "e"],
        Sentiment.SADNESS: ["a", "e-"],
        Sentiment.NEUTRAL: ["C", "a"],
    }

    PAUSE_DURATIONS = {
        Sentiment.JOY: [0.25, 0.25],
        Sentiment.FEAR: [1.5, 2],
        Sentiment.ANGER: [0.5, 1, 1.5],
        Sentiment.SADNESS: [1.5, 2],
        Sentiment.NEUTRAL: [0.5, 0.5],
    }

    def run(self, sentiment: Sentiment) -> list[MelodyInfo]:
        n_melodies = 2  # TODO - n_melodies for every sentiment
        instruments: list[instrument.Instrument] = random.sample(
            self.INSTRUMENTS[sentiment], n_melodies
        )
        measure_lengths: list[int] = random.sample(self.MEASURE_LENGTHS[sentiment], n_melodies)
        durations: list[float] = random.sample(self.DURATIONS[sentiment], n_melodies)
        octave_offsets: list[float] = random.sample(self.OCTAVE_OFFSETS[sentiment], n_melodies)
        song_keys: list[str] = [random.choice(self.KEYS[sentiment])] * n_melodies  # keep same key throughout song
        pause_durations: list[float] = random.sample(self.PAUSE_DURATIONS[sentiment], n_melodies)
        # TODO - create a getter method for sample, so that I don't have to duplicate data
        print("instruments", instruments)
        print("measure_lengths", measure_lengths)
        print("durations", durations)
        print("octave_offsets", octave_offsets)
        print("song_keys", song_keys)
        print("pause_durations", pause_durations)
        return [
            MelodyInfo(
                instruments[0],
                measure_lengths[0],
                durations[0],
                0,
                octave_offsets[0],
                song_keys[0],
                1,
                pause_durations[0],
            )
        ] + [
            MelodyInfo(
                instruments[i],
                measure_lengths[i],
                durations[i],
                0,
                octave_offsets[i],
                song_keys[i],
                0.5,
                pause_durations[i],
            )
            for i in range(1, n_melodies)
        ]
