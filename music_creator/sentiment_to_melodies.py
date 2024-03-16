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
            instrument.UnpitchedPercussion(),
            instrument.Alto(),
            instrument.Clarinet(),
            instrument.Marimba(),
            instrument.Ocarina(),
            instrument.Piano(),
            instrument.Recorder(),
            instrument.Soprano(),
            instrument.SteelDrum(),
            instrument.Tenor(),
            instrument.Vibraphone(),
            instrument.Xylophone(),
        ],
        Sentiment.FEAR: [
            instrument.AcousticGuitar(),
            instrument.Bagpipes(),
            instrument.BrassInstrument(),
            instrument.Choir(),
            instrument.ChurchBells(),
            instrument.Dulcimer(),
            instrument.ElectricBass(),
            instrument.MezzoSoprano(),  # ?
            instrument.Organ(),
            instrument.Piano(),
            instrument.PipeOrgan(),
            instrument.Sampler(),
            instrument.Shakuhachi(),
            instrument.Sitar(),
            instrument.StringInstrument(),
            instrument.Timpani(),
            instrument.Tuba(),
            instrument.TubularBells(),
            instrument.Ukulele(),
        ],
        Sentiment.ANGER: [
            instrument.Bagpipes(),  # asta uneori e mai mult surprise, disgust
            instrument.Organ(),  # asta mai mult cauzeaza fear
            instrument.Piano(),
            instrument.PipeOrgan(),  # si asta tot e mai mult fear
            instrument.Sampler(),
            instrument.Timpani(),
        ],
        Sentiment.SADNESS: [
            instrument.Celesta(),
            instrument.Contrabass(),
            instrument.EnglishHorn(),
            instrument.Flute(),
            instrument.Glockenspiel(),
            instrument.Harp(),
            instrument.Piano(),
            instrument.StringInstrument(),
            instrument.Violin(),
            instrument.Violoncello(),
        ],
        Sentiment.NEUTRAL: [
            instrument.AcousticBass(),
            instrument.AcousticGuitar(),
            instrument.UnpitchedPercussion(),
            instrument.Alto(),
            instrument.Banjo(),
            instrument.Baritone(),
            instrument.Bass(),
            instrument.BassClarinet(),
            instrument.Bassoon(),
            instrument.BassTrombone(),
            instrument.Celesta(),
            instrument.ChurchBells(),
            instrument.Clarinet(),
            instrument.Clavichord(),
            instrument.Contrabass(),
            instrument.ElectricBass(),
            instrument.ElectricGuitar(),
            instrument.ElectricPiano(),
            instrument.EnglishHorn(),
            instrument.Flute(),
            instrument.FretlessBass(),
            instrument.Glockenspiel(),
            instrument.Guitar(),
            instrument.Harp(),
            instrument.Horn(),
            instrument.Kalimba(),
            instrument.Lute(),
            instrument.Marimba(),
            instrument.Ocarina(),
            instrument.PanFlute(),
            instrument.Piano(),
            instrument.Piccolo(),
            instrument.Recorder(),
            instrument.ReedOrgan(),
            instrument.Shamisen(),
            instrument.Soprano(),
            instrument.SteelDrum(),
            instrument.Tenor(),
            instrument.Vibraphone(),
            instrument.Vocalist(),
            instrument.Whistle(),
            instrument.Xylophone(),
        ],
        Sentiment.DISGUST: [
            instrument.Accordion(),
            instrument.UnpitchedPercussion(),
            instrument.AltoSaxophone(),
            instrument.Bagpipes(),
            instrument.Banjo(),
            instrument.BaritoneSaxophone(),
            instrument.BassClarinet(),
            instrument.Bassoon(),
            instrument.Contrabassoon(),
            instrument.ElectricOrgan(),
            instrument.Harmonica(),
            instrument.Koto(),
            instrument.Oboe(),
            instrument.Saxophone(),
            instrument.Shehnai(),
            instrument.SopranoSaxophone(),
            instrument.TenorSaxophone(),
            instrument.Trombone(),
        ],
        Sentiment.ANTICIPATION: [
            instrument.Bagpipes(),
            instrument.Baritone(),
            instrument.BrassInstrument(),
            instrument.Choir(),
            instrument.Dulcimer(),
            instrument.ElectricBass(),
            instrument.Horn(),
            instrument.Mandolin(),
            instrument.MezzoSoprano(),
            instrument.Organ(),
            instrument.Piano(),
            instrument.Sitar(),
            instrument.StringInstrument(),
            instrument.Tenor(),
            instrument.Timpani(),
            instrument.TubularBells(),
            instrument.Ukulele(),
            instrument.Xylophone(),
        ],
        Sentiment.SURPRISE: [
            instrument.Accordion(),
            instrument.UnpitchedPercussion(),
            instrument.AltoSaxophone(),
            instrument.Bagpipes(),
            instrument.Banjo(),
            instrument.BaritoneSaxophone(),
            instrument.BassClarinet(),
            instrument.Bassoon(),
            instrument.BassTrombone(),
            instrument.BrassInstrument(),
            instrument.Choir(),
            instrument.ChurchBells(),
            instrument.Clarinet(),
            instrument.Clavichord(),
            instrument.Dulcimer(),
            instrument.FretlessBass(),
            instrument.Harmonica(),
            instrument.Harpsichord(),
            instrument.Koto(),
            instrument.Oboe(),
            instrument.Shamisen(),
            instrument.Soprano(),
            instrument.SopranoSaxophone(),
            instrument.Trombone(),
            instrument.Trumpet(),
            instrument.Viola(),
            instrument.Violin(),
            instrument.Violoncello(),
        ],
        Sentiment.TRUST: [
            instrument.AcousticBass(),
            instrument.Alto(),
            instrument.Baritone(),
            instrument.Celesta(),
            instrument.Flute(),
            instrument.Glockenspiel(),
            instrument.Guitar(),
            instrument.Harp(),
            instrument.Kalimba(),
            instrument.Lute(),
            instrument.Marimba(),
            instrument.Vibraphone(),
            instrument.Vocalist(),
            instrument.Xylophone(),
        ],
    }

    MEASURE_LENGTHS = {
        Sentiment.JOY: [2, 4],
        Sentiment.FEAR: [1, 2, 4],
        Sentiment.ANGER: [1, 1, 1],
        Sentiment.SADNESS: [1, 1],
        Sentiment.NEUTRAL: [1, 1],
        Sentiment.DISGUST: [1, 2, 4],
        Sentiment.ANTICIPATION: [1, 2],
        Sentiment.SURPRISE: [1, 1],
        Sentiment.TRUST: [1, 2, 4],
    }

    DURATIONS = {
        Sentiment.JOY: [0.25, 0.5],
        Sentiment.FEAR: [1.5, 2],
        Sentiment.ANGER: [0.25, 0.25, 0.25],
        Sentiment.SADNESS: [2, 2],
        Sentiment.NEUTRAL: [1, 1],
        Sentiment.DISGUST: [0.25, 0.33, 0.5, 0.66, 1],
        Sentiment.ANTICIPATION: [1.25, 1.5, 2],
        Sentiment.SURPRISE: [0.25, 0.5],
        Sentiment.TRUST: [1, 1.5, 2],
    }
    # TODO - look into how to incorporate staccato and legato

    OCTAVE_OFFSETS = {
        Sentiment.JOY: [1, 2],
        Sentiment.FEAR: [-3, -2],
        Sentiment.ANGER: [-2, -1, -2],
        Sentiment.SADNESS: [-2, -1],
        Sentiment.NEUTRAL: [0, 0],
        Sentiment.DISGUST: [-2, 2],
        Sentiment.ANTICIPATION: [1, 2],
        Sentiment.SURPRISE: [-1, 1, 2],
        Sentiment.TRUST: [1, 2],
    }

    KEYS = {
        Sentiment.JOY: ["C", "G"],
        Sentiment.FEAR: ["f", "b-"],
        Sentiment.ANGER: ["d", "e", "e"],
        Sentiment.SADNESS: ["a", "e-"],
        Sentiment.NEUTRAL: ["C", "a"],
        Sentiment.DISGUST: ["f", "b-"],
        Sentiment.ANTICIPATION: ["E", "D", "f#"],
        Sentiment.SURPRISE: ["C", "G", "d"],
        Sentiment.TRUST: ["D", "A"],
    }

    # TODO - disjointed pauses
    # TODO - durata + durata pauzelor sa fie egale, eventual doar un mic offset?
    PAUSE_DURATIONS = {
        Sentiment.JOY: [0.25, 0.25],
        Sentiment.FEAR: [1.5, 2],
        Sentiment.ANGER: [0, 0.25, 0.5],
        Sentiment.SADNESS: [1.5, 2],
        Sentiment.NEUTRAL: [0.5, 0.5],
        Sentiment.DISGUST: [0, 0.25, 0.33, 0.5, 0.66, 1],
        Sentiment.ANTICIPATION: [0.25, 0.5],
        Sentiment.SURPRISE: [0.25, 0.5, 1, 2],
        Sentiment.TRUST: [0.25, 0.5],
    }

    def run(self, sentiment: Sentiment) -> list[MelodyInfo]:
        n_melodies = 2  # TODO - n_melodies for every sentiment
        instruments: list[instrument.Instrument] = random.sample(
            self.INSTRUMENTS[sentiment], n_melodies
        )
        measure_lengths: list[int] = random.sample(self.MEASURE_LENGTHS[sentiment], n_melodies)
        durations: list[float] = random.sample(self.DURATIONS[sentiment], n_melodies)
        octave_offsets: list[float] = random.sample(self.OCTAVE_OFFSETS[sentiment], n_melodies)
        song_keys: list[str] = [
            random.choice(self.KEYS[sentiment])
        ] * n_melodies  # keep same key throughout song
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
