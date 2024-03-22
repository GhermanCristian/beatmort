from dataclasses import dataclass
from typing import Optional
from music21 import instrument, articulations
import random
from sentiment_detector.sentiment import Sentiment


# TODO - think about lazy instantiation: don't instantiate the elements in the list, but rather when creating melodyinfo
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
    articulation: Optional[articulations.Articulation]


class SentimentToMelodies:
    INSTRUMENTS = {
        Sentiment.JOY: [
            instrument.UnpitchedPercussion(),
            instrument.Alto(),  # mai strica filmu
            instrument.Marimba(),
            instrument.Piano(),
            instrument.Recorder(),  # mai strica filmu
            instrument.Soprano(),
            instrument.SteelDrum(),
            instrument.Vibraphone(),
            instrument.Xylophone(),
        ],
        Sentiment.FEAR: [
            instrument.AcousticGuitar(),
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
            instrument.Shakuhachi(),  # nu e foarte fear
            instrument.Sitar(),
            instrument.StringInstrument(),
            instrument.Timpani(),
            instrument.TubularBells(),
            instrument.Ukulele(),
        ],
        Sentiment.ANGER: [
            instrument.BrassInstrument(),  # ?
            instrument.ElectricGuitar(),  # nu e foarte angry
            instrument.Piano(),
            instrument.Sampler(),
            instrument.Timpani(),
        ],
        Sentiment.SADNESS: [
            instrument.Celesta(),
            instrument.Contrabass(),
            instrument.Flute(),
            instrument.Glockenspiel(),
            instrument.Harp(),
            instrument.Piano(),
            instrument.StringInstrument(),  # asta e mai mult fear / anticipation
            instrument.Violin(),
            instrument.Violoncello(),  # e un pic cam agresiv
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
            instrument.Shehnai(),
            instrument.SopranoSaxophone(),
            instrument.Trombone(),
        ],
        Sentiment.ANTICIPATION: [
            instrument.Baritone(),
            instrument.BrassInstrument(),
            instrument.Choir(),
            instrument.Dulcimer(),
            instrument.ElectricBass(),
            instrument.Horn(),
            instrument.Mandolin(),
            instrument.MezzoSoprano(),
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
            instrument.Trombone(),  # un pic prea jos
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
            instrument.Flute(),  # un pic prea ascutit uneori
            instrument.Glockenspiel(),
            instrument.Guitar(),
            instrument.Harp(),
            instrument.Kalimba(),
            instrument.Lute(),
            instrument.Marimba(),
            instrument.Vibraphone(),
            instrument.Xylophone(),
        ],
    }

    MEASURE_LENGTHS = {
        Sentiment.JOY: [2, 4],
        Sentiment.FEAR: [1, 2, 4],
        Sentiment.ANGER: [1, 2],
        Sentiment.SADNESS: [1, 2],
        Sentiment.NEUTRAL: [1, 2],
        Sentiment.DISGUST: [1, 2, 4],
        Sentiment.ANTICIPATION: [1, 2],
        Sentiment.SURPRISE: [1],
        Sentiment.TRUST: [1, 2, 4],
    }

    DURATIONS = {
        Sentiment.JOY: [0.25, 0.5],
        Sentiment.FEAR: [1.25, 1.5, 1.75, 2],
        Sentiment.ANGER: [0.25],
        Sentiment.SADNESS: [1.25, 1.5, 2],
        Sentiment.NEUTRAL: [0.75, 1],
        Sentiment.DISGUST: [0.25, 0.33, 0.5, 0.66, 1],
        Sentiment.ANTICIPATION: [1, 1.25, 1.5, 2],
        Sentiment.SURPRISE: [0.25, 0.5],
        Sentiment.TRUST: [0.75, 1, 1.5, 1.75],
    }

    OCTAVE_OFFSETS = {
        Sentiment.JOY: [1, 2],
        Sentiment.FEAR: [-3, -2],
        Sentiment.ANGER: [-2, -1, -2],
        Sentiment.SADNESS: [-2, -1],
        Sentiment.NEUTRAL: [0],
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
    # TODO - make pauses/durations consistent for a single bar?
    # TODO - look into tremolo vs vibrato
    PAUSE_DURATIONS = {
        Sentiment.JOY: [0.25],
        Sentiment.FEAR: [1.25, 1.5, 1.75, 2],
        Sentiment.ANGER: [0, 0.25, 0.5],
        Sentiment.SADNESS: [1.25, 1.5, 1.75],
        Sentiment.NEUTRAL: [0.25, 0.5],
        Sentiment.DISGUST: [0, 0.25, 0.33, 0.5, 0.66, 1],
        Sentiment.ANTICIPATION: [0.25, 0.5],
        Sentiment.SURPRISE: [0.25, 0.5, 1, 2],
        Sentiment.TRUST: [0.25, 0.5],
    }

    ARTICULATIONS = {
        Sentiment.JOY: [articulations.Staccatissimo()],
        Sentiment.FEAR: [articulations.DetachedLegato()],
        Sentiment.ANGER: [articulations.Accent(), articulations.StrongAccent()],
        Sentiment.SADNESS: [articulations.BreathMark(), articulations.Tenuto()],
        Sentiment.NEUTRAL: [None],
        Sentiment.DISGUST: [articulations.Tenuto()],
        Sentiment.ANTICIPATION: [articulations.Staccato()],
        Sentiment.SURPRISE: [articulations.Stress()],
        Sentiment.TRUST: [articulations.Unstress()],
    }

    def _sample_properties(
        self, property_list: list, n_melodies: int, identical: bool = False
    ) -> list:
        n_samples = min(n_melodies, len(property_list))
        if identical:
            return [random.choice(property_list)] * n_melodies
        samples = random.sample(property_list, n_samples) * (n_melodies // n_samples + 1)
        return samples[:n_melodies]

    def run(self, sentiment: Sentiment) -> list[MelodyInfo]:
        n_melodies = 2  # TODO - n_melodies for every sentiment
        instruments: list[instrument.Instrument] = self._sample_properties(
            self.INSTRUMENTS[sentiment], n_melodies
        )
        measure_lengths: list[int] = self._sample_properties(
            self.MEASURE_LENGTHS[sentiment], n_melodies
        )
        durations: list[float] = self._sample_properties(
            self.DURATIONS[sentiment], n_melodies, identical=True
        )
        octave_offsets: list[float] = self._sample_properties(
            self.OCTAVE_OFFSETS[sentiment], n_melodies
        )
        song_keys: list[str] = self._sample_properties(
            self.KEYS[sentiment], n_melodies, identical=True
        )
        pause_durations: list[float] = self._sample_properties(
            self.PAUSE_DURATIONS[sentiment], n_melodies, identical=True
        )
        articulations: list[articulations.Articulation] = self._sample_properties(
            self.ARTICULATIONS[sentiment], n_melodies
        )
        print("instruments", instruments)
        print("measure_lengths", measure_lengths)
        print("durations", durations)
        print("octave_offsets", octave_offsets)
        print("song_keys", song_keys)
        print("pause_durations", pause_durations)
        print("articulations", articulations)
        melodies = [
            MelodyInfo(
                instruments[i],
                measure_lengths[i],
                durations[i],
                (0.0625 + 8 * (durations[0] + pause_durations[0])) * i,
                octave_offsets[i],
                song_keys[i],
                1 - i * 0.25,
                pause_durations[i],
                articulations[i],
            )
            for i in range(n_melodies)
        ]
        print("offsets", [m.offset for m in melodies])
        return melodies
