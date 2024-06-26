from dataclasses import dataclass
from typing import Optional, Type
from music21 import instrument, articulations
import random
from sentiment import Sentiment


@dataclass
class MelodyInfo:
    instrument: instrument.Instrument
    n_groups: int
    note_durations: list[float]
    pause_durations: list[float]
    offset: float
    octave_offset: int
    key: str
    vol: float
    articulation: Optional[articulations.Articulation]


class SentimentToMelodies:
    INSTRUMENTS: dict[Sentiment, list[Type[instrument.Instrument]]] = {
        Sentiment.JOY: [
            instrument.Marimba,
            instrument.Piano,
            instrument.Recorder,
            instrument.SteelDrum,
            instrument.Vibraphone,
            instrument.Xylophone,
        ],
        Sentiment.FEAR: [
            instrument.Choir,
            instrument.ChurchBells,
            instrument.Dulcimer,
            instrument.ElectricBass,
            instrument.Organ,
            instrument.Piano,
            instrument.PipeOrgan,
            instrument.Sampler,
            instrument.Shakuhachi,
            instrument.Sitar,
            instrument.StringInstrument,
            instrument.Timpani,
            instrument.TubularBells,
            instrument.Ukulele,
        ],
        Sentiment.ANGER: [
            instrument.BrassInstrument,
            instrument.Piano,
            instrument.Sampler,
            instrument.Timpani,
        ],
        Sentiment.SADNESS: [
            instrument.Celesta,
            instrument.Flute,
            instrument.Glockenspiel,
            instrument.Harp,
            instrument.Piano,
            instrument.Violin,
            instrument.Violoncello,
        ],
        Sentiment.NEUTRAL: [
            instrument.AcousticBass,
            instrument.AcousticGuitar,
            instrument.UnpitchedPercussion,
            instrument.Alto,
            instrument.Banjo,
            instrument.Bass,
            instrument.BassClarinet,
            instrument.Bassoon,
            instrument.Celesta,
            instrument.ChurchBells,
            instrument.Clarinet,
            instrument.Clavichord,
            instrument.ElectricBass,
            instrument.ElectricGuitar,
            instrument.ElectricPiano,
            instrument.EnglishHorn,
            instrument.FretlessBass,
            instrument.Glockenspiel,
            instrument.Guitar,
            instrument.Harp,
            instrument.Horn,
            instrument.Kalimba,
            instrument.Lute,
            instrument.Marimba,
            instrument.Ocarina,
            instrument.PanFlute,
            instrument.Piano,
            instrument.Piccolo,
            instrument.Recorder,
            instrument.ReedOrgan,
            instrument.Shamisen,
            instrument.SteelDrum,
            instrument.Tenor,
            instrument.Vibraphone,
            instrument.Vocalist,
            instrument.Whistle,
            instrument.Xylophone,
        ],
        Sentiment.DISGUST: [
            instrument.Accordion,
            instrument.UnpitchedPercussion,
            instrument.AltoSaxophone,
            instrument.Banjo,
            instrument.Bassoon,
            instrument.Contrabassoon,
            instrument.ElectricOrgan,
            instrument.Koto,
            instrument.Shehnai,
            instrument.Trombone,
        ],
        Sentiment.ANTICIPATION: [
            instrument.BrassInstrument,
            instrument.Dulcimer,
            instrument.ElectricBass,
            instrument.Horn,
            instrument.Mandolin,
            instrument.MezzoSoprano,
            instrument.Piano,
            instrument.StringInstrument,
            instrument.Timpani,
            instrument.Trumpet,
            instrument.TubularBells,
            instrument.Ukulele,
        ],
        Sentiment.SURPRISE: [
            instrument.Accordion,
            instrument.UnpitchedPercussion,
            instrument.AltoSaxophone,
            instrument.Banjo,
            instrument.BaritoneSaxophone,
            instrument.BassClarinet,
            instrument.Bassoon,
            instrument.BassTrombone,
            instrument.BrassInstrument,
            instrument.ChurchBells,
            instrument.Clarinet,
            instrument.Clavichord,
            instrument.Dulcimer,
            instrument.FretlessBass,
            instrument.Harmonica,
            instrument.Harpsichord,
            instrument.Koto,
            instrument.Shamisen,
            instrument.SopranoSaxophone,
            instrument.Trumpet,
            instrument.Viola,
            instrument.Violin,
            instrument.Violoncello,
        ],
        Sentiment.TRUST: [
            instrument.AcousticBass,
            instrument.Celesta,
            instrument.Glockenspiel,
            instrument.Guitar,
            instrument.Harp,
            instrument.Kalimba,
            instrument.Lute,
            instrument.Marimba,
            instrument.Vibraphone,
            instrument.Xylophone,
        ],
    }

    N_GROUPS: dict[Sentiment, list[int]] = {
        Sentiment.JOY: [2, 4],
        Sentiment.FEAR: [1, 2, 4],
        Sentiment.ANGER: [2, 4],
        Sentiment.SADNESS: [1, 2],
        Sentiment.NEUTRAL: [1, 2],
        Sentiment.DISGUST: [1, 2, 4],
        Sentiment.ANTICIPATION: [1, 2],
        Sentiment.SURPRISE: [1, 2],
        Sentiment.TRUST: [1, 2, 4],
    }

    DURATIONS: dict[Sentiment, list[float]] = {
        Sentiment.JOY: [0.125, 0.25, 0.5],
        Sentiment.FEAR: [1.25, 1.5, 1.75, 2],
        Sentiment.ANGER: [0.125, 0.25],
        Sentiment.SADNESS: [1.25, 1.5, 2],
        Sentiment.NEUTRAL: [0.75, 1],
        Sentiment.DISGUST: [0.125, 0.25, 0.33, 0.5, 0.66, 1],
        Sentiment.ANTICIPATION: [1, 1.25, 1.5, 2],
        Sentiment.SURPRISE: [0.125, 0.25, 0.5],
        Sentiment.TRUST: [0.75, 1, 1.5, 1.75],
    }

    OCTAVE_OFFSETS: dict[Sentiment, list[int]] = {
        Sentiment.JOY: [1, 2],
        Sentiment.FEAR: [-3, -2],
        Sentiment.ANGER: [-2, -1, -2],
        Sentiment.SADNESS: [-1, 0],
        Sentiment.NEUTRAL: [0],
        Sentiment.DISGUST: [-1, 1],
        Sentiment.ANTICIPATION: [0, 1],
        Sentiment.SURPRISE: [-1, 0, 1],
        Sentiment.TRUST: [1, 2],
    }

    KEYS: dict[Sentiment, list[str]] = {
        Sentiment.JOY: ["C", "G"],
        Sentiment.FEAR: ["f#"],
        Sentiment.ANGER: ["d", "e"],
        Sentiment.SADNESS: ["c", "f", "a"],
        Sentiment.NEUTRAL: ["D", "F"],
        Sentiment.DISGUST: ["c#", "g#", "b"],
        Sentiment.ANTICIPATION: ["E"],
        Sentiment.SURPRISE: ["C#", "F#"],
        Sentiment.TRUST: ["B"],
    }

    PAUSE_DURATIONS: dict[Sentiment, list[float]] = {
        Sentiment.JOY: [0, 0.125, 0.25],
        Sentiment.FEAR: [0.75, 1, 1.25, 1.5],
        Sentiment.ANGER: [0, 0.125, 0.25, 0.5],
        Sentiment.SADNESS: [0.75, 1, 1.25],
        Sentiment.NEUTRAL: [0.25, 0.5],
        Sentiment.DISGUST: [0, 0.125, 0.25, 0.33, 0.5, 0.66, 1],
        Sentiment.ANTICIPATION: [0, 0.125, 0.25, 0.5],
        Sentiment.SURPRISE: [0, 0.125, 0.25, 0.5, 1, 1.25],
        Sentiment.TRUST: [0.125, 0.25, 0.5],
    }

    ARTICULATIONS: dict[Sentiment, list[Optional[Type[articulations.Articulation]]]] = {
        Sentiment.JOY: [articulations.Staccatissimo],
        Sentiment.FEAR: [articulations.DetachedLegato],
        Sentiment.ANGER: [articulations.Accent, articulations.StrongAccent],
        Sentiment.SADNESS: [articulations.BreathMark, articulations.Tenuto],
        Sentiment.NEUTRAL: [None],
        Sentiment.DISGUST: [articulations.Tenuto],
        Sentiment.ANTICIPATION: [articulations.Staccato],
        Sentiment.SURPRISE: [articulations.Stress],
        Sentiment.TRUST: [articulations.Unstress],
    }

    N_MELODIES: dict[Sentiment, list[int]] = {
        Sentiment.JOY: [1, 2],
        Sentiment.FEAR: [2],
        Sentiment.ANGER: [2, 3],
        Sentiment.SADNESS: [2],
        Sentiment.NEUTRAL: [1, 2],
        Sentiment.DISGUST: [2, 3],
        Sentiment.ANTICIPATION: [2],
        Sentiment.SURPRISE: [2],
        Sentiment.TRUST: [1, 2],
    }

    def _sample_properties(
        self,
        property_list: list,
        n_items: int,
        identical: bool = False,
        durations: bool = False,
    ) -> list:
        """Samples n_items from a list of properties.

        Args:
            property_list (list): Properties that are sampled
            n_items (int): Number of items to be retrieved
            identical (bool, optional): If set, all retrieved items are identical. Defaults to False.
            durations (bool, optional): If set, the properties are sorted in decreasing order. Defaults to False.

        Returns:
            list: List of sampled properties
        """
        n_properties = len(property_list)
        if identical:
            return [random.choice(property_list)] * n_items
        samples = random.sample(property_list * (n_items // n_properties + 1), n_items)
        if durations:
            return sorted(samples, reverse=True)
        return samples

    def run(self, sentiment: Sentiment) -> list[MelodyInfo]:
        """Randomly selects the properties that will be applied to each of the song melodies

        Args:
            sentiment (Sentiment): Sentiment which the melodies will follow

        Returns:
            list[MelodyInfo]: Information about every melody
        """
        n_melodies: int = random.choice(self.N_MELODIES[sentiment])
        instrument_types: list[Type[instrument.Instrument]] = self._sample_properties(
            self.INSTRUMENTS[sentiment], n_melodies
        )
        n_groups: list[int] = self._sample_properties(self.N_GROUPS[sentiment], n_melodies)

        durations_single_melody: list[float] = self._sample_properties(
            self.DURATIONS[sentiment], 8, durations=True
        )
        durations_single_melody.reverse()
        durations: list[list[float]] = []
        for _ in range(n_melodies):
            durations.append(durations_single_melody)

        pause_durations_single_melody: list[float] = self._sample_properties(
            self.PAUSE_DURATIONS[sentiment], 8, durations=True
        )
        pause_durations: list[list[float]] = []
        for _ in range(n_melodies):
            pause_durations.append(pause_durations_single_melody)

        octave_offsets: list[float] = self._sample_properties(
            self.OCTAVE_OFFSETS[sentiment], n_melodies
        )
        song_keys: list[str] = self._sample_properties(
            self.KEYS[sentiment], n_melodies, identical=True
        )
        articulation_types: list[Optional[Type[articulations.Articulation]]] = (
            self._sample_properties(self.ARTICULATIONS[sentiment], n_melodies)
        )
        melodies = [
            MelodyInfo(
                instrument_types[i](),
                n_groups[i],
                durations[i],
                pause_durations[i],
                (0.0625 + (sum(durations[0]) + sum(pause_durations[0]))) * i,
                octave_offsets[i],
                song_keys[i],
                1 - i * 0.25,
                articulation_types[i]() if articulation_types[i] else None,
            )
            for i in range(n_melodies)
        ]
        return melodies
