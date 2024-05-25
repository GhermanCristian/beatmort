import pickle

import numpy as np
from constants import Constants
from sentiment import Sentiment
from pathlib import Path
from predict.lyric_generation.lyric_generator import LyricGenerator
from predict.music_creator.music_creator import MusicCreator
from predict.music_creator.sentiment_to_melodies import SentimentToMelodies
from predict.music_creator.song_saver import SongSaver
from predict.sentiment_classifier.sentiment_classifier import SentimentClassifier
from tensorflow.keras.models import load_model


class Predictor:
    def __init__(self) -> None:
        self._lyrics: list[str] = []

    @property
    def lyrics(self) -> list[str]:
        return self._lyrics

    def load_artifacts(self) -> None:
        with open(Constants.SENTIMENT_TOKENIZER_PATH, "rb") as tokenizer_path:
            self._sentiment_tokenizer = pickle.load(tokenizer_path)
        self._sentiment_model = load_model(Constants.SENTIMENT_MODEL_PATH)

        with open(Constants.MUSIC_REVERSE_INDEX_PATH, "rb") as reverse_index_path:
            self._music_reverse_index = pickle.load(reverse_index_path)
        with open(Constants.MUSIC_SEED_PATH, "rb") as music_seed_path:
            self._music_seed = np.load(music_seed_path)
        self._music_model = load_model(Constants.MUSIC_MODEL_PATH)

        with open(Constants.LYRICS_TOKENIZER_PATH, "rb") as tokenizer_path:
            self._lyrics_tokenizer = pickle.load(tokenizer_path)
        self._lyrics_model = load_model(Constants.LYRICS_MODEL_PATH)

    def _classify_sentiment(self, prompt: str) -> Sentiment:
        sentiment_classifier = SentimentClassifier(
            self._sentiment_tokenizer, self._sentiment_model, Constants.SENTIMENT_MAX_SEQ_LEN
        )
        sentiment: Sentiment = sentiment_classifier.run(prompt)
        return sentiment

    def _generate_music(self, sentiment: Sentiment) -> None:
        music_creator = MusicCreator(self._music_model, self._music_seed, self._music_reverse_index)
        melodies = SentimentToMelodies().run(sentiment)
        main_score = music_creator.run(128, melodies)
        SongSaver.save_song_to_disk(
            main_score,
            "..\\Outputs/test",
            Path("C:\Program Files\MuseScore 4\\bin\MuseScore4.exe"),
            Path("..\\FluidSynth\\fluidsynth.exe"),
            Path("..\\FluidSynth\\GeneralUser GS v1.471.sf2"),
        )

    def _generate_lyrics(self, sentiment: Sentiment, n_verses: int) -> list[str]:
        lyric_generator = LyricGenerator(
            Constants.LYRICS_MAX_SEQ_LEN, self._lyrics_tokenizer, self._lyrics_model
        )
        lyrics = lyric_generator.run(n_verses, sentiment)
        return lyrics

    def run(self, prompt: str) -> None:
        sentiment = self._classify_sentiment(prompt)
        # TODO - generate music and lyrics in parallel
        self._generate_music(sentiment)
        self._lyrics = self._generate_lyrics(sentiment, 16)
