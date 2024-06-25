from datetime import datetime
import pickle
import threading

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
        self._sentiment: Sentiment

    @property
    def lyrics(self) -> list[str]:
        return self._lyrics

    @property
    def sentiment(self) -> str:
        return self._sentiment.value

    def load_artifacts(self) -> None:
        """Loads the artifacts necessary for the classification and predictions"""
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
        """Classifies the sentiment expressed in the prompt

        Args:
            prompt (str): Prompt that is used for classifying the sentiment

        Returns:
            Sentiment: Sentiment expressed in the prompt
        """
        sentiment_classifier = SentimentClassifier(
            self._sentiment_tokenizer, self._sentiment_model, Constants.SENTIMENT_MAX_SEQ_LEN
        )
        sentiment: Sentiment = sentiment_classifier.run(prompt)
        return sentiment

    def _generate_music(self) -> None:
        """Generates music and saves it to disk"""
        music_creator = MusicCreator(self._music_model, self._music_seed, self._music_reverse_index)
        melodies = SentimentToMelodies().run(self._sentiment)
        main_score = music_creator.run(128, melodies)
        SongSaver.save_song_to_disk(
            main_score,
            f"{Constants.OUTPUT_SAVE_DIR}/{self._output_name}",
            Path("..\\FluidSynth\\fluidsynth.exe"),
            Path("..\\FluidSynth\\GeneralUser GS v1.471.sf2"),
        )

    def _generate_lyrics(self, n_verses: int) -> list[str]:
        """Generates verses and saves output to disk

        Args:
            n_verses (int): Number of verses

        Returns:
            list[str]: List of verses
        """
        lyric_generator = LyricGenerator(
            Constants.LYRICS_MAX_SEQ_LEN, self._lyrics_tokenizer, self._lyrics_model
        )
        lyrics = lyric_generator.run(n_verses, self._sentiment)
        Path(f"{Constants.OUTPUT_SAVE_DIR}/{self._output_name}.txt").write_text("\n".join(lyrics))
        return lyrics

    def run(self, prompt: str, n_verses: int) -> None:
        """Given a prompt, detects the sentiment expressed in it and creates
        verses and melodies accordingly

        Args:
            prompt (str): Prompt that is used for classifying the sentiment
            n_verses (int): Number of output verses
        """
        self._sentiment = self._classify_sentiment(prompt)
        self._output_name = str(datetime.now()).replace(" ", "___").replace(":", "_")[:-7]
        music_thread = threading.Thread(target=self._generate_music)
        music_thread.start()
        self._lyrics = self._generate_lyrics(n_verses)
