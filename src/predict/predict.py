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


def classify_sentiment(prompt: str) -> Sentiment:
    with open(Constants.SENTIMENT_TOKENIZER_PATH, "rb") as tokenizer_path:
        tokenizer = pickle.load(tokenizer_path)
    model = load_model(Constants.SENTIMENT_MODEL_PATH)

    sentiment_classifier = SentimentClassifier(tokenizer, model, Constants.SENTIMENT_MAX_SEQ_LEN)
    sentiment: Sentiment = sentiment_classifier.run(prompt)
    return sentiment


def generate_music(sentiment: Sentiment) -> None:
    with open(Constants.MUSIC_REVERSE_INDEX_PATH, "rb") as reverse_index_path:
        reverse_index = pickle.load(reverse_index_path)
    with open(Constants.MUSIC_SEED_PATH, "rb") as music_seed_path:
        seed = np.load(music_seed_path)
    model = load_model(Constants.MUSIC_MODEL_PATH)

    music_creator = MusicCreator(model, seed, reverse_index)
    melodies = SentimentToMelodies().run(sentiment)
    main_score = music_creator.run(128, melodies)
    SongSaver.save_song_to_disk(
        main_score,
        "..\\Outputs/test",
        Path("C:\Program Files\MuseScore 4\\bin\MuseScore4.exe"),
        Path("..\\FluidSynth\\fluidsynth.exe"),
        Path("..\\FluidSynth\\GeneralUser GS v1.471.sf2"),
    )


def generate_lyrics(sentiment: Sentiment, n_verses: int) -> list[str]:
    with open(Constants.LYRICS_TOKENIZER_PATH, "rb") as tokenizer_path:
        tokenizer = pickle.load(tokenizer_path)
    model = load_model(Constants.LYRICS_MODEL_PATH)

    lyric_generator = LyricGenerator(Constants.LYRICS_MAX_SEQ_LEN, tokenizer, model)
    lyrics = lyric_generator.run(n_verses, sentiment)
    return lyrics


def run(prompt: str) -> None:
    sentiment = classify_sentiment(prompt)
    generate_music(sentiment)
    lyrics = generate_lyrics(sentiment, 16)
    for l in lyrics:
        print(l)
