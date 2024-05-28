from pathlib import Path
from constants import Constants
from sentiment import Sentiment
from train.lyric_generation.data_loader import DataLoader as DataLoaderLyrics
from train.lyric_generation.model_creator import ModelCreator as ModelCreatorLyrics
from train.music_creator.data_loader import DataContainer as DataContainerMusic
from train.music_creator.data_loader import DataLoader as DataLoaderMusic
from train.music_creator.model_creator import ModelCreator as ModelCreatorMusic
from train.sentiment_classifier.data_loader import DataLoader as DataLoaderSentiment
from train.sentiment_classifier.model_creator import ModelCreator as ModelCreatorSentiment
from tensorflow.keras.models import load_model


def train_sentiment_classifier() -> None:
    num_classes = len(Sentiment)

    data_loader = DataLoaderSentiment()
    data_container, tokenizer = data_loader.run()
    model_creator = ModelCreatorSentiment(
        num_classes,
        tokenizer.word_index,
        256,
        data_container,
    )
    model = model_creator.get_new_model()
    train_history = model_creator.train_model(model, 50)
    print(train_history.history)
    model = load_model(Constants.SENTIMENT_MODEL_PATH)
    eval_history = model_creator.evaluate_model(model)
    print(eval_history)


def train_music_creator() -> None:
    seed_size = 0.05
    lim = 2

    dataset = Path("..", "data", "Datasets", "D1")
    data_loader = DataLoaderMusic(lim)
    all_notes = data_loader.get_notes_from_txt(str(dataset / "all_notes.txt"))
    filtered_notes = data_loader.filter_notes(all_notes)
    index = data_loader.create_indices(filtered_notes)
    data_container: DataContainerMusic = data_loader.run(filtered_notes, seed_size, index)
    DataLoaderMusic.save_seed_to_disk(data_container)
    print("After data_container")

    model_creator = ModelCreatorMusic(
        validation_size=0.2,
        batch_size=256,
        learning_rate=0.005,
        data_container=data_container,
    )

    model = model_creator.create_model()
    train_history = model_creator.train_model(model, 100)
    print(train_history.history)
    model = load_model(Constants.MUSIC_MODEL_PATH)
    eval_history = model_creator.evaluate_model(model)
    print(eval_history)


def train_lyric_generator() -> None:
    data_loader = DataLoaderLyrics()
    data_container, tokenizer = data_loader.run()
    model_creator = ModelCreatorLyrics(
        Constants.LYRICS_MAX_SEQ_LEN, tokenizer.word_index, 512, data_container
    )
    model = model_creator.get_new_model()
    train_history = model_creator.train_model(model, 1000)
    print(train_history.history)
    model = load_model(Constants.LYRICS_MODEL_PATH)
    eval_history = model_creator.evaluate_model(model)
    print(eval_history)
