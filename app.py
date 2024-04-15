from pathlib import Path
from music_creator.data_loader import DataContainer as DataContainerMusic
from music_creator.data_loader import DataLoader as DataLoaderMusic
from music_creator.model_creator import ModelCreator as ModelCreatorMusic
from music_creator.music_creator import MusicCreator
from music_creator.sentiment_to_melodies import SentimentToMelodies
from sentiment_detector.data_loader import DataLoader as DataLoaderSentiment
from sentiment_detector.model_creator import ModelCreator as ModelCreatorSentiment
from sentiment_detector.sentiment import Sentiment
from music_creator.song_saver import SongSaver
from sentiment_detector.sentiment_detector import SentimentDetector


def detect_sentiment(prompt: str) -> Sentiment:
    n_dims_embedding = 300
    max_seq_len = 500
    batch_size = 256

    num_classes = len(Sentiment)

    is_new_model = False  # TODO - this should be a class attr / function param
    if is_new_model:
        data_loader = DataLoaderSentiment(max_seq_len)
        data_container, tokenizer = data_loader.run()
        model_creator = ModelCreatorSentiment(
            n_dims_embedding,
            max_seq_len,
            num_classes,
            tokenizer.word_index,
            batch_size,
            data_container,
        )
        model = model_creator.get_new_model()
        train_history = model_creator.train_model(model, 50)
        print(train_history.history)
        model = model_creator.get_model(new_model=False)
        eval_history = model_creator.evaluate_model(model)
        print(eval_history)
    else:
        tokenizer = DataLoaderSentiment.load_tokenizer()
        model = ModelCreatorSentiment.load_model(batch_size)

    sentiment_detector = SentimentDetector(tokenizer, model, max_seq_len)
    sentiment: Sentiment = sentiment_detector.run(prompt)
    return sentiment


def create_music(sentiment: Sentiment) -> None:
    seed_size = 0.05
    batch_size = 256
    learning_rate = 0.005
    validation_size = 0.2
    feature_length = 8
    lim = 2

    dataset = Path("Datasets", "D1")
    data_loader = DataLoaderMusic(feature_length, lim)
    all_notes = data_loader.get_notes_from_txt(str(dataset / "all_notes.txt"))
    filtered_notes = data_loader.filter_notes(all_notes)
    index, reverse_index = data_loader.create_indices(filtered_notes)
    vocab_size = len(index)
    data_container: DataContainerMusic = data_loader.run(filtered_notes, seed_size, index)
    print("After data_container")

    n_notes = len(filtered_notes)
    model_creator = ModelCreatorMusic(
        n_notes,
        feature_length,
        validation_size,
        seed_size,
        batch_size,
        lim,
        learning_rate,
        data_container,
    )
    new_model = False
    model = model_creator.get_model(new_model)
    if new_model:
        train_history = model_creator.train_model(model, 100)
        print(train_history.history)
        model = model_creator.get_model(new_model=False)
        eval_history = model_creator.evaluate_model(model)
        print(eval_history)

    print("Got model")
    music_creator = MusicCreator(
        model, data_container.x_seed, feature_length, vocab_size, reverse_index
    )
    melodies = SentimentToMelodies().run(sentiment)
    main_score = music_creator.run(128, melodies)
    SongSaver.save_song_to_disk(
        main_score,
        "Outputs/test",
        Path("C:\Program Files\MuseScore 4\\bin\MuseScore4.exe"),
        Path("FluidSynth\\fluidsynth.exe"),
        Path("FluidSynth\\GeneralUser GS v1.471.sf2"),
    )


def run_app() -> None:
    prompt = "i am really anxious about this"
    sentiment = detect_sentiment(prompt)
    create_music(sentiment)


if __name__ == "__main__":
    run_app()
