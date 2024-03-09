from music_creator.data_preprocesser import DataContainer, DataPreprocessor
from music_creator.model_creator import ModelCreator
from music_creator.music_creator import MusicCreator


def create_music() -> None:
    seed_size = 0.05
    batch_size = 256
    learning_rate = 0.005
    validation_size = 0.2
    feature_length = 8
    lim = 2

    data_preprocessor = DataPreprocessor(feature_length, lim)
    all_notes = data_preprocessor.get_notes_from_txt("all_notes.txt")
    filtered_notes = data_preprocessor.filter_notes(all_notes)
    index, reverse_index = data_preprocessor.create_indices(filtered_notes)
    vocab_size = len(index)
    data_container: DataContainer = data_preprocessor.run(filtered_notes, seed_size, index)
    print("After data_container")

    n_notes = len(filtered_notes)
    model_creator = ModelCreator(
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
    music_creator.run(32, "testsong1")


def run_app() -> None:
    create_music()


if __name__ == "__main__":
    run_app()
