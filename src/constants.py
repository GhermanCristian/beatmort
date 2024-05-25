class Constants:
    SENTIMENT_MODEL_PATH = "../data/Models/model_sentiment.keras"
    SENTIMENT_TOKENIZER_PATH = "../data/Tokenizers/tokenizer_sentiment.pkl"
    SENTIMENT_MAX_SEQ_LEN = 500  # TODO - reevaluate this, maybe retrain the model => has a smaller size

    MUSIC_MODEL_PATH = "../data/Models/model_music.keras"
    MUSIC_FEATURE_LENGTH = 8
    MUSIC_REVERSE_INDEX_PATH = "../data/Tokenizers/reverse_index_music.pkl"
    MUSIC_SEED_PATH = "../data/Seeds/music_seed.npy"

    LYRICS_MODEL_PATH = "../data/Models/model_lyrics.keras"
    LYRICS_TOKENIZER_PATH = "../data/Tokenizers/tokenizer_lyrics.pkl"
    LYRICS_MAX_SEQ_LEN = 34
