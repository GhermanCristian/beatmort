from enum import Enum


class Sentiment(str, Enum):
    ANTICIPATION = "anticipation"
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    JOY = "joy"
    NEUTRAL = "neutral"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    TRUST = "trust"
