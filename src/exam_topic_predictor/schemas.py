from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str


@dataclass(frozen=True)
class TopicMatch:
    topic: str
    similarity: float


@dataclass(frozen=True)
class QuestionTopicMapping:
    year: int
    question: Question
    matches: tuple[TopicMatch, ...]


@dataclass(frozen=True)
class TopicForecast:
    topic: str
    score: float
    frequency: int
    year_coverage: int
    years: tuple[int, ...]
