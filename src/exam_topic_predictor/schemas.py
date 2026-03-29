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
    paper_name: str
    question: Question
    matches: tuple[TopicMatch, ...]


@dataclass(frozen=True)
class TopicForecast:
    topic: str
    score: float
    frequency: int
    year_coverage: int
    last_appeared_year: int
    recency_score: float
    pattern_score: float
    pattern: str
    confidence: str
    years: tuple[int, ...]


@dataclass(frozen=True)
class QuestionPattern:
    topic: str
    pattern_id: str
    representative_question: str
    question_count: int
    similarity_to_topic: float
    years: tuple[int, ...]


@dataclass(frozen=True)
class SyllabusTopic:
    topic: str
    source_line: str
