from dataclasses import dataclass


@dataclass(frozen=True)
class MappingConfig:
    min_similarity: float = 0.18
    top_k_topics: int = 3
    min_question_characters: int = 24


@dataclass(frozen=True)
class ForecastConfig:
    frequency_weight: float = 0.55
    coverage_weight: float = 0.30
    cycle_weight: float = 0.15
