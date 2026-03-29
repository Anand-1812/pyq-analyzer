from dataclasses import dataclass


@dataclass(frozen=True)
class MappingConfig:
    min_similarity: float = 0.4
    top_k_topics: int = 3
    min_question_characters: int = 24
    embedding_model_name: str = "all-MiniLM-L6-v2"
    topic_dedupe_similarity: float = 0.85
    max_topic_words: int = 10
    minimum_topic_words: int = 1
    topic_heading_prefixes: tuple[str, ...] = (
        "to understand",
        "introduction to",
        "overview of",
    )


@dataclass(frozen=True)
class ForecastConfig:
    frequency_weight: float = 0.4
    coverage_weight: float = 0.2
    recency_weight: float = 0.3
    pattern_weight: float = 0.1
    high_confidence_threshold: float = 0.75
    medium_confidence_threshold: float = 0.5
    pattern_distance_threshold: float = 0.32
    min_pattern_cluster_size: int = 2
    recency_half_life_years: float = 2.0
    min_question_clusters: int = 3
