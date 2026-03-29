from collections import Counter, defaultdict
from collections.abc import Sequence
import math

import numpy as np

from exam_topic_predictor.config import ForecastConfig
from exam_topic_predictor.schemas import QuestionTopicMapping, TopicForecast


class TopicForecaster:
    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()

    def forecast(self, mappings: Sequence[QuestionTopicMapping], top_n: int = 10) -> list[TopicForecast]:
        """Forecast likely important topics using frequency, coverage, recency, and gap signals."""
        if not mappings:
            return []

        topic_counts: Counter[str] = Counter()
        topic_years: dict[str, set[int]] = defaultdict(set)
        all_years = sorted({mapping.year for mapping in mappings})

        for mapping in mappings:
            if not mapping.matches:
                continue
            primary_topic = mapping.matches[0].topic
            topic_counts[primary_topic] += 1
            topic_years[primary_topic].add(mapping.year)

        if not topic_counts:
            return []

        paper_count = len({(mapping.year, mapping.paper_name) for mapping in mappings})
        predictions = compute_scores(
            topic_counts=topic_counts,
            topic_years=topic_years,
            all_years=all_years,
            paper_count=paper_count,
            config=self.config,
        )
        return predictions[:top_n]


def _recency_signal(last_appeared_year: int, current_year: int, half_life_years: float) -> float:
    age = max(0, current_year - last_appeared_year)
    return math.exp(-age / max(0.5, half_life_years))


def _pattern_signal(years: list[int], current_year: int) -> tuple[float, str]:
    """Return a normalized recurrence score and an explicit pattern description."""
    if len(years) < 2:
        return 0.0, "No stable repeating pattern"

    gaps = [next_year - year for year, next_year in zip(years, years[1:], strict=False)]
    average_gap = sum(gaps) / len(gaps)
    predicted_next_year = years[-1] + round(average_gap)
    if len(gaps) == 1:
        stability_score = 1.0
    else:
        mean_gap = average_gap
        variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
        stability_score = 1.0 / (1.0 + math.sqrt(variance))

    if 1.5 <= average_gap < 2.5:
        pattern_score = stability_score
        description = f"Repeats every 2 years (next likely {predicted_next_year})"
    elif 2.5 <= average_gap < 3.5:
        pattern_score = stability_score
        description = f"Repeats every 3 years (next likely {predicted_next_year})"
    else:
        pattern_score = 0.5 * stability_score
        description = f"Irregular gap pattern (~{average_gap:.1f} years)"

    del current_year
    return min(1.0, max(0.0, pattern_score)), description


def _confidence_label(score: float, high_threshold: float, medium_threshold: float) -> str:
    if score >= high_threshold:
        return "HIGH"
    if score >= medium_threshold:
        return "MEDIUM"
    return "LOW"


def _assign_percentile_confidence(predictions: list[TopicForecast]) -> list[TopicForecast]:
    """Assign HIGH/MEDIUM/LOW labels from score percentiles."""
    if not predictions:
        return predictions

    scores = np.asarray([item.score for item in predictions], dtype=float)
    high_cutoff = float(np.quantile(scores, 0.75))
    medium_cutoff = float(np.quantile(scores, 0.25))

    labeled: list[TopicForecast] = []
    for item in predictions:
        if item.score >= high_cutoff:
            confidence = "HIGH"
        elif item.score <= medium_cutoff:
            confidence = "LOW"
        else:
            confidence = "MEDIUM"
        labeled.append(
            TopicForecast(
                topic=item.topic,
                score=item.score,
                frequency=item.frequency,
                year_coverage=item.year_coverage,
                last_appeared_year=item.last_appeared_year,
                recency_score=item.recency_score,
                pattern_score=item.pattern_score,
                pattern=item.pattern,
                confidence=confidence,
                years=item.years,
            )
        )
    labeled.sort(key=lambda item: (item.score, item.frequency), reverse=True)
    return labeled


def compute_scores(
    topic_counts: Counter[str],
    topic_years: dict[str, set[int]],
    all_years: list[int],
    paper_count: int,
    config: ForecastConfig,
) -> list[TopicForecast]:
    """Compute normalized topic scores and assign confidence labels."""
    max_frequency = max(topic_counts.values())
    total_years = max(1, len(all_years))
    current_year = max(all_years)
    predictions: list[TopicForecast] = []

    for topic, frequency in topic_counts.items():
        years = sorted(topic_years[topic])
        frequency_signal = frequency / max_frequency
        coverage_signal = len(years) / total_years
        recency_signal = _recency_signal(
            years[-1],
            current_year,
            half_life_years=config.recency_half_life_years,
        )
        pattern_score, pattern_description = _pattern_signal(years, current_year)

        score = (
            config.frequency_weight * frequency_signal
            + config.coverage_weight * coverage_signal
            + config.recency_weight * recency_signal
            + config.pattern_weight * pattern_score
        )
        predictions.append(
            TopicForecast(
                topic=topic,
                score=round(min(1.0, max(0.0, score)), 4),
                frequency=frequency,
                year_coverage=len(years),
                last_appeared_year=years[-1],
                recency_score=round(recency_signal, 4),
                pattern_score=round(pattern_score, 4),
                pattern=pattern_description,
                confidence="MEDIUM",
                years=tuple(years),
            )
        )

    predictions.sort(key=lambda item: (item.score, item.frequency), reverse=True)
    if paper_count < 5:
        return _assign_small_dataset_confidence(predictions)
    return _assign_percentile_confidence(predictions)


def _assign_small_dataset_confidence(predictions: list[TopicForecast]) -> list[TopicForecast]:
    labeled: list[TopicForecast] = []
    for item in predictions:
        confidence = _confidence_label(item.score, high_threshold=0.7, medium_threshold=0.4)
        labeled.append(
            TopicForecast(
                topic=item.topic,
                score=item.score,
                frequency=item.frequency,
                year_coverage=item.year_coverage,
                last_appeared_year=item.last_appeared_year,
                recency_score=item.recency_score,
                pattern_score=item.pattern_score,
                pattern=item.pattern,
                confidence=confidence,
                years=item.years,
            )
        )
    labeled.sort(key=lambda item: (item.score, item.frequency), reverse=True)
    return labeled
