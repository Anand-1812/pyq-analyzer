from collections import Counter, defaultdict
from collections.abc import Sequence

from exam_topic_predictor.config import ForecastConfig
from exam_topic_predictor.schemas import QuestionTopicMapping, TopicForecast


class TopicForecaster:
    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()

    def forecast(self, mappings: Sequence[QuestionTopicMapping], top_n: int = 10) -> list[TopicForecast]:
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

        max_frequency = max(topic_counts.values())
        paper_count = max(1, len(all_years))
        current_year = max(all_years)
        predictions: list[TopicForecast] = []

        for topic, frequency in topic_counts.items():
            years = sorted(topic_years[topic])
            frequency_signal = frequency / max_frequency
            coverage_signal = len(years) / paper_count
            cycle_signal = _cycle_signal(years, current_year, paper_count)

            score = (
                self.config.frequency_weight * frequency_signal
                + self.config.coverage_weight * coverage_signal
                + self.config.cycle_weight * cycle_signal
            )
            predictions.append(
                TopicForecast(
                    topic=topic,
                    score=round(score, 4),
                    frequency=frequency,
                    year_coverage=len(years),
                    years=tuple(years),
                )
            )

        predictions.sort(key=lambda item: (item.score, item.frequency), reverse=True)
        return predictions[:top_n]


def _cycle_signal(years: list[int], current_year: int, paper_count: int) -> float:
    if not years:
        return 0.0

    if len(years) == 1:
        gap = max(1, current_year - years[0])
        return min(1.0, gap / max(1, paper_count))

    gaps = [next_year - year for year, next_year in zip(years, years[1:], strict=False)]
    average_gap = sum(gaps) / len(gaps)
    current_gap = max(0.0, current_year - years[-1])

    # Higher score when the topic is close to its historical repetition cycle.
    return 1.0 / (1.0 + abs(current_gap - average_gap))
