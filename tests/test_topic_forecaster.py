from exam_topic_predictor.modeling.topic_forecaster import TopicForecaster
from exam_topic_predictor.schemas import Question, QuestionTopicMapping, TopicMatch


def _mapping(year: int, topic: str, question_id: str) -> QuestionTopicMapping:
    return QuestionTopicMapping(
        year=year,
        paper_name=f"paper_{year}.pdf",
        question=Question(question_id=question_id, text=f"Question on {topic}"),
        matches=(TopicMatch(topic=topic, similarity=0.8),),
    )


def test_forecast_prioritizes_higher_frequency_topics() -> None:
    mappings = [
        _mapping(2021, "Normalization", "Q1"),
        _mapping(2022, "Normalization", "Q1"),
        _mapping(2023, "Normalization", "Q1"),
        _mapping(2022, "Transactions", "Q2"),
    ]

    forecast = TopicForecaster().forecast(mappings, top_n=5)
    assert forecast[0].topic == "Normalization"
    assert forecast[0].frequency == 3
    assert forecast[1].topic == "Transactions"
    assert forecast[0].last_appeared_year == 2023
    assert forecast[0].confidence in {"HIGH", "MEDIUM", "LOW"}


def test_forecast_adds_pattern_description() -> None:
    mappings = [
        _mapping(2021, "Normalization", "Q1"),
        _mapping(2023, "Normalization", "Q1"),
    ]

    forecast = TopicForecaster().forecast(mappings, top_n=5)
    assert "Repeats every 2 years" in forecast[0].pattern


def test_forecast_uses_percentile_confidence_buckets() -> None:
    mappings = [
        _mapping(2021, "Normalization", "Q1"),
        _mapping(2022, "Normalization", "Q2"),
        _mapping(2023, "Normalization", "Q3"),
        _mapping(2021, "Transactions", "Q4"),
        _mapping(2023, "Transactions", "Q5"),
        _mapping(2022, "Indexing", "Q6"),
    ]

    forecast = TopicForecaster().forecast(mappings, top_n=5)
    confidences = {item.topic: item.confidence for item in forecast}
    assert confidences["Normalization"] == "HIGH"
    assert "LOW" in confidences.values()
