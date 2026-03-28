from exam_topic_predictor.modeling.topic_forecaster import TopicForecaster
from exam_topic_predictor.schemas import Question, QuestionTopicMapping, TopicMatch


def _mapping(year: int, topic: str, question_id: str) -> QuestionTopicMapping:
    return QuestionTopicMapping(
        year=year,
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
