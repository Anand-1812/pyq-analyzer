from dataclasses import dataclass
from pathlib import Path
import re

from exam_topic_predictor.config import ForecastConfig, MappingConfig
from exam_topic_predictor.io import extract_text_from_pdf
from exam_topic_predictor.mapping import TopicMapper
from exam_topic_predictor.modeling import TopicForecaster
from exam_topic_predictor.parsing import extract_questions, extract_syllabus_topics
from exam_topic_predictor.schemas import QuestionTopicMapping, TopicForecast

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


@dataclass(frozen=True)
class PipelineResult:
    syllabus_topics: list[str]
    question_mappings: list[QuestionTopicMapping]
    repeated_topics: list[TopicForecast]
    likely_topics: list[TopicForecast]
    paper_count: int
    question_count: int

    def to_dict(self) -> dict:
        return {
            "paper_count": self.paper_count,
            "question_count": self.question_count,
            "syllabus_topic_count": len(self.syllabus_topics),
            "syllabus_topics": self.syllabus_topics,
            "repeated_topics": [
                {
                    "topic": item.topic,
                    "score": item.score,
                    "frequency": item.frequency,
                    "year_coverage": item.year_coverage,
                    "years": list(item.years),
                }
                for item in self.repeated_topics
            ],
            "likely_topics": [
                {
                    "topic": item.topic,
                    "score": item.score,
                    "frequency": item.frequency,
                    "year_coverage": item.year_coverage,
                    "years": list(item.years),
                }
                for item in self.likely_topics
            ],
            "question_topic_mapping": [
                {
                    "year": mapping.year,
                    "question_id": mapping.question.question_id,
                    "question_text": mapping.question.text,
                    "matches": [{"topic": match.topic, "similarity": match.similarity} for match in mapping.matches],
                }
                for mapping in self.question_mappings
            ],
        }


def run_pipeline(
    syllabus_pdf: Path,
    paper_pdfs: list[Path],
    mapping_config: MappingConfig | None = None,
    forecast_config: ForecastConfig | None = None,
    top_n: int = 10,
) -> PipelineResult:
    if not paper_pdfs:
        raise ValueError("At least one previous-year paper PDF is required.")

    mapping_cfg = mapping_config or MappingConfig()
    forecast_cfg = forecast_config or ForecastConfig()

    syllabus_text = extract_text_from_pdf(syllabus_pdf)
    syllabus_topics = extract_syllabus_topics(syllabus_text)
    if not syllabus_topics:
        raise ValueError("Could not extract syllabus topics from the syllabus PDF.")

    mapper = TopicMapper(
        topics=syllabus_topics,
        min_similarity=mapping_cfg.min_similarity,
        top_k_topics=mapping_cfg.top_k_topics,
    )

    all_mappings: list[QuestionTopicMapping] = []
    for paper_path in sorted(paper_pdfs):
        year = extract_year_from_filename(paper_path)
        paper_text = extract_text_from_pdf(paper_path)
        questions = extract_questions(paper_text, min_question_characters=mapping_cfg.min_question_characters)
        if not questions:
            continue
        all_mappings.extend(mapper.map_questions(questions, year))

    forecaster = TopicForecaster(config=forecast_cfg)
    full_forecast = forecaster.forecast(all_mappings, top_n=max(len(syllabus_topics), top_n, 1))
    likely_topics = full_forecast[: max(top_n, 1)]

    return PipelineResult(
        syllabus_topics=syllabus_topics,
        question_mappings=all_mappings,
        repeated_topics=sorted(full_forecast, key=lambda item: item.frequency, reverse=True),
        likely_topics=likely_topics,
        paper_count=len(paper_pdfs),
        question_count=len(all_mappings),
    )


def extract_year_from_filename(file_path: Path) -> int:
    match = YEAR_PATTERN.search(file_path.name)
    if not match:
        raise ValueError(
            f"Could not infer year from '{file_path.name}'. Include the year in filename like 'dbms_2023.pdf'."
        )
    return int(match.group(0))
