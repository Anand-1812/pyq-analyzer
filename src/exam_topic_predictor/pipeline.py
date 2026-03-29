from dataclasses import dataclass
from pathlib import Path
import re

from exam_topic_predictor.config import ForecastConfig, MappingConfig
from exam_topic_predictor.io import extract_text_from_pdf
from exam_topic_predictor.mapping import map_questions_to_topics
from exam_topic_predictor.modeling import TopicForecaster, predict_questions
from exam_topic_predictor.parsing import extract_questions, extract_syllabus_topics
from exam_topic_predictor.schemas import QuestionPattern, QuestionTopicMapping, TopicForecast

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


@dataclass(frozen=True)
class PipelineResult:
    syllabus_topics: list[str]
    question_mappings: list[QuestionTopicMapping]
    repeated_topics: list[TopicForecast]
    likely_topics: list[TopicForecast]
    question_patterns: list[QuestionPattern]
    recommendations: dict[str, list[TopicForecast]]
    papers_by_year: dict[int, list[str]]
    paper_count: int
    question_count: int

    def to_dict(self) -> dict:
        return {
            "paper_count": self.paper_count,
            "question_count": self.question_count,
            "syllabus_topic_count": len(self.syllabus_topics),
            "syllabus_topics": self.syllabus_topics,
            "papers_by_year": {str(year): sorted(papers) for year, papers in sorted(self.papers_by_year.items())},
            "repeated_topics": [
                {
                    "topic": item.topic,
                    "score": item.score,
                    "frequency": item.frequency,
                    "year_coverage": item.year_coverage,
                    "last_appeared_year": item.last_appeared_year,
                    "recency_score": item.recency_score,
                    "pattern_score": item.pattern_score,
                    "pattern": item.pattern,
                    "confidence": item.confidence,
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
                    "last_appeared_year": item.last_appeared_year,
                    "recency_score": item.recency_score,
                    "pattern_score": item.pattern_score,
                    "pattern": item.pattern,
                    "confidence": item.confidence,
                    "years": list(item.years),
                }
                for item in self.likely_topics
            ],
            "question_patterns": [
                {
                    "topic": pattern.topic,
                    "pattern_id": pattern.pattern_id,
                    "representative_question": pattern.representative_question,
                    "question_count": pattern.question_count,
                    "similarity_to_topic": pattern.similarity_to_topic,
                    "years": list(pattern.years),
                }
                for pattern in self.question_patterns
            ],
            "recommendations": {
                label: [
                    {
                        "topic": item.topic,
                        "score": item.score,
                        "confidence": item.confidence,
                        "pattern": item.pattern,
                    }
                    for item in items
                ]
                for label, items in self.recommendations.items()
            },
            "question_topic_mapping": [
                {
                    "year": mapping.year,
                    "paper_name": mapping.paper_name,
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
    manual_topics: list[str] | None = None,
) -> PipelineResult:
    if not paper_pdfs:
        raise ValueError("At least one previous-year paper PDF is required.")

    mapping_cfg = mapping_config or MappingConfig()
    forecast_cfg = forecast_config or ForecastConfig()

    syllabus_text = extract_text_from_pdf(syllabus_pdf)
    syllabus_topics = (
        extract_syllabus_topics("\n".join(manual_topics), config=mapping_cfg)
        if manual_topics
        else extract_syllabus_topics(syllabus_text, config=mapping_cfg)
    )
    if not syllabus_topics:
        raise ValueError("Could not extract syllabus topics from the syllabus PDF.")

    all_mappings: list[QuestionTopicMapping] = []
    papers_by_year: dict[int, list[str]] = {}
    for paper_path in sorted(paper_pdfs):
        year = extract_year_from_filename(paper_path)
        papers_by_year.setdefault(year, []).append(paper_path.name)
        paper_text = extract_text_from_pdf(paper_path)
        questions = extract_questions(paper_text, min_question_characters=mapping_cfg.min_question_characters)
        if not questions:
            continue
        all_mappings.extend(
            map_questions_to_topics(
                questions,
                syllabus_topics=syllabus_topics,
                year=year,
                paper_name=paper_path.name,
                min_similarity=mapping_cfg.min_similarity,
                top_k_topics=mapping_cfg.top_k_topics,
                embedding_model_name=mapping_cfg.embedding_model_name,
            )
        )

    all_mappings = _enforce_syllabus_only_mappings(all_mappings, allowed_topics=set(syllabus_topics))

    forecaster = TopicForecaster(config=forecast_cfg)
    full_forecast = forecaster.forecast(all_mappings, top_n=max(len(syllabus_topics), top_n, 1))
    likely_topics = full_forecast[: max(top_n, 1)]
    question_patterns = predict_questions(
        all_mappings,
        syllabus_topics=syllabus_topics,
        mapping_config=mapping_cfg,
        forecast_config=forecast_cfg,
        top_k_per_topic=3,
    )
    recommendations = build_recommendations(likely_topics)

    return PipelineResult(
        syllabus_topics=syllabus_topics,
        question_mappings=all_mappings,
        repeated_topics=sorted(full_forecast, key=lambda item: item.frequency, reverse=True),
        likely_topics=likely_topics,
        question_patterns=question_patterns,
        recommendations=recommendations,
        papers_by_year=papers_by_year,
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


def build_recommendations(topics: list[TopicForecast]) -> dict[str, list[TopicForecast]]:
    """Group predicted topics into student-friendly recommendation buckets."""
    return {
        "high_priority": [item for item in topics if item.confidence == "HIGH"],
        "medium_priority": [item for item in topics if item.confidence == "MEDIUM"],
        "low_priority": [item for item in topics if item.confidence == "LOW"],
    }


def _enforce_syllabus_only_mappings(
    mappings: list[QuestionTopicMapping],
    allowed_topics: set[str],
) -> list[QuestionTopicMapping]:
    """Drop any topic matches that are not present in the cleaned syllabus."""
    constrained_mappings: list[QuestionTopicMapping] = []
    for mapping in mappings:
        matches = tuple(match for match in mapping.matches if match.topic in allowed_topics)
        constrained_mappings.append(
            QuestionTopicMapping(
                year=mapping.year,
                paper_name=mapping.paper_name,
                question=mapping.question,
                matches=matches,
            )
        )
    return constrained_mappings
