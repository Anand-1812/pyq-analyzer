import csv
import json
from pathlib import Path

from exam_topic_predictor.pipeline import PipelineResult


def write_reports(result: PipelineResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "analysis_summary.json"
    mapping_path = output_dir / "question_topic_mapping.csv"
    prediction_path = output_dir / "topic_predictions.csv"
    prediction_json_path = output_dir / "topic_predictions.json"
    pattern_csv_path = output_dir / "question_patterns.csv"
    pattern_json_path = output_dir / "question_patterns.json"
    recommendation_json_path = output_dir / "study_recommendations.json"

    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    prediction_json_path.write_text(
        json.dumps(result.to_dict()["likely_topics"], indent=2),
        encoding="utf-8",
    )
    pattern_json_path.write_text(
        json.dumps(result.to_dict()["question_patterns"], indent=2),
        encoding="utf-8",
    )
    recommendation_json_path.write_text(
        json.dumps(result.to_dict()["recommendations"], indent=2),
        encoding="utf-8",
    )
    _write_mapping_csv(result, mapping_path)
    _write_prediction_csv(result, prediction_path)
    _write_pattern_csv(result, pattern_csv_path)

    return {
        "summary_json": summary_path,
        "mapping_csv": mapping_path,
        "prediction_csv": prediction_path,
        "prediction_json": prediction_json_path,
        "pattern_csv": pattern_csv_path,
        "pattern_json": pattern_json_path,
        "recommendation_json": recommendation_json_path,
    }


def _write_mapping_csv(result: PipelineResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["year", "paper_name", "question_id", "question_text", "topic", "similarity"])
        for mapping in result.question_mappings:
            if not mapping.matches:
                writer.writerow([mapping.year, mapping.paper_name, mapping.question.question_id, mapping.question.text, "", ""])
                continue
            for match in mapping.matches:
                writer.writerow(
                    [
                        mapping.year,
                        mapping.paper_name,
                        mapping.question.question_id,
                        mapping.question.text,
                        match.topic,
                        match.similarity,
                    ]
                )


def _write_prediction_csv(result: PipelineResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "topic",
                "forecast_score",
                "confidence",
                "frequency",
                "year_coverage",
                "last_appeared_year",
                "recency_score",
                "pattern_score",
                "pattern",
                "years",
            ]
        )
        for item in result.likely_topics:
            writer.writerow(
                [
                    item.topic,
                    item.score,
                    item.confidence,
                    item.frequency,
                    item.year_coverage,
                    item.last_appeared_year,
                    item.recency_score,
                    item.pattern_score,
                    item.pattern,
                    ",".join(map(str, item.years)),
                ]
            )


def _write_pattern_csv(result: PipelineResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["topic", "pattern_id", "question_count", "similarity_to_topic", "years", "representative_question"]
        )
        for pattern in result.question_patterns:
            writer.writerow(
                [
                    pattern.topic,
                    pattern.pattern_id,
                    pattern.question_count,
                    pattern.similarity_to_topic,
                    ",".join(map(str, pattern.years)),
                    pattern.representative_question,
                ]
            )
