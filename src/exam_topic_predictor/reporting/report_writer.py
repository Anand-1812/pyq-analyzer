import csv
import json
from pathlib import Path

from exam_topic_predictor.pipeline import PipelineResult


def write_reports(result: PipelineResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "analysis_summary.json"
    mapping_path = output_dir / "question_topic_mapping.csv"
    prediction_path = output_dir / "topic_predictions.csv"

    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    _write_mapping_csv(result, mapping_path)
    _write_prediction_csv(result, prediction_path)

    return {
        "summary_json": summary_path,
        "mapping_csv": mapping_path,
        "prediction_csv": prediction_path,
    }


def _write_mapping_csv(result: PipelineResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["year", "question_id", "question_text", "topic", "similarity"])
        for mapping in result.question_mappings:
            if not mapping.matches:
                writer.writerow([mapping.year, mapping.question.question_id, mapping.question.text, "", ""])
                continue
            for match in mapping.matches:
                writer.writerow(
                    [
                        mapping.year,
                        mapping.question.question_id,
                        mapping.question.text,
                        match.topic,
                        match.similarity,
                    ]
                )


def _write_prediction_csv(result: PipelineResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["topic", "forecast_score", "frequency", "year_coverage", "years"])
        for item in result.likely_topics:
            writer.writerow([item.topic, item.score, item.frequency, item.year_coverage, ",".join(map(str, item.years))])
