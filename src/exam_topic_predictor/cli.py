from argparse import ArgumentParser
from pathlib import Path
import sys

from exam_topic_predictor.config import ForecastConfig, MappingConfig
from exam_topic_predictor.pipeline import run_pipeline
from exam_topic_predictor.reporting import write_reports


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="exam-topic-predictor",
        description=(
            "Extract questions from previous-year paper PDFs, map questions to syllabus topics, "
            "and predict likely upcoming topics."
        ),
    )
    parser.add_argument("--syllabus", type=Path, required=True, help="Path to syllabus PDF.")
    parser.add_argument("--papers-dir", type=Path, help="Directory containing previous-year paper PDFs.")
    parser.add_argument("--papers", type=Path, nargs="+", help="One or more previous-year paper PDF files.")
    parser.add_argument("--topics-file", type=Path, help="Optional text file with one syllabus topic per line.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs"), help="Directory for generated reports.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of likely topics to output.")
    parser.add_argument("--min-similarity", type=float, default=0.4, help="Minimum question-topic similarity threshold.")
    parser.add_argument("--top-k-topics", type=int, default=3, help="Maximum syllabus topics to keep per question.")
    parser.add_argument("--min-question-chars", type=int, default=24, help="Minimum characters to accept a parsed question.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.papers and not args.papers_dir:
        parser.error("Provide either --papers-dir or --papers.")

    syllabus_path = args.syllabus.resolve()
    paper_paths = [path for path in resolve_paper_paths(args.papers_dir, args.papers) if path != syllabus_path]
    if not paper_paths:
        parser.error("No PDF files were found for previous-year papers.")

    mapping_config = MappingConfig(
        min_similarity=args.min_similarity,
        top_k_topics=args.top_k_topics,
        min_question_characters=args.min_question_chars,
    )
    forecast_config = ForecastConfig()
    manual_topics = read_manual_topics(args.topics_file) if args.topics_file else None

    result = run_pipeline(
        syllabus_pdf=args.syllabus,
        paper_pdfs=paper_paths,
        mapping_config=mapping_config,
        forecast_config=forecast_config,
        top_n=args.top_n,
        manual_topics=manual_topics,
    )
    reports = write_reports(result, output_dir=args.output_dir)

    print(f"Syllabus topics extracted: {len(result.syllabus_topics)}")
    print(f"Questions mapped: {result.question_count}")
    print(f"Papers processed: {result.paper_count}")
    print("Top likely topics:")
    for idx, item in enumerate(result.likely_topics, start=1):
        print(
            f"{idx}. {item.topic} | score={item.score:.4f} | confidence={item.confidence} "
            f"| freq={item.frequency} | years={','.join(map(str, item.years))} | pattern={item.pattern}"
        )

    print("\nGenerated files:")
    for key, path in reports.items():
        print(f"- {key}: {path}")

    return 0


def resolve_paper_paths(papers_dir: Path | None, papers: list[Path] | None) -> list[Path]:
    paper_paths: list[Path] = []
    if papers_dir:
        paper_paths.extend(sorted(path for path in papers_dir.glob("*.pdf") if path.is_file()))
    if papers:
        paper_paths.extend(path for path in papers if path.is_file() and path.suffix.casefold() == ".pdf")
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paper_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)
    return unique_paths


def read_manual_topics(path: Path) -> list[str]:
    """Read a manual syllabus topic list from a text file."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    sys.exit(main())
