from pathlib import Path

from exam_topic_predictor.pipeline import extract_year_from_filename


def test_extract_year_from_filename_supports_multiple_papers_per_year() -> None:
    assert extract_year_from_filename(Path("dbms_2023_set1.pdf")) == 2023
    assert extract_year_from_filename(Path("dbms_2023_midterm.pdf")) == 2023
