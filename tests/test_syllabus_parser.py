from exam_topic_predictor.config import MappingConfig
from exam_topic_predictor.parsing.syllabus_parser import extract_syllabus_topics


def test_extract_syllabus_topics_from_unit_lines() -> None:
    text = """
    Syllabus:
    Unit 1: Relational Model, Relational Algebra, SQL Queries
    Unit 2: Functional Dependency, Normalization
    Unit 3: Transaction Management, Concurrency Control
    Day Wise Lesson Plan
    """
    topics = extract_syllabus_topics(text, MappingConfig())
    assert "Relational Model" in topics
    assert "Normalization" in topics
    assert "Transaction Management" in topics


def test_extract_syllabus_topics_removes_noise_and_normalizes() -> None:
    text = """
    Syllabus:
    END SEM EXAM
    2023 (Link)
    Unit 1: Normalization, normalization.
    time
    points
    Day Wise Lesson Plan
    """
    topics = extract_syllabus_topics(text, MappingConfig())
    assert topics == ["Normalization"]


def test_extract_syllabus_topics_removes_incomplete_and_numbered_fragments() -> None:
    text = """
    Syllabus:
    Unit 4: id3 algorithm 2, clustering data points and, association rules
    Day Wise Lesson Plan
    """
    topics = extract_syllabus_topics(text, MappingConfig())
    assert "Id3 Algorithm" in topics
    assert "Association Rules" in topics
    assert "Clustering Data Points And" not in topics


def test_extract_syllabus_topics_ignores_long_sentences_and_intro_prefixes() -> None:
    text = """
    Syllabus:
    Introduction to database systems
    To understand data modeling and advanced enterprise analytics workflows
    - relational algebra
    - query processing
    Day Wise Lesson Plan
    """
    topics = extract_syllabus_topics(text, MappingConfig())
    assert "Relational Algebra" in topics
    assert "Query Processing" in topics
    assert "Introduction To Database Systems" not in topics


def test_extract_syllabus_topics_stops_before_lesson_plan() -> None:
    text = """
    Syllabus:
    Linear Regression, Logistic Regression
    Day Wise Lesson Plan
    Activity 1
    Case Study on Marketing
    """
    topics = extract_syllabus_topics(text, MappingConfig())
    assert topics == ["Linear Regression", "Logistic Regression"]
