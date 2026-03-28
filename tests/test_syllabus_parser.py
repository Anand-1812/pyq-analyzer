from exam_topic_predictor.parsing.syllabus_parser import extract_syllabus_topics


def test_extract_syllabus_topics_from_unit_lines() -> None:
    text = """
    Unit 1: Relational Model, Relational Algebra, SQL Queries
    Unit 2: Functional Dependency, Normalization
    Unit 3: Transaction Management, Concurrency Control
    """
    topics = extract_syllabus_topics(text)
    assert "Relational Model" in topics
    assert "Normalization" in topics
    assert "Transaction Management" in topics
