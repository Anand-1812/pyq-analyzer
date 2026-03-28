from exam_topic_predictor.parsing.question_parser import extract_questions


def test_extract_questions_with_numbered_format() -> None:
    text = """
    1) Explain normalization in DBMS with suitable examples.
    2) What is indexing? Compare clustered and non-clustered indexing.
    3) Explain transaction properties and ACID with examples.
    """
    questions = extract_questions(text, min_question_characters=20)
    assert len(questions) == 3
    assert questions[0].question_id == "Q1"
    assert "normalization" in questions[0].text.lower()


def test_fallback_question_split_works_without_numbering() -> None:
    text = "Define operating system? Explain process scheduling? What is deadlock and prevention?"
    questions = extract_questions(text, min_question_characters=10)
    assert len(questions) == 3
