import re
from collections.abc import Iterable

from exam_topic_predictor.schemas import Question
from exam_topic_predictor.text_utils import normalize_text

QUESTION_BOUNDARY = re.compile(
    r"(?im)^\s*(?:(?:question|q)\s*)?(\d{1,2}|[ivxlcdm]{1,8})\s*[\).:-]\s+"
)


def extract_questions(text: str, min_question_characters: int = 24) -> list[Question]:
    cleaned = normalize_text(text)
    matches = list(QUESTION_BOUNDARY.finditer(cleaned))
    if not matches:
        return _fallback_split(cleaned, min_question_characters)

    questions: list[Question] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        chunk = _squash_whitespace(cleaned[start:end])
        if len(chunk) < min_question_characters:
            continue
        questions.append(Question(question_id=_canonical_question_id(match.group(1), idx + 1), text=chunk))

    return questions if questions else _fallback_split(cleaned, min_question_characters)


def _fallback_split(text: str, min_question_characters: int) -> list[Question]:
    pieces: Iterable[str] = re.split(r"(?<=\?)\s+", text)
    questions: list[Question] = []
    for idx, piece in enumerate(pieces, start=1):
        chunk = _squash_whitespace(piece)
        if len(chunk) < min_question_characters:
            continue
        questions.append(Question(question_id=f"Q{idx}", text=chunk))
    return questions


def _canonical_question_id(raw_id: str, fallback_index: int) -> str:
    token = raw_id.strip().upper()
    if token.isdigit():
        return f"Q{int(token)}"
    return f"Q{fallback_index}"


def _squash_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
