import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from exam_topic_predictor.text_utils import normalize_line, normalize_text

NOISE_KEYS = (
    "maximum marks",
    "max marks",
    "time allowed",
    "instructions",
    "course outcomes",
    "reference books",
)

LEADING_INDEX = re.compile(r"^\s*(?:unit|module|chapter)?\s*[ivxlcdm\d.]*\s*[:.)-]?\s*", re.IGNORECASE)
SPLIT_CHARS = re.compile(r"\s*[,;/]\s*")


def extract_syllabus_topics(text: str) -> list[str]:
    cleaned = normalize_text(text)
    lines = [normalize_line(line) for line in cleaned.splitlines()]
    candidates: list[str] = []

    for line in lines:
        if _is_noise(line):
            continue

        normalized = LEADING_INDEX.sub("", line).strip(" -:")
        if not normalized:
            continue

        if "," in normalized or ";" in normalized or "/" in normalized:
            parts = [segment.strip(" .") for segment in SPLIT_CHARS.split(normalized)]
            candidates.extend(_filter_topics(parts))
            continue

        candidates.extend(_filter_topics([normalized]))

    if candidates:
        return _dedupe_preserve_order(candidates)

    return _fallback_topics_from_text(cleaned)


def _filter_topics(candidates: list[str]) -> list[str]:
    topics: list[str] = []
    for item in candidates:
        words = item.split()
        if len(words) > 12:
            continue
        lower_words = [word.lower().strip(".,:;()[]{}") for word in words]
        if not lower_words:
            continue
        if len(words) == 1 and (len(lower_words[0]) < 4 or lower_words[0] in ENGLISH_STOP_WORDS):
            continue
        if all(word in ENGLISH_STOP_WORDS for word in lower_words):
            continue
        topics.append(item.strip())
    return topics


def _fallback_topics_from_text(text: str) -> list[str]:
    chunks = re.split(r"[.\n]", text)
    topics = _filter_topics([normalize_line(chunk) for chunk in chunks if chunk.strip()])
    return _dedupe_preserve_order(topics)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _is_noise(line: str) -> bool:
    if not line:
        return True
    lower = line.casefold()
    if any(key in lower for key in NOISE_KEYS):
        return True
    if re.fullmatch(r"[\d\s./-]+", line):
        return True
    if len(line.split()) > 20:
        return True
    return False
