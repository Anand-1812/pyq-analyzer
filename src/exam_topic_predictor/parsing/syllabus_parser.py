import re

from exam_topic_predictor.config import MappingConfig
from exam_topic_predictor.parsing.topic_cleaner import clean_topic_candidates, format_topic_for_display
from exam_topic_predictor.text_utils import normalize_line, normalize_text

NOISE_KEYS = (
    "maximum marks",
    "max marks",
    "time allowed",
    "instructions",
    "course outcomes",
    "reference books",
    "activities",
    "activity",
    "case study",
    "case studies",
    "lesson plan",
)

UNIT_PREFIX = re.compile(r"^\s*(?:unit|module|chapter)\s*[ivxlcdm\d.]*\s*[:.)-]?\s*", re.IGNORECASE)
NUMBER_PREFIX = re.compile(r"^\s*(?:[ivxlcdm]+|\d+)\s*[:.)-]\s*", re.IGNORECASE)
SPLIT_CHARS = re.compile(r"\s*[,;/•]\s*")
BULLET_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
LECTURE_PREFIX = re.compile(r"^\s*(?:lecture|lec)\s*\d+\s*[:.)-]?\s*", re.IGNORECASE)
SYLLABUS_START = re.compile(r"\bsyllabus\s*:\s*", re.IGNORECASE)
SYLLABUS_END = re.compile(r"\bday\s+wise\s+lesson\s+plan\b", re.IGNORECASE)


def extract_syllabus_topics(text: str, config: MappingConfig | None = None) -> list[str]:
    """Extract syllabus headings or bullet topics and clean them aggressively."""
    mapping_config = config or MappingConfig()
    cleaned = _extract_syllabus_section(normalize_text(text))
    lines = [normalize_line(line) for line in cleaned.splitlines()]
    candidates: list[str] = []

    for line in lines:
        if _is_noise(line):
            continue

        normalized = BULLET_PREFIX.sub("", line)
        normalized = LECTURE_PREFIX.sub("", normalized)
        normalized = UNIT_PREFIX.sub("", normalized)
        normalized = NUMBER_PREFIX.sub("", normalized).strip(" -:")
        if not normalized:
            continue

        if "," in normalized or ";" in normalized or "/" in normalized:
            parts = [segment.strip(" .") for segment in SPLIT_CHARS.split(normalized)]
            candidates.extend(part for part in parts if _looks_like_topic_line(part, mapping_config))
            continue

        if not _looks_like_topic_line(normalized, mapping_config):
            continue
        candidates.append(normalized)

    if candidates:
        cleaned_topics = clean_topic_candidates(
            candidates,
            embedding_model_name=mapping_config.embedding_model_name,
            semantic_threshold=mapping_config.topic_dedupe_similarity,
            max_topic_words=min(5, mapping_config.max_topic_words),
            minimum_topic_words=mapping_config.minimum_topic_words,
            blocked_prefixes=mapping_config.topic_heading_prefixes,
        )
        return [format_topic_for_display(topic) for topic in cleaned_topics]

    return _fallback_topics_from_text(cleaned, mapping_config)


def _fallback_topics_from_text(text: str, config: MappingConfig) -> list[str]:
    chunks = re.split(r"[.\n]", text)
    cleaned_topics = clean_topic_candidates(
        [normalize_line(chunk) for chunk in chunks if chunk.strip()],
        embedding_model_name=config.embedding_model_name,
        semantic_threshold=config.topic_dedupe_similarity,
        max_topic_words=min(5, config.max_topic_words),
        minimum_topic_words=config.minimum_topic_words,
        blocked_prefixes=config.topic_heading_prefixes,
    )
    return [format_topic_for_display(topic) for topic in cleaned_topics]


def _is_noise(line: str) -> bool:
    if not line:
        return True
    lower = line.casefold()
    if any(key in lower for key in NOISE_KEYS):
        return True
    if re.fullmatch(r"[\d\s./-]+", line):
        return True
    if len(line.split()) > 10:
        return True
    return False


def _looks_like_topic_line(line: str, config: MappingConfig) -> bool:
    word_count = len(line.split())
    if word_count == 0 or word_count > min(5, config.max_topic_words):
        return False
    lowered = line.casefold()
    if any(lowered.startswith(prefix) for prefix in config.topic_heading_prefixes):
        return False
    return True


def _extract_syllabus_section(text: str) -> str:
    """Keep only the content inside the syllabus section of the document."""
    start_match = SYLLABUS_START.search(text)
    if not start_match:
        return text
    section = text[start_match.end() :]
    end_match = SYLLABUS_END.search(section)
    if end_match:
        section = section[: end_match.start()]
    return section.strip()
