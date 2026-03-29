"""Utilities for cleaning and deduplicating syllabus topics."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import string

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from exam_topic_predictor.embeddings import encode_texts

NOISE_PHRASES = (
    "end sem exam",
    "mid sem exam",
    "midterm",
    "time",
    "points",
    "marks",
    "instructions",
    "link",
    "course outcomes",
    "reference books",
    "maximum marks",
    "time allowed",
)
MIN_TOPIC_TOKEN_LENGTH = 3
MAX_TOPIC_WORDS = 12
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
INCOMPLETE_ENDINGS = ("and", "of", "for", "to", "with", "in", "on", "by", "like")
TRAILING_NUMBER_PATTERN = re.compile(r"\s+\d+$")
BROKEN_QUOTE_PATTERN = re.compile(r"[\"'`]+")
VERB_HINTS = (
    "understand",
    "describe",
    "explain",
    "discuss",
    "identify",
    "learn",
    "compare",
    "analyze",
    "evaluate",
)


def clean_topic_candidates(
    candidates: list[str],
    embedding_model_name: str = "all-MiniLM-L6-v2",
    semantic_threshold: float = 0.85,
    max_topic_words: int = 10,
    minimum_topic_words: int = 1,
    blocked_prefixes: tuple[str, ...] = (),
) -> list[str]:
    """Normalize, filter, and deduplicate syllabus topic candidates."""
    normalized_topics: list[str] = []
    for candidate in candidates:
        normalized = normalize_topic(candidate)
        if not normalized or _is_meaningless_topic(
            normalized,
            max_topic_words=max_topic_words,
            minimum_topic_words=minimum_topic_words,
            blocked_prefixes=blocked_prefixes,
        ):
            continue
        normalized_topics.append(normalized)
    return dedupe_similar_topics(
        normalized_topics,
        embedding_model_name=embedding_model_name,
        semantic_threshold=semantic_threshold,
    )


def normalize_topic(value: str) -> str:
    """Lowercase and strip punctuation/noise from a raw topic string."""
    compact = re.sub(r"\s+", " ", value).strip()
    compact = re.sub(r"\([^)]*\)", "", compact)
    compact = TRAILING_NUMBER_PATTERN.sub("", compact)
    compact = BROKEN_QUOTE_PATTERN.sub("", compact)
    compact = compact.translate(PUNCT_TRANSLATION)
    compact = compact.casefold().strip()
    compact = re.sub(r"\s+", " ", compact)
    return compact


def format_topic_for_display(value: str) -> str:
    """Convert a normalized topic into a clean title-cased label."""
    words: list[str] = []
    for token in value.split():
        if token.isdigit():
            words.append(token)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def dedupe_similar_topics(
    topics: list[str],
    embedding_model_name: str = "all-MiniLM-L6-v2",
    semantic_threshold: float = 0.85,
) -> list[str]:
    """Remove duplicates and near-duplicates while preserving order."""
    deduped: list[str] = []
    seen_embeddings = None
    try:
        seen_embeddings = encode_texts(topics, embedding_model_name)
    except Exception:  # pragma: no cover - fallback for offline or missing-model environments
        seen_embeddings = None

    for topic in topics:
        if any(_topic_similarity(topic, existing) >= 0.92 for existing in deduped):
            continue
        deduped.append(topic)
    if seen_embeddings is None or len(deduped) < 2:
        return deduped

    deduped_embeddings = encode_texts(deduped, embedding_model_name)
    merged_topics: list[str] = []
    keep_mask = [True] * len(deduped)
    for index, topic in enumerate(deduped):
        if not keep_mask[index]:
            continue
        merged_topics.append(topic)
        similarities = cosine_similarity(deduped_embeddings[index : index + 1], deduped_embeddings).reshape(-1)
        for other_index in range(index + 1, len(deduped)):
            if similarities[other_index] >= semantic_threshold and _topic_similarity(topic, deduped[other_index]) >= 0.88:
                keep_mask[other_index] = False
    return merged_topics


def _is_meaningless_topic(
    topic: str,
    max_topic_words: int = 10,
    minimum_topic_words: int = 1,
    blocked_prefixes: tuple[str, ...] = (),
) -> bool:
    words = topic.split()
    if not words or len(words) > min(max_topic_words, MAX_TOPIC_WORDS):
        return True
    if len(words) < max(1, minimum_topic_words):
        return True
    if topic in NOISE_PHRASES:
        return True
    if re.fullmatch(r"[\d\s]+", topic):
        return True
    if words[-1] in INCOMPLETE_ENDINGS:
        return True
    if any(topic.startswith(prefix) for prefix in blocked_prefixes):
        return True

    valid_words = [
        word
        for word in words
        if len(word) >= MIN_TOPIC_TOKEN_LENGTH and word not in ENGLISH_STOP_WORDS and not word.isdigit()
    ]
    if len(valid_words) == 0:
        return True
    if len(words) >= 4 and any(word in VERB_HINTS for word in words[:2]):
        return True
    return False


def _topic_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    return SequenceMatcher(a=left, b=right).ratio()
