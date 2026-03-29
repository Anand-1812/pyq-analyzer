"""Cluster semantically similar questions into reusable exam patterns."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from exam_topic_predictor.config import ForecastConfig, MappingConfig
from exam_topic_predictor.embeddings import encode_texts
from exam_topic_predictor.mapping import TopicMapper
from exam_topic_predictor.schemas import QuestionPattern, QuestionTopicMapping


class QuestionPatternMiner:
    """Find recurring question patterns and tie them back to syllabus topics."""

    def __init__(self, mapping_config: MappingConfig, forecast_config: ForecastConfig) -> None:
        self.mapping_config = mapping_config
        self.forecast_config = forecast_config

    def extract_patterns(
        self,
        mappings: Sequence[QuestionTopicMapping],
        topics: Sequence[str] | None = None,
        top_k_per_topic: int = 3,
    ) -> list[QuestionPattern]:
        """Cluster all questions, then map each representative cluster question to a syllabus topic."""
        if len(mappings) < self.forecast_config.min_pattern_cluster_size or not topics:
            return []

        topic_mapper = TopicMapper(
            topics=list(topics or []),
            min_similarity=self.mapping_config.min_similarity,
            top_k_topics=1,
            embedding_model_name=self.mapping_config.embedding_model_name,
        )
        allowed_topics = set(topics or [])
        questions = [mapping.question.text for mapping in mappings]
        embeddings = encode_texts(questions, self.mapping_config.embedding_model_name)
        cluster_count = min(
            len(questions),
            max(
                self.forecast_config.min_question_clusters,
                int(math.sqrt(len(questions))),
            ),
        )
        cluster_count = max(1, cluster_count)
        labels = KMeans(
            n_clusters=cluster_count,
            n_init=10,
            random_state=42,
        ).fit_predict(embeddings)

        clusters_by_topic: dict[str, list[QuestionPattern]] = defaultdict(list)
        clusters: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            clusters[int(label)].append(index)

        for cluster_index, question_indices in clusters.items():
            cluster_embeddings = embeddings[question_indices]
            centroid = np.mean(cluster_embeddings, axis=0, dtype=float, keepdims=True)
            similarities_to_centroid = cosine_similarity(cluster_embeddings, centroid).reshape(-1)
            representative_index = question_indices[int(np.argmax(similarities_to_centroid))]
            representative_mapping = mappings[representative_index]
            topic_mapping = topic_mapper.map_question(
                representative_mapping.question,
                year=representative_mapping.year,
                paper_name=representative_mapping.paper_name,
            )
            if not topic_mapping.matches:
                continue
            topic_match = topic_mapping.matches[0]
            if allowed_topics and topic_match.topic not in allowed_topics:
                continue

            years = sorted({mappings[index].year for index in question_indices})
            clusters_by_topic[topic_match.topic].append(
                QuestionPattern(
                    topic=topic_match.topic,
                    pattern_id=f"{topic_match.topic[:24].replace(' ', '_')}_{cluster_index + 1}",
                    representative_question=representative_mapping.question.text,
                    question_count=len(question_indices),
                    similarity_to_topic=round(topic_match.similarity, 4),
                    years=tuple(years),
                )
            )

        patterns: list[QuestionPattern] = []
        for topic, topic_patterns in clusters_by_topic.items():
            del topic  # keeps loop explicit and avoids nested list comprehension
            topic_patterns.sort(key=lambda item: (item.question_count, item.similarity_to_topic), reverse=True)
            patterns.extend(topic_patterns[:top_k_per_topic])

        patterns.sort(key=lambda item: (item.question_count, item.years[-1], item.similarity_to_topic), reverse=True)
        return patterns


def predict_questions(
    mappings: Sequence[QuestionTopicMapping],
    syllabus_topics: Sequence[str],
    mapping_config: MappingConfig,
    forecast_config: ForecastConfig,
    top_k_per_topic: int = 3,
) -> list[QuestionPattern]:
    """Cluster questions and return representative predicted questions per syllabus topic."""
    miner = QuestionPatternMiner(mapping_config=mapping_config, forecast_config=forecast_config)
    return miner.extract_patterns(mappings, topics=syllabus_topics, top_k_per_topic=top_k_per_topic)
