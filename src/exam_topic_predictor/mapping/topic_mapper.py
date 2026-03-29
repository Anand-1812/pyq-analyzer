from collections.abc import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from exam_topic_predictor.embeddings import encode_texts
from exam_topic_predictor.schemas import Question, QuestionTopicMapping, TopicMatch


class TopicMapper:
    """Map questions to syllabus topics with sentence embeddings."""

    def __init__(
        self,
        topics: Sequence[str],
        min_similarity: float = 0.35,
        top_k_topics: int = 3,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        if not topics:
            raise ValueError("At least one syllabus topic is required for mapping.")

        self.topics = list(topics)
        self.min_similarity = min_similarity
        self.top_k_topics = max(1, top_k_topics)
        self.embedding_model_name = embedding_model_name
        self.topic_matrix = self._encode_texts(self.topics)

    def map_question(self, question: Question, year: int, paper_name: str) -> QuestionTopicMapping:
        """Return the top semantic topic matches for a question."""
        question_vector = self._encode_texts([question.text])
        similarities = cosine_similarity(self.topic_matrix, question_vector).reshape(-1)
        ranking = np.argsort(similarities)[::-1]

        matches: list[TopicMatch] = []
        for idx in ranking[: self.top_k_topics]:
            score = float(similarities[idx])
            if score < self.min_similarity:
                continue
            matches.append(TopicMatch(topic=self.topics[idx], similarity=round(score, 4)))

        if not matches and ranking.size:
            best_index = int(ranking[0])
            matches.append(TopicMatch(topic=self.topics[best_index], similarity=round(float(similarities[best_index]), 4)))

        return QuestionTopicMapping(year=year, paper_name=paper_name, question=question, matches=tuple(matches))

    def map_questions(self, questions: Sequence[Question], year: int, paper_name: str) -> list[QuestionTopicMapping]:
        """Map a batch of questions from a single paper."""
        if not questions:
            return []
        question_vectors = self._encode_texts([question.text for question in questions])
        similarity_matrix = cosine_similarity(question_vectors, self.topic_matrix)
        mappings: list[QuestionTopicMapping] = []

        for question, similarities in zip(questions, similarity_matrix, strict=False):
            ranking = np.argsort(similarities)[::-1]
            matches: list[TopicMatch] = []
            for idx in ranking[: self.top_k_topics]:
                score = float(similarities[idx])
                if score < self.min_similarity:
                    continue
                matches.append(TopicMatch(topic=self.topics[idx], similarity=round(score, 4)))

            if not matches and ranking.size:
                best_index = int(ranking[0])
                matches.append(
                    TopicMatch(topic=self.topics[best_index], similarity=round(float(similarities[best_index]), 4))
                )

            mappings.append(
                QuestionTopicMapping(year=year, paper_name=paper_name, question=question, matches=tuple(matches))
            )
        return mappings

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return encode_texts(texts, self.embedding_model_name)


def map_questions_to_topics(
    questions: Sequence[Question],
    syllabus_topics: Sequence[str],
    year: int,
    paper_name: str,
    min_similarity: float = 0.4,
    top_k_topics: int = 3,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> list[QuestionTopicMapping]:
    """Strictly map questions to the provided syllabus topics only."""
    mapper = TopicMapper(
        topics=syllabus_topics,
        min_similarity=min_similarity,
        top_k_topics=top_k_topics,
        embedding_model_name=embedding_model_name,
    )
    return mapper.map_questions(questions, year=year, paper_name=paper_name)
