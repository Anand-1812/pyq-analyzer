from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from exam_topic_predictor.schemas import Question, QuestionTopicMapping, TopicMatch


class TopicMapper:
    def __init__(self, topics: Sequence[str], min_similarity: float = 0.18, top_k_topics: int = 3) -> None:
        if not topics:
            raise ValueError("At least one syllabus topic is required for mapping.")

        self.topics = list(topics)
        self.min_similarity = min_similarity
        self.top_k_topics = max(1, top_k_topics)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), min_df=1)
        self.topic_matrix = self.vectorizer.fit_transform(self.topics)

    def map_question(self, question: Question, year: int) -> QuestionTopicMapping:
        question_vector = self.vectorizer.transform([question.text])
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

        return QuestionTopicMapping(year=year, question=question, matches=tuple(matches))

    def map_questions(self, questions: Sequence[Question], year: int) -> list[QuestionTopicMapping]:
        return [self.map_question(question, year) for question in questions]
