"""Dense retrieval over DTU courses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from course import Course

EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"


def courses_path() -> Path:
    """Return the path to the course JSONL file."""
    configured = os.environ.get("COURSES_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "data" / "dtu_courses.jsonl"


def passage_texts(course: Course) -> list[str]:
    """Texts that are embedded for one course."""
    objectives = "\n".join(course.learning_objectives)
    values = [
        course.title,
        course.responsible,
        course.course_co_responsible,
        course.department,
        objectives,
    ]
    return [value.strip() for value in values if value and value.strip()]


class Data:
    """Load course embeddings and search them with cosine similarity."""

    def __init__(self) -> None:
        self.courses: list[Course] = []
        self.vectors: np.ndarray | None = None
        self.course_index: np.ndarray | None = None
        self._model: object | None = None

    def load(self, path: str | Path | None = None) -> None:
        """Load courses and embed the fields used for search."""
        data_path = Path(path) if path is not None else courses_path()
        if not data_path.is_file():
            raise FileNotFoundError(f"course data not found: {data_path}")

        courses: list[Course] = []
        seen: set[str] = set()
        lines = data_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{data_path} line {line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{data_path} line {line_number} is not a JSON object")
            course = Course.from_record(record)
            if not course.course_code or course.course_code in seen:
                continue
            seen.add(course.course_code)
            courses.append(course)

        texts: list[str] = []
        indexes: list[int] = []
        for index, course in enumerate(courses):
            for text in passage_texts(course):
                texts.append(text)
                indexes.append(index)

        if texts:
            vectors = self._embed(texts)
            course_index = np.asarray(indexes, dtype=np.int32)
        else:
            vectors = np.zeros((0, 0), dtype=np.float64)
            course_index = np.zeros(0, dtype=np.int32)
        self.courses = courses
        self.vectors = vectors
        self.course_index = course_index

    def search(self, query: str, topk: int = 5) -> list[tuple[Course, float]]:
        """Return the closest unique courses for a query."""
        if self.vectors is None or self.course_index is None:
            raise RuntimeError("course index is not loaded")
        if topk <= 0 or len(self.courses) == 0 or self.vectors.size == 0:
            return []

        query_vector = self._embed([query])[0]
        scores = self.vectors @ query_vector
        best: dict[int, float] = {}
        for row, course_i in enumerate(self.course_index):
            course_i = int(course_i)
            score = float(scores[row])
            if course_i not in best or score > best[course_i]:
                best[course_i] = score

        ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)[:topk]
        return [(self.courses[index], score) for index, score in ranked]

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Turn texts into normalized embedding vectors."""
        model = self._load_model()
        vectors = np.asarray(
            model.encode(texts, normalize_embeddings=True),
            dtype=np.float64,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    def _load_model(self):
        """Load the sentence embedding model once."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model
