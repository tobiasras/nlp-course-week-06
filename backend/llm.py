"""Generate answers from retrieved courses with the CampusAI LLM."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from course import Course

SNIPPET_LIMIT = 1500


class LLM:
    """Call the local OpenAI-compatible model."""

    def __init__(self) -> None:
        load_dotenv(Path(__file__).resolve().parent / ".env")
        self.api_key = os.getenv("CAMPUSAI_API_KEY")
        self.model = os.getenv("CAMPUSAI_MODEL", "cai-vision")
        self.api_url = os.getenv(
            "CAMPUSAI_API_URL", "https://api.campusai.compute.dtu.dk/v1"
        )
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.api_url, timeout=60)
            if self.api_key
            else None
        )

    def prep_query(self, user_query: str, results: list[tuple[Course, float]]) -> str:
        """Aggregate retrieved courses into one prompt."""
        blocks = [self._snippet(course) for course, _score in results]
        context = "\n\n".join(blocks)
        return (
            "You are a helpful assistant for DTU courses.\n"
            "Use the context below to answer the user's question.\n"
            "If the answer is not present, say you don't know.\n\n"
            f"Question: {user_query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

    def fulfill(self, prompt: str) -> str:
        """Send the prompt to the LLM and return the answer text."""
        if self.client is None:
            raise RuntimeError(
                "CAMPUSAI_API_KEY is not set. Add it to backend/.env to answer questions."
            )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise RuntimeError(f"The language model request failed: {exc}") from exc
        return (response.choices[0].message.content or "").strip()

    def query(self, user_query: str, results: list[tuple[Course, float]]) -> str:
        """Build a prompt from the search results and ask the LLM."""
        prompt = self.prep_query(user_query, results)
        return self.fulfill(prompt)

    def _snippet(self, course: Course) -> str:
        """One course, shortened so the prompt stays small."""
        objectives = "\n".join(f"- {item}" for item in course.learning_objectives)
        text = "\n".join(
            [
                f"Course code: {course.course_code}",
                f"Title: {course.title}",
                f"Responsible: {course.responsible}",
                f"Course co-responsible: {course.course_co_responsible}",
                f"Department: {course.department}",
                f"ECTS: {course.ects}",
                f"Learning objectives:\n{objectives}",
            ]
        )
        if len(text) <= SNIPPET_LIMIT:
            return text
        return text[: SNIPPET_LIMIT - 3].rstrip() + "..."
