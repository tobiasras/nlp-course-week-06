"""One DTU course, with every field from the course JSONL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _text(value: Any) -> str:
    """Turn a JSON value into one string. Lists are joined with newlines."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value).strip()


def _objectives(value: Any) -> list[str]:
    """Keep learning objectives as a list of strings."""
    if isinstance(value, list):
        return [text for item in value if (text := _text(item))]
    text = _text(value)
    return [text] if text else []


@dataclass
class Course:
    """A course record from dtu_courses.jsonl."""

    course_code: str
    url: str
    title: str
    academic_year: str
    learning_objectives: list[str]
    danish_title: str
    language_of_instruction: str
    ects: str
    course_type: str
    location: str
    duration_of_course: str
    type_of_assessment: str
    evaluation: str
    responsible: str
    department: str
    registration_sign_up: str
    scope_and_form: str
    schedule: str
    date_of_examination: str
    aid: str
    academic_prerequisites: str
    course_co_responsible: str
    green_challenge_participation: str
    not_applicable_together_with: str
    exam_duration: str
    previous_course: str
    participants_restrictions: str
    department_involved: str
    home_page: str
    mandatory_prerequisites: str
    external_institution: str
    offered_as: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Course:
        """Build a course from one JSON object."""
        fields = record.get("fields") or {}
        if not isinstance(fields, dict):
            fields = {}

        def field(key: str) -> str:
            return _text(fields.get(key))

        return cls(
            course_code=_text(record.get("course_code")),
            url=_text(record.get("url")),
            title=_text(record.get("title")),
            academic_year=_text(record.get("academic_year")),
            learning_objectives=_objectives(record.get("learning_objectives")),
            danish_title=field("Danish title"),
            language_of_instruction=field("Language of instruction"),
            ects=field("Point( ECTS )"),
            course_type=field("Course type"),
            location=field("Location"),
            duration_of_course=field("Duration of Course"),
            type_of_assessment=field("Type of assessment"),
            evaluation=field("Evaluation"),
            responsible=field("Responsible"),
            department=field("Department"),
            registration_sign_up=field("Registration Sign up"),
            scope_and_form=field("Scope and form"),
            schedule=field("Schedule"),
            date_of_examination=field("Date of examination"),
            aid=field("Aid"),
            academic_prerequisites=field("Academic prerequisites"),
            course_co_responsible=field("Course co-responsible"),
            green_challenge_participation=field("Green challenge participation"),
            not_applicable_together_with=field("Not applicable together with"),
            exam_duration=field("Exam duration"),
            previous_course=field("Previous Course"),
            participants_restrictions=field("Participants restrictions"),
            department_involved=field("Department involved"),
            home_page=field("Home page"),
            mandatory_prerequisites=field("Mandatory Prerequisites"),
            external_institution=field("External Institution"),
            offered_as=field("Offered as"),
        )
