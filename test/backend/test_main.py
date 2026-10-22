import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from course import Course
from data import Data, passage_texts
from main import app


def test_root(monkeypatch):
    monkeypatch.setattr("main.data_service.load", lambda: None)
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "app is running"}


def test_course_from_record_keeps_every_field():
    course = Course.from_record(
        {
            "course_code": "02465",
            "url": "https://kurser.dtu.dk/course/02465",
            "title": "02465 Introduction to reinforcement learning and control",
            "academic_year": "2025/2026",
            "learning_objectives": ["Explain the objective", ""],
            "fields": {
                "Responsible": "Tue Herlau",
                "Point( ECTS )": 5,
                "Schedule": ["Autumn", "E4A"],
                "Course co-responsible": ["Ada Lovelace", "Grace Hopper"],
            },
        }
    )
    assert course.course_code == "02465"
    assert course.ects == "5"
    assert course.responsible == "Tue Herlau"
    assert course.schedule == "Autumn\nE4A"
    assert course.course_co_responsible == "Ada Lovelace\nGrace Hopper"
    assert course.learning_objectives == ["Explain the objective"]
    assert course.department == ""


def test_passage_texts_skip_empty_fields():
    course = Course.from_record(
        {
            "course_code": "02465",
            "title": "02465 Control",
            "learning_objectives": ["Explain control"],
            "fields": {"Responsible": "Tue Herlau", "Department": "01 Compute"},
        }
    )
    assert passage_texts(course) == [
        "02465 Control",
        "Tue Herlau",
        "01 Compute",
        "Explain control",
    ]


def test_search_returns_unique_courses_by_best_score():
    service = Data()
    service.courses = [
        Course.from_record({"course_code": "A", "title": "Alpha"}),
        Course.from_record({"course_code": "B", "title": "Beta"}),
    ]
    service.vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]],
        dtype=np.float64,
    )
    service.course_index = np.array([0, 0, 1], dtype=np.int32)
    service._embed = lambda texts: np.array([[1.0, 0.0]])

    hits = service.search("alpha", topk=5)

    assert [course.course_code for course, _score in hits] == ["A", "B"]
    assert hits[0][1] == 1.0
    assert hits[1][1] == 0.8


def test_load_rejects_invalid_json(tmp_path):
    path = tmp_path / "courses.jsonl"
    path.write_text('{"course_code": "1"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        Data().load(path)


def test_non_dict_fields_are_ignored():
    course = Course.from_record({"course_code": "1", "title": "One", "fields": "nope"})
    assert course.course_code == "1"
    assert course.department == ""


def test_search_reports_when_index_is_not_loaded(monkeypatch):
    monkeypatch.setattr("main.data_service.load", lambda: None)
    response = TestClient(app).get("/v1/search", params={"query": "MRI"})
    assert response.status_code == 503
    assert response.json()["detail"] == "course index is not loaded"
