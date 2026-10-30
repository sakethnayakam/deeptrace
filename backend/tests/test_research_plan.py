import json

import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.main import app
from app.services.gemini import GeminiServiceError
from app.services.research_planner import ResearchPlanner


class FakeGeminiService:
    def __init__(self, generated_text: str) -> None:
        self.generated_text = generated_text
        self.prompt: str | None = None

    def generate_json(self, prompt: str) -> str:
        self.prompt = prompt
        return self.generated_text


class FailingGeminiService:
    def generate_json(self, prompt: str) -> str:
        raise GeminiServiceError("A provider error that must stay private")


class TrackingGeminiService:
    def __init__(self) -> None:
        self.was_called = False

    def generate_json(self, prompt: str) -> str:
        self.was_called = True
        return '{"subquestions": []}'


client = TestClient(app)


def valid_plan_json() -> str:
    return json.dumps(
        {
            "subquestions": [
                "How is software developer productivity measured?",
                "What evidence compares remote and office-based work?",
                "What limitations affect the available evidence?",
            ]
        }
    )


def test_research_plan_returns_validated_question_and_subquestions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeGeminiService(valid_plan_json())
    monkeypatch.setattr(main, "get_gemini_service", lambda request: service)

    response = client.post(
        "/research/plan",
        json={"question": "  Does remote work improve developer productivity?  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "Does remote work improve developer productivity?",
        "subquestions": [
            "How is software developer productivity measured?",
            "What evidence compares remote and office-based work?",
            "What limitations affect the available evidence?",
        ],
    }


@pytest.mark.parametrize("question", ["", "   "])
def test_research_plan_rejects_blank_questions(question: str) -> None:
    response = client.post("/research/plan", json={"question": question})

    assert response.status_code == 422


def test_research_plan_rejects_an_overly_long_question_before_calling_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TrackingGeminiService()
    monkeypatch.setattr(main, "get_gemini_service", lambda request: service)

    response = client.post("/research/plan", json={"question": "a" * 4_001})

    assert response.status_code == 422
    assert service.was_called is False


def test_research_plan_reports_missing_gemini_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "get_gemini_api_key", lambda: None)

    with TestClient(app) as configured_client:
        response = configured_client.post(
            "/research/plan",
            json={"question": "What is RAG?"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Gemini service is not configured."}


def test_research_plan_hides_gemini_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "get_gemini_service", lambda request: FailingGeminiService())

    response = client.post("/research/plan", json={"question": "What is RAG?"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Gemini could not create a research plan. Please try again later."
    }


@pytest.mark.parametrize(
    "generated_text",
    [
        "not valid JSON",
        "{}",
        json.dumps({"subquestions": ["One", "Two"]}),
        json.dumps({"subquestions": ["One", "Two", "Three", "Four", "Five", "Six", "Seven"]}),
        json.dumps({"subquestions": ["One", "one", "Three"]}),
        json.dumps({"subquestions": ["One", "   ", "Three"]}),
        json.dumps({"subquestions": ["One", "Two", "a" * 501]}),
    ],
)
def test_research_plan_rejects_invalid_generated_plans(
    monkeypatch: pytest.MonkeyPatch,
    generated_text: str,
) -> None:
    service = FakeGeminiService(generated_text)
    monkeypatch.setattr(main, "get_gemini_service", lambda request: service)

    response = client.post("/research/plan", json={"question": "What is RAG?"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Gemini could not create a research plan. Please try again later."
    }


def test_research_planner_sends_a_prompt_and_parses_valid_json() -> None:
    service = FakeGeminiService(valid_plan_json())

    subquestions = ResearchPlanner(service).plan("What is RAG?")

    assert service.prompt is not None
    assert "What is RAG?" in service.prompt
    assert subquestions == [
        "How is software developer productivity measured?",
        "What evidence compares remote and office-based work?",
        "What limitations affect the available evidence?",
    ]
