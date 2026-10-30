import asyncio
import json
import logging
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.main import app
from app.services.research_planner import ResearchPlannerError
from app.services.web_search import WebSearchServiceError


SUBQUESTIONS = [
    "What evidence supports the claim?",
    "What are the main limitations?",
    "How do findings compare across studies?",
]


class FakeGeminiService:
    def ask(self, question: str) -> str:
        return "Mocked answer"

    def close(self) -> None:
        pass


class FakePlanner:
    def __init__(self, gemini_service: FakeGeminiService, subquestions: list[str] | None = None) -> None:
        self.gemini_service = gemini_service
        self.question: str | None = None
        self.subquestions = subquestions or SUBQUESTIONS

    def plan(self, question: str) -> list[str]:
        self.question = question
        return self.subquestions


class FailingPlanner:
    def __init__(self, gemini_service: FakeGeminiService) -> None:
        self.gemini_service = gemini_service

    def plan(self, question: str) -> list[str]:
        raise ResearchPlannerError("A provider error that must stay private")


class FakeSearchService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str) -> list[dict[str, str | float]]:
        self.queries.append(query)
        return [{"title": f"Result for {query}", "url": f"https://example.com/{query}", "snippet": f"Evidence about {query}", "score": 0.9, "published_date": "2026-10-30"}]


class FailingSearchService:
    async def search(self, query: str) -> list[dict[str, str | float]]:
        raise WebSearchServiceError("A provider error that must stay private")


client = TestClient(app)


def test_research_executes_each_planned_subquestion_and_groups_results(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=main.logger.name)
    planner = FakePlanner(FakeGeminiService())
    search_service = FakeSearchService()
    monkeypatch.setattr(main, "get_gemini_service", lambda request: FakeGeminiService())
    monkeypatch.setattr(main, "get_search_service", lambda request: search_service)
    monkeypatch.setattr(main, "ResearchPlanner", lambda gemini_service: planner)
    response = client.post("/research", json={"question": "  Is remote work effective?  "})
    assert response.status_code == 200
    assert response.json() == {"question": "Is remote work effective?", "subquestions": SUBQUESTIONS, "search_results": [{"subquestion": subquestion, "results": [{"title": f"Result for {subquestion}", "url": f"https://example.com/{subquestion}", "snippet": f"Evidence about {subquestion}", "score": 0.9, "published_date": "2026-10-30"}]} for subquestion in SUBQUESTIONS]}
    assert planner.question == "Is remote work effective?"
    assert sorted(search_service.queries) == sorted(SUBQUESTIONS)
    assert len(search_service.queries) == len(SUBQUESTIONS)
    log_messages = [record.getMessage() for record in caplog.records]
    assert sum("research_search_completed" in message for message in log_messages) == 3
    assert any(
        "research_completed" in message
        and "successful_searches=3" in message
        and "failed_searches=0" in message
        for message in log_messages
    )
    performance_records = [
        record for record in caplog.records if "research_" in record.getMessage()
    ]
    assert all(record.levelno == logging.INFO for record in performance_records)
    assert all(record.name == main.logger.name for record in performance_records)
    assert all("Is remote work effective?" not in message for message in log_messages)


def test_research_limits_concurrency_and_preserves_subquestion_order(monkeypatch: pytest.MonkeyPatch) -> None:
    subquestions = [f"Question {number}" for number in range(1, 6)]

    class ConcurrentSearchService:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def search(self, query: str) -> list[dict[str, str]]:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.06 - (int(query.rsplit(" ", 1)[1]) * 0.01))
                return [{"title": query, "url": f"https://example.com/{query}", "snippet": query}]
            finally:
                self.active -= 1

    planner = FakePlanner(FakeGeminiService(), subquestions)
    search_service = ConcurrentSearchService()
    monkeypatch.setattr(main, "get_gemini_service", lambda request: FakeGeminiService())
    monkeypatch.setattr(main, "get_search_service", lambda request: search_service)
    monkeypatch.setattr(main, "ResearchPlanner", lambda gemini_service: planner)
    response = client.post("/research", json={"question": "What is RAG?"})
    assert response.status_code == 200
    assert search_service.max_active == 3
    assert [group["subquestion"] for group in response.json()["search_results"]] == subquestions


@pytest.mark.parametrize("question", ["", "   "])
def test_research_rejects_blank_questions(question: str) -> None:
    assert client.post("/research", json={"question": question}).status_code == 422


def test_research_rejects_an_overly_long_question(monkeypatch: pytest.MonkeyPatch) -> None:
    search_service = FakeSearchService()
    monkeypatch.setattr(main, "get_gemini_service", lambda request: FakeGeminiService())
    monkeypatch.setattr(main, "get_search_service", lambda request: search_service)
    monkeypatch.setattr(main, "ResearchPlanner", FakePlanner)
    assert client.post("/research", json={"question": "a" * 4_001}).status_code == 422
    assert search_service.queries == []


def test_research_reports_missing_gemini_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "get_gemini_api_key", lambda: None)
    monkeypatch.setattr(config, "get_tavily_api_key", lambda: "test-key")
    with TestClient(app) as configured_client:
        response = configured_client.post("/research", json={"question": "What is RAG?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Gemini service is not configured."}


def test_research_reports_missing_tavily_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "get_gemini_api_key", lambda: "test-key")
    monkeypatch.setattr(config, "get_tavily_api_key", lambda: None)
    monkeypatch.setattr(main, "GeminiService", lambda api_key: FakeGeminiService())
    with TestClient(app) as configured_client:
        response = configured_client.post("/research", json={"question": "What is RAG?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Web search service is not configured."}


def test_research_hides_planner_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "get_gemini_service", lambda request: FakeGeminiService())
    monkeypatch.setattr(main, "get_search_service", lambda request: FakeSearchService())
    monkeypatch.setattr(main, "ResearchPlanner", FailingPlanner)
    response = client.post("/research", json={"question": "What is RAG?"})
    assert response.status_code == 502
    assert response.json() == {"detail": "Research could not be completed. Please try again later."}


def test_research_returns_partial_results_when_one_search_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=main.logger.name)
    class PartiallyFailingSearchService(FakeSearchService):
        async def search(self, query: str) -> list[dict[str, str | float]]:
            if query == SUBQUESTIONS[1]:
                raise WebSearchServiceError("provider detail")
            return await super().search(query)

    monkeypatch.setattr(main, "get_gemini_service", lambda request: FakeGeminiService())
    monkeypatch.setattr(main, "get_search_service", lambda request: PartiallyFailingSearchService())
    monkeypatch.setattr(main, "ResearchPlanner", FakePlanner)
    response = client.post("/research", json={"question": "What is RAG?"})
    assert response.status_code == 200
    groups = response.json()["search_results"]
    assert groups[1] == {"subquestion": SUBQUESTIONS[1], "results": [], "error": "Web search could not be completed for this subquestion."}
    assert "error" not in groups[0]
    assert "error" not in groups[2]
    log_messages = [record.getMessage() for record in caplog.records]
    assert any("subquestion_index=1 status=failed" in message for message in log_messages)
    assert any(
        "research_completed" in message
        and "successful_searches=2" in message
        and "failed_searches=1" in message
        for message in log_messages
    )


def test_research_returns_safe_error_when_all_searches_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "get_gemini_service", lambda request: FakeGeminiService())
    monkeypatch.setattr(main, "get_search_service", lambda request: FailingSearchService())
    monkeypatch.setattr(main, "ResearchPlanner", FakePlanner)
    response = client.post("/research", json={"question": "What is RAG?"})
    assert response.status_code == 502
    assert response.json() == {"detail": "Research web searches could not be completed. Please try again later."}


def test_existing_endpoints_remain_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExistingEndpointGeminiService(FakeGeminiService):
        def generate_json(self, prompt: str) -> str:
            return json.dumps({"subquestions": ["One", "Two", "Three"]})

    gemini_service = ExistingEndpointGeminiService()
    search_service = FakeSearchService()
    monkeypatch.setattr(main, "get_gemini_service", lambda request: gemini_service)
    monkeypatch.setattr(main, "get_search_service", lambda request: search_service)
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.post("/ask", json={"question": "What is RAG?"}).status_code == 200
    assert client.post("/research/plan", json={"question": "What is RAG?"}).status_code == 200
    assert client.post("/search", json={"query": "What is RAG?"}).status_code == 200


def test_research_offloads_blocking_planning_to_a_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planning_started = threading.Event()
    release_planning = threading.Event()

    class BlockingPlanner(FakePlanner):
        def __init__(self, gemini_service: FakeGeminiService) -> None:
            super().__init__(gemini_service)
            self.planning_thread_name: str | None = None

        def plan(self, question: str) -> list[str]:
            self.planning_thread_name = threading.current_thread().name
            planning_started.set()
            assert release_planning.wait(timeout=5)
            return super().plan(question)

    planner = BlockingPlanner(FakeGeminiService())
    monkeypatch.setattr(main, "get_gemini_service", lambda request: FakeGeminiService())
    monkeypatch.setattr(main, "get_search_service", lambda request: FakeSearchService())
    monkeypatch.setattr(main, "ResearchPlanner", lambda gemini_service: planner)

    async def run_research_and_probe_the_event_loop() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as async_client:
            request_task = asyncio.create_task(
                async_client.post("/research", json={"question": "What is RAG?"})
            )
            started = await asyncio.to_thread(planning_started.wait, 5)
            assert started is True
            for _ in range(3):
                await asyncio.sleep(0)
            release_planning.set()
            return await request_task

    response = asyncio.run(run_research_and_probe_the_event_loop())

    assert response.status_code == 200
    assert planner.planning_thread_name is not None
    assert planner.planning_thread_name.startswith("asyncio")
    assert planner.question == "What is RAG?"
