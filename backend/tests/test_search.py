import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.main import app
from app.services.web_search import WebSearchServiceError


class FakeSearchService:
    def __init__(self) -> None:
        self.query: str | None = None

    async def search(self, query: str) -> list[dict[str, str | float]]:
        self.query = query
        return [{"title": "Example result", "url": "https://example.com", "snippet": "An example result.", "score": 0.92, "published_date": "2026-10-30"}]


class TrackingSearchService:
    def __init__(self) -> None:
        self.was_called = False

    async def search(self, query: str) -> list[dict[str, str | float]]:
        self.was_called = True
        return []


class FailingSearchService:
    async def search(self, query: str) -> list[dict[str, str | float]]:
        raise WebSearchServiceError("A provider error that must stay private")


client = TestClient(app)


def test_search_returns_normalized_tavily_results(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FakeSearchService()
    monkeypatch.setattr(main, "get_search_service", lambda request: service)
    response = client.post("/search", json={"query": "  DeepTrace  "})
    assert response.status_code == 200
    assert response.json() == {"query": "DeepTrace", "results": [{"title": "Example result", "url": "https://example.com", "snippet": "An example result.", "score": 0.92, "published_date": "2026-10-30"}]}
    assert service.query == "DeepTrace"


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_blank_queries(query: str) -> None:
    assert client.post("/search", json={"query": query}).status_code == 422


def test_search_rejects_an_overly_long_query_before_calling_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    service = TrackingSearchService()
    monkeypatch.setattr(main, "get_search_service", lambda request: service)
    assert client.post("/search", json={"query": "a" * 501}).status_code == 422
    assert service.was_called is False


def test_search_reports_missing_tavily_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "get_gemini_api_key", lambda: None)
    monkeypatch.setattr(config, "get_tavily_api_key", lambda: None)
    with TestClient(app) as configured_client:
        response = configured_client.post("/search", json={"query": "DeepTrace"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Web search service is not configured."}


def test_search_hides_tavily_service_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "get_search_service", lambda request: FailingSearchService())
    response = client.post("/search", json={"query": "DeepTrace"})
    assert response.status_code == 502
    assert response.json() == {"detail": "Web search could not be completed. Please try again later."}


def test_lifespan_creates_and_closes_the_shared_search_service(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSharedSearchService:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.was_closed = False

        async def close(self) -> None:
            self.was_closed = True

    fake_service = FakeSharedSearchService("test-key")
    monkeypatch.setattr(config, "get_gemini_api_key", lambda: None)
    monkeypatch.setattr(config, "get_tavily_api_key", lambda: "test-key")
    monkeypatch.setattr(main, "TavilySearchService", lambda api_key: fake_service)
    with TestClient(app):
        assert app.state.search_service is fake_service
    assert fake_service.was_closed is True
