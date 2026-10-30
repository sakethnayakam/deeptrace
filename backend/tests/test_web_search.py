import asyncio
import json

import httpx
import pytest

from app import config
from app.services.web_search import (
    MAX_RESULTS,
    SEARCH_API_URL,
    SEARCH_DEPTH,
    TavilySearchService,
    WebSearchServiceError,
)


def make_service(handler: httpx.MockTransport) -> TavilySearchService:
    return TavilySearchService("test-key", client=httpx.AsyncClient(transport=handler))


def test_search_normalizes_tavily_results_and_uses_expected_request() -> None:
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json={"results": [{"title": " Example title ", "url": "https://example.com/article ", "content": " Example content ", "score": 0.91, "published_date": " 2026-10-30 "}, {"title": "Another result", "url": "https://example.org", "content": "Another snippet"}]})

    async def run_test() -> list[dict[str, str | float]]:
        service = make_service(httpx.MockTransport(handler))
        try:
            return await service.search("deep research")
        finally:
            await service.close()

    results = asyncio.run(run_test())
    assert results == [{"title": "Example title", "url": "https://example.com/article", "snippet": "Example content", "score": 0.91, "published_date": "2026-10-30"}, {"title": "Another result", "url": "https://example.org", "snippet": "Another snippet"}]
    assert captured_request is not None
    assert captured_request.url == httpx.URL(SEARCH_API_URL)
    assert json.loads(captured_request.content) == {"api_key": "test-key", "query": "deep research", "search_depth": SEARCH_DEPTH, "max_results": MAX_RESULTS}


@pytest.mark.parametrize("error", [httpx.ReadTimeout("timed out"), httpx.ConnectError("connection failed")])
def test_search_hides_timeout_and_network_errors(error: httpx.RequestError) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    async def run_test() -> WebSearchServiceError:
        service = make_service(httpx.MockTransport(handler))
        try:
            with pytest.raises(WebSearchServiceError) as exception_info:
                await service.search("deep research")
            return exception_info.value
        finally:
            await service.close()

    assert "test-key" not in str(asyncio.run(run_test()))


def test_search_hides_unsuccessful_http_response() -> None:
    async def run_test() -> WebSearchServiceError:
        service = make_service(httpx.MockTransport(lambda request: httpx.Response(401, text="invalid key")))
        try:
            with pytest.raises(WebSearchServiceError) as exception_info:
                await service.search("deep research")
            return exception_info.value
        finally:
            await service.close()

    assert str(asyncio.run(run_test())) == "Web search provider returned an unsuccessful response."


def test_search_rejects_malformed_json() -> None:
    async def run_test() -> WebSearchServiceError:
        service = make_service(httpx.MockTransport(lambda request: httpx.Response(200, content=b"not json")))
        try:
            with pytest.raises(WebSearchServiceError) as exception_info:
                await service.search("deep research")
            return exception_info.value
        finally:
            await service.close()

    assert str(asyncio.run(run_test())) == "Web search provider returned invalid data."


@pytest.mark.parametrize("payload", [{}, {"results": "not a list"}, {"results": ["not an object"]}, {"results": [{"title": "Title", "url": "https://example.com"}]}, {"results": [{"title": "Title", "url": "https://example.com", "content": "Snippet", "score": "high"}]}])
def test_search_rejects_invalid_result_structures(payload: object) -> None:
    async def run_test() -> WebSearchServiceError:
        service = make_service(httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
        try:
            with pytest.raises(WebSearchServiceError) as exception_info:
                await service.search("deep research")
            return exception_info.value
        finally:
            await service.close()

    assert str(asyncio.run(run_test())) == "Web search provider returned invalid data."


def test_search_uses_configured_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "get_tavily_search_timeout", lambda: 12.5)
    service = TavilySearchService("test-key")
    try:
        assert service.client.timeout.read == 12.5
    finally:
        asyncio.run(service.close())


def test_close_closes_the_shared_async_client() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    service = TavilySearchService("test-key", client=client)
    asyncio.run(service.close())
    assert client.is_closed is True
