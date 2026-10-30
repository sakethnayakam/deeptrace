"""Asynchronous wrapper around Tavily's Search API."""

from collections.abc import Mapping

import httpx

from app import config


SEARCH_API_URL = "https://api.tavily.com/search"
SEARCH_DEPTH = "basic"
MAX_RESULTS = 5


class WebSearchServiceError(Exception):
    """Raised when Tavily cannot provide usable search results."""


class TavilySearchService:
    """Search the web through Tavily without exposing provider errors."""

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key or config.get_tavily_api_key()
        if not self.api_key:
            raise WebSearchServiceError("Tavily API key is not configured.")

        self.client = client or httpx.AsyncClient(
            timeout=timeout if timeout is not None else config.get_tavily_search_timeout()
        )

    async def search(self, query: str) -> list[dict[str, str | float]]:
        """Return normalized Tavily results for a search query."""
        try:
            response = await self.client.post(
                SEARCH_API_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": SEARCH_DEPTH,
                    "max_results": MAX_RESULTS,
                },
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise WebSearchServiceError("Web search request timed out.") from error
        except httpx.HTTPStatusError as error:
            raise WebSearchServiceError(
                "Web search provider returned an unsuccessful response."
            ) from error
        except httpx.RequestError as error:
            raise WebSearchServiceError("Web search request failed.") from error

        try:
            payload = response.json()
        except ValueError as error:
            raise WebSearchServiceError("Web search provider returned invalid data.") from error

        return self._normalize_results(payload)

    async def close(self) -> None:
        """Close the HTTP client and release its network resources."""
        await self.client.aclose()

    @staticmethod
    def _normalize_results(payload: object) -> list[dict[str, str | float]]:
        if not isinstance(payload, Mapping):
            raise WebSearchServiceError("Web search provider returned invalid data.")

        results = payload.get("results")
        if not isinstance(results, list):
            raise WebSearchServiceError("Web search provider returned invalid data.")

        normalized_results: list[dict[str, str | float]] = []
        for result in results:
            normalized_results.append(TavilySearchService._normalize_result(result))

        return normalized_results

    @staticmethod
    def _normalize_result(result: object) -> dict[str, str | float]:
        if not isinstance(result, Mapping):
            raise WebSearchServiceError("Web search provider returned invalid data.")

        title = result.get("title")
        url = result.get("url")
        content = result.get("content")
        if not all(isinstance(value, str) for value in (title, url, content)):
            raise WebSearchServiceError("Web search provider returned invalid data.")

        normalized_result: dict[str, str | float] = {
            "title": title.strip(),
            "url": url.strip(),
            "snippet": content.strip(),
        }
        if not normalized_result["title"] or not normalized_result["url"]:
            raise WebSearchServiceError("Web search provider returned invalid data.")

        score = result.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, int | float):
                raise WebSearchServiceError("Web search provider returned invalid data.")
            normalized_result["score"] = float(score)

        published_date = result.get("published_date")
        if published_date is not None:
            if not isinstance(published_date, str):
                raise WebSearchServiceError("Web search provider returned invalid data.")
            normalized_result["published_date"] = published_date.strip()

        return normalized_result
