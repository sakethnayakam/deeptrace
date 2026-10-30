import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app import config
from app.services.gemini import GeminiService, GeminiServiceError
from app.services.research_planner import ResearchPlanner, ResearchPlannerError
from app.services.web_search import TavilySearchService, WebSearchServiceError


MAX_QUESTION_LENGTH = 4_000
MAX_SEARCH_QUERY_LENGTH = 500
MAX_CONCURRENT_RESEARCH_SEARCHES = 3


# Use Uvicorn's existing error-log handler so application records are visible
# in the server output without registering a duplicate handler.
logger = logging.getLogger("uvicorn.error.deeptrace")
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create configured provider clients at startup and close them at shutdown."""
    gemini_api_key = config.get_gemini_api_key()
    gemini_service = GeminiService(gemini_api_key) if gemini_api_key else None
    application.state.gemini_service = gemini_service

    tavily_api_key = config.get_tavily_api_key()
    search_service = TavilySearchService(tavily_api_key) if tavily_api_key else None
    application.state.search_service = search_service

    try:
        yield
    finally:
        if search_service:
            await search_service.close()
        if gemini_service:
            gemini_service.close()


app = FastAPI(title="DeepTrace API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_LENGTH)

    @field_validator("question", mode="before")
    @classmethod
    def question_must_not_be_blank(cls, question: object) -> object:
        if not isinstance(question, str):
            return question
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        return question


class AskResponse(BaseModel):
    answer: str


class ResearchPlanRequest(AskRequest):
    """A research question using the same validation as /ask."""


class ResearchPlanResponse(BaseModel):
    question: str
    subquestions: list[str]


class ResearchRequest(ResearchPlanRequest):
    """A research execution request using the same validation as /research/plan."""


class SearchRequest(BaseModel):
    query: str = Field(max_length=MAX_SEARCH_QUERY_LENGTH)

    @field_validator("query", mode="before")
    @classmethod
    def query_must_not_be_blank(cls, query: object) -> object:
        if not isinstance(query, str):
            return query
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")
        return query


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    score: float | None = None
    published_date: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class ResearchSearchResultGroup(BaseModel):
    subquestion: str
    results: list[SearchResult]
    error: str | None = None


class ResearchResponse(BaseModel):
    question: str
    subquestions: list[str]
    search_results: list[ResearchSearchResultGroup]


def get_gemini_service(request: Request) -> GeminiService:
    """Return the shared Gemini service when the app is configured."""
    gemini_service = getattr(request.app.state, "gemini_service", None)
    if gemini_service is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini service is not configured.",
        )
    return gemini_service


def get_search_service(request: Request) -> TavilySearchService:
    """Return the shared Tavily service when the app is configured."""
    search_service = getattr(request.app.state, "search_service", None)
    if search_service is None:
        raise HTTPException(
            status_code=503,
            detail="Web search service is not configured.",
        )
    return search_service


@app.get("/")
def read_root() -> dict[str, str]:
    """Describe the API's current, early implementation."""
    return {
        "description": "DeepTrace autonomous AI research assistant API",
        "status": "initial backend scaffold",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report that the API is running."""
    return {
        "status": "ok",
        "service": "deeptrace-api",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    http_request: Request,
) -> AskResponse:
    """Answer one non-empty question using Gemini."""
    gemini_service = get_gemini_service(http_request)

    try:
        answer = gemini_service.ask(request.question)
    except GeminiServiceError:
        raise HTTPException(
            status_code=502,
            detail="Gemini could not answer the question. Please try again later.",
        ) from None

    return AskResponse(answer=answer)


@app.post("/search", response_model=SearchResponse, response_model_exclude_none=True)
async def search_web(
    request: SearchRequest,
    http_request: Request,
) -> SearchResponse:
    """Search the web for one non-empty query using Tavily."""
    search_service = get_search_service(http_request)

    try:
        results = await search_service.search(request.query)
    except WebSearchServiceError:
        raise HTTPException(
            status_code=502,
            detail="Web search could not be completed. Please try again later.",
        ) from None

    return SearchResponse(query=request.query, results=results)


@app.post("/research/plan", response_model=ResearchPlanResponse)
def create_research_plan(
    request: ResearchPlanRequest,
    http_request: Request,
) -> ResearchPlanResponse:
    """Create a structured set of subquestions for a research question."""
    gemini_service = get_gemini_service(http_request)
    planner = ResearchPlanner(gemini_service)

    try:
        subquestions = planner.plan(request.question)
    except ResearchPlannerError:
        raise HTTPException(
            status_code=502,
            detail="Gemini could not create a research plan. Please try again later.",
        ) from None

    return ResearchPlanResponse(
        question=request.question,
        subquestions=subquestions,
    )


@app.post("/research", response_model=ResearchResponse, response_model_exclude_none=True)
async def execute_research(
    request: ResearchRequest,
    http_request: Request,
) -> ResearchResponse:
    """Plan research with Gemini, then search subquestions with bounded concurrency."""
    request_started_at = time.perf_counter()
    gemini_service = get_gemini_service(http_request)
    search_service = get_search_service(http_request)
    planner = ResearchPlanner(gemini_service)

    planning_started_at = time.perf_counter()
    try:
        subquestions = await asyncio.to_thread(planner.plan, request.question)
    except ResearchPlannerError:
        planning_duration = time.perf_counter() - planning_started_at
        total_duration = time.perf_counter() - request_started_at
        logger.info(
            "research_failed stage=planning total_duration_seconds=%.3f "
            "planning_duration_seconds=%.3f",
            total_duration,
            planning_duration,
        )
        raise HTTPException(
            status_code=502,
            detail="Research could not be completed. Please try again later.",
        ) from None

    planning_duration = time.perf_counter() - planning_started_at

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_RESEARCH_SEARCHES)

    async def search_subquestion(
        subquestion_index: int,
        subquestion: str,
    ) -> ResearchSearchResultGroup:
        async with semaphore:
            search_started_at = time.perf_counter()
            try:
                results = await search_service.search(subquestion)
            except WebSearchServiceError:
                search_duration = time.perf_counter() - search_started_at
                logger.info(
                    "research_search_completed subquestion_index=%d status=failed "
                    "duration_seconds=%.3f",
                    subquestion_index,
                    search_duration,
                )
                return ResearchSearchResultGroup(
                    subquestion=subquestion,
                    results=[],
                    error="Web search could not be completed for this subquestion.",
                )

        search_duration = time.perf_counter() - search_started_at
        logger.info(
            "research_search_completed subquestion_index=%d status=success "
            "duration_seconds=%.3f",
            subquestion_index,
            search_duration,
        )
        return ResearchSearchResultGroup(subquestion=subquestion, results=results)

    tavily_searches_started_at = time.perf_counter()
    search_results = await asyncio.gather(
        *(
            search_subquestion(subquestion_index, subquestion)
            for subquestion_index, subquestion in enumerate(subquestions)
        )
    )
    tavily_search_duration = time.perf_counter() - tavily_searches_started_at
    successful_searches = sum(group.error is None for group in search_results)
    failed_searches = len(search_results) - successful_searches
    total_duration = time.perf_counter() - request_started_at

    logger.info(
        "research_completed total_duration_seconds=%.3f planning_duration_seconds=%.3f "
        "tavily_search_duration_seconds=%.3f successful_searches=%d "
        "failed_searches=%d",
        total_duration,
        planning_duration,
        tavily_search_duration,
        successful_searches,
        failed_searches,
    )

    if all(group.error is not None for group in search_results):
        raise HTTPException(
            status_code=502,
            detail="Research web searches could not be completed. Please try again later.",
        )

    return ResearchResponse(
        question=request.question,
        subquestions=subquestions,
        search_results=search_results,
    )
