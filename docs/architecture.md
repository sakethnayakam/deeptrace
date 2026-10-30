# DeepTrace architecture

## Implemented backend

DeepTrace is a FastAPI backend in `backend/app/main.py`. It manages one shared
client for each configured provider through its application lifespan:

- `GeminiService` is created when `GEMINI_API_KEY` is configured and closed at
  shutdown. It provides text answers and JSON generation for research planning.
  The model is configurable with `GEMINI_MODEL` and defaults to
  `gemini-flash-latest`. Optional output-token and thinking-budget settings are
  applied only when configured.
- `TavilySearchService` is created when `TAVILY_API_KEY` is configured and
  closed at shutdown. It uses a reusable `httpx.AsyncClient` to call Tavily's
  Search API with basic search depth and at most five results per query. Its
  timeout is configurable with `TAVILY_SEARCH_TIMEOUT` and defaults to 30
  seconds.

Provider clients are stored on `app.state`, so requests reuse them instead of
creating a client per request. Provider errors are converted to safe API
responses and do not expose API keys, prompts, or raw upstream errors.

## Gemini request timeout

Gemini calls are bounded with `GEMINI_REQUEST_TIMEOUT`, expressed in seconds and
defaulting to 60. `GeminiService` applies it through the SDK's supported
mechanism, `types.HttpOptions(timeout=<milliseconds>)`, when constructing the
shared client. The SDK converts the value to seconds for each HTTPX request.

When a request exceeds the timeout, the SDK raises an HTTPX timeout exception.
`GeminiService` wraps every provider exception in `GeminiServiceError`; the API
translates that into the existing generic 502 response. This keeps a stalled
provider from leaving requests pending indefinitely without exposing internal
details.

The timeout bounds a single Gemini request attempt. The SDK's default retry
configuration used here performs one attempt, so no hidden retry multiplier
extends the bound.

## Research workflow

`POST /research` executes the current end-to-end workflow:

1. Validate and normalize the research question.
2. Plan with Gemini: `ResearchPlanner.plan()` prompts Gemini for 3–6 distinct
   subquestions, then validates the JSON for count, blank entries, duplicates,
   and length.
3. Because `/research` is asynchronous and the Gemini SDK is synchronous, the
   planner call is offloaded with `asyncio.to_thread()`. This keeps the event
   loop free to serve other requests during the Gemini round-trip. `/ask` and
   `/research/plan` remain synchronous endpoints, so FastAPI runs them in its
   threadpool instead.
4. Search with Tavily: every subquestion is searched concurrently using
   `asyncio.gather()`, with an `asyncio.Semaphore(3)` limiting the research
   request to three in-flight Tavily searches.
5. Group and order results: `asyncio.gather()` preserves the original
   subquestion order, so each result group stays attached to the subquestion
   that produced it.
6. Return partial results on failure: a failed search produces an empty
   `results` list plus a safe `error` field for that group while successful
   groups still return their results. If every search fails, the endpoint
   returns a safe HTTP 502.

Performance instrumentation uses `time.perf_counter()` and the
`uvicorn.error.deeptrace` logger. Logs cover total research duration, Gemini
planning duration, aggregate and per-subquestion Tavily duration, search status,
and success/failure counts. They do not include API keys, prompts, search result
content, or raw provider errors.

## API endpoints

- `GET /` returns the API description and status.
- `GET /health` reports whether the API process is running.
- `POST /ask` accepts a non-empty question of up to 4,000 characters and
  returns a Gemini answer.
- `POST /research/plan` accepts the same question format and returns three to
  six Gemini-generated research subquestions.
- `POST /search` accepts a non-empty, trimmed query of up to 500 characters
  and returns normalized Tavily web-search results.
- `POST /research` accepts the same question format as `/research/plan`.
  It runs the workflow described above.

Tavily results include `title`, `url`, and `snippet`, with `score` and
`published_date` included when Tavily supplies them.

## Error handling

- Missing provider configuration returns HTTP 503 for the affected endpoints.
- Invalid requests return HTTP 422 before any provider call.
- Gemini and Tavily failures return generic HTTP 502 messages.
- Individual research search failures are preserved as per-group errors unless
  all searches fail.
- No API keys, prompts, or raw provider error messages are returned to clients.

## Current boundaries

Research execution uses bounded concurrency to reduce latency while keeping
provider usage predictable. The system does not yet fetch full page content,
extract claims, detect contradictions, synthesize cited reports, or answer
follow-up questions from collected evidence.

LangGraph orchestration, persistence, vector retrieval, authentication,
background processing, and a frontend are not implemented. They are planned
only when the workflow complexity justifies them.
