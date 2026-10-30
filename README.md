# DeepTrace

DeepTrace is an autonomous AI research assistant. Given a complex research
question, it plans focused subquestions, searches the web for each one, and
returns the evidence grouped by subquestion. The long-term goal is evidence
traceability: every important claim in a final report should link back to the
source that supports it.

This repository currently contains the FastAPI backend. Report synthesis,
citations, persistence, and the frontend are planned but not implemented yet.

## Implemented features

- Question answering with Google Gemini (`POST /ask`).
- Research planning: Gemini generates 3–6 distinct subquestions
  (`POST /research/plan`).
- Web search through the Tavily Search API, with normalized results
  (`POST /search`).
- End-to-end research: plan with Gemini, search every subquestion concurrently,
  and return results grouped and ordered by subquestion (`POST /research`).
- Bounded search concurrency (maximum 3 Tavily requests in flight per research
  request) and partial-result preservation when a search fails.
- Configurable Gemini request timeout so provider hangs fail safely.
- Shared provider clients, created at startup and closed at shutdown.
- Safe error responses that never expose API keys, prompts, or raw provider
  errors.

## Planned features

- Full-page content fetching, extraction, cleaning, and source deduplication.
- Claim and evidence extraction with claim-to-source linking.
- Contradiction detection and source-quality evaluation.
- Cited research reports and evidence-grounded follow-up questions.
- LangGraph orchestration when conditional workflows justify it.
- PostgreSQL with pgvector for persistence and retrieval.
- Next.js frontend, authentication, background processing, and deployment.

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| API | FastAPI + Uvicorn |
| LLM | Google Gemini via `google-genai` |
| Web search | Tavily Search API |
| HTTP client | HTTPX (async) |
| Validation | Pydantic |
| Configuration | `python-dotenv` |
| Testing | pytest |

## Architecture

```text
backend/app/
├── main.py                  # FastAPI app, lifespan, endpoints, research workflow
├── config.py                # Environment-based configuration
└── services/
    ├── gemini.py            # Shared Gemini client, generation config, timeout
    ├── research_planner.py  # Prompting and validation of research plans
    └── web_search.py        # Async Tavily search client and result normalization
```

`GeminiService` and `TavilySearchService` are created once during application
startup and stored on `app.state`, so requests reuse them. Both are closed on
shutdown. Provider failures are converted into safe HTTP errors at the API
boundary.

The Gemini SDK is synchronous. `/ask` and `/research/plan` are synchronous
endpoints, so FastAPI runs them in its threadpool. `/research` is asynchronous,
so its planning call is offloaded with `asyncio.to_thread()` to keep the event
loop responsive.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API description and status |
| GET | `/health` | Health check |
| POST | `/ask` | Answer a question with Gemini |
| POST | `/research/plan` | Generate 3–6 research subquestions |
| POST | `/search` | Search the web with Tavily |
| POST | `/research` | Plan, search subquestions concurrently, and group results |

Interactive documentation is available at `http://127.0.0.1:8000/docs` while
the server is running.

### Request examples

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is retrieval-augmented generation?"}'

curl -X POST http://127.0.0.1:8000/research/plan \
  -H 'Content-Type: application/json' \
  -d '{"question": "How is AI changing software engineering?"}'

curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "retrieval-augmented generation evidence"}'

curl -X POST http://127.0.0.1:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"question": "How is AI changing software engineering?"}'
```

## Setup

Requirements: Python 3.13 (the virtual environment in this workspace uses it)
and API keys for Gemini and Tavily.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

Then edit `backend/.env` and add your real keys. Never commit that file.

## Environment variables

All backend variables are read from `backend/.env`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Gemini API key |
| `TAVILY_API_KEY` | Yes | — | Tavily API key |
| `GEMINI_MODEL` | No | `gemini-flash-latest` | Gemini model name or alias |
| `GEMINI_MAX_OUTPUT_TOKENS` | No | unset (provider default) | Positive output-token limit |
| `GEMINI_THINKING_BUDGET` | No | unset | Thinking budget; only sent for explicitly configured `gemini-2.5-flash*` models |
| `GEMINI_REQUEST_TIMEOUT` | No | `60` | Gemini request timeout in seconds |
| `TAVILY_SEARCH_TIMEOUT` | No | `30` | Tavily search timeout in seconds |

If either provider key is missing, the corresponding service is not created and
its endpoints return HTTP 503.

## Run the backend

From the repository root:

```bash
.venv/bin/uvicorn app.main:app --app-dir backend --reload
```

The API is then available at `http://127.0.0.1:8000`, with Swagger UI at
`http://127.0.0.1:8000/docs`.

## Run tests

From the repository root:

```bash
.venv/bin/python -m pytest
```

`pytest.ini` sets `pythonpath = backend` and `testpaths = backend/tests`, so the
suite runs from the repository root.
