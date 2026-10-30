import httpx
import pytest
from fastapi.testclient import TestClient

from app import config
from app import main
from app.main import app
from app.services import gemini
from app.services.gemini import GeminiServiceError


class FakeGeminiService:
    def ask(self, question: str) -> str:
        return f"Mocked answer for: {question}"


class TrackingGeminiService:
    def __init__(self) -> None:
        self.was_called = False

    def ask(self, question: str) -> str:
        self.was_called = True
        return "This answer should not be returned."


class FailingGeminiService:
    def ask(self, question: str) -> str:
        raise GeminiServiceError("A provider error that must stay private")


client = TestClient(app)


def test_ask_returns_a_mocked_gemini_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "get_gemini_service", lambda request: FakeGeminiService())

    response = client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Mocked answer for: What is RAG?"}


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_rejects_blank_questions(question: str) -> None:
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 422


def test_ask_rejects_an_overly_long_question_before_calling_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TrackingGeminiService()
    monkeypatch.setattr(main, "get_gemini_service", lambda request: service)

    response = client.post("/ask", json={"question": "a" * 4_001})

    assert response.status_code == 422
    assert service.was_called is False


def test_ask_reports_missing_gemini_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "get_gemini_api_key", lambda: None)

    with TestClient(app) as configured_client:
        response = configured_client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Gemini service is not configured."}


def test_ask_hides_gemini_service_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "get_gemini_service", lambda request: FailingGeminiService())

    response = client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Gemini could not answer the question. Please try again later."
    }


def test_lifespan_closes_the_shared_gemini_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSharedGeminiService:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.was_closed = False

        def close(self) -> None:
            self.was_closed = True

    fake_service = FakeSharedGeminiService("test-key")
    monkeypatch.setattr(config, "get_gemini_api_key", lambda: "test-key")
    monkeypatch.setattr(main, "GeminiService", lambda api_key: fake_service)

    with TestClient(app):
        assert app.state.gemini_service is fake_service

    assert fake_service.was_closed is True


def test_gemini_service_uses_the_expected_model_and_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        text = "A mocked answer"

    class FakeModels:
        def __init__(self) -> None:
            self.model: str | None = None
            self.contents: str | None = None
            self.config: object | None = None

        def generate_content(
            self,
            *,
            model: str,
            contents: str,
            config: object,
        ) -> FakeResponse:
            self.model = model
            self.contents = contents
            self.config = config
            return FakeResponse()

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()
            self.was_closed = False

        def close(self) -> None:
            self.was_closed = True

    fake_client = FakeClient()
    monkeypatch.setattr(
        gemini.genai, "Client", lambda *, api_key, http_options: fake_client
    )

    service = gemini.GeminiService(
        "test-key",
        model=gemini.MODEL_NAME,
        max_output_tokens=512,
    )
    answer = service.ask("What is RAG?")
    service.close()

    assert answer == "A mocked answer"
    assert fake_client.models.model == gemini.MODEL_NAME
    assert fake_client.models.contents == "What is RAG?"
    assert fake_client.models.config.max_output_tokens == 512
    assert fake_client.was_closed is True


def test_gemini_service_sends_supported_configured_generation_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        text = '{"subquestions": ["One", "Two", "Three"]}'

    class FakeModels:
        def __init__(self) -> None:
            self.model: str | None = None
            self.config: object | None = None

        def generate_content(
            self,
            *,
            model: str,
            contents: str,
            config: object,
        ) -> FakeResponse:
            self.model = model
            self.config = config
            return FakeResponse()

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

        def close(self) -> None:
            pass

    fake_client = FakeClient()
    monkeypatch.setattr(
        gemini.genai, "Client", lambda *, api_key, http_options: fake_client
    )
    service = gemini.GeminiService(
        "test-key",
        model="gemini-2.5-flash",
        max_output_tokens=512,
        thinking_budget=0,
    )

    service.generate_json("Create a research plan")

    assert fake_client.models.model == "gemini-2.5-flash"
    assert fake_client.models.config.response_mime_type == "application/json"
    assert fake_client.models.config.max_output_tokens == 512
    assert fake_client.models.config.thinking_config.thinking_budget == 0


def test_gemini_service_omits_unsupported_thinking_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        text = '{"subquestions": ["One", "Two", "Three"]}'

    class FakeModels:
        def __init__(self) -> None:
            self.config: object | None = None

        def generate_content(
            self,
            *,
            model: str,
            contents: str,
            config: object,
        ) -> FakeResponse:
            self.config = config
            return FakeResponse()

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

        def close(self) -> None:
            pass

    fake_client = FakeClient()
    monkeypatch.setattr(
        gemini.genai, "Client", lambda *, api_key, http_options: fake_client
    )
    service = gemini.GeminiService(
        "test-key",
        model="gemini-flash-latest",
        thinking_budget=0,
    )

    service.generate_json("Create a research plan")

    assert fake_client.models.config.thinking_config is None


@pytest.mark.parametrize(
    ("configured_timeout", "injected_timeout", "expected_milliseconds"),
    [
        (12.5, None, 12_500),
        (30.0, 2.5, 2_500),
    ],
)
def test_gemini_service_applies_a_bounded_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
    configured_timeout: float,
    injected_timeout: float | None,
    expected_milliseconds: int,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self) -> None:
            self.models = None

        def close(self) -> None:
            pass

    def create_client(*, api_key: str, http_options: object) -> FakeClient:
        captured["api_key"] = api_key
        captured["http_options"] = http_options
        return FakeClient()

    monkeypatch.setattr(gemini.genai, "Client", create_client)
    monkeypatch.setattr(
        config, "get_gemini_request_timeout", lambda: configured_timeout
    )

    gemini.GeminiService("test-key", request_timeout=injected_timeout)

    assert captured["http_options"].timeout == expected_milliseconds


def test_gemini_service_wraps_timeout_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModels:
        def generate_content(self, **kwargs: object) -> object:
            raise httpx.ReadTimeout("timed out")

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

        def close(self) -> None:
            pass

    fake_client = FakeClient()
    monkeypatch.setattr(
        gemini.genai,
        "Client",
        lambda *, api_key, http_options: fake_client,
    )
    service = gemini.GeminiService("test-key")

    with pytest.raises(GeminiServiceError) as ask_error:
        service.ask("What is RAG?")

    with pytest.raises(GeminiServiceError) as plan_error:
        service.generate_json("Create a research plan")

    assert str(ask_error.value) == "Gemini request failed"
    assert isinstance(ask_error.value.__cause__, httpx.ReadTimeout)
    assert str(plan_error.value) == "Gemini request failed"
    assert isinstance(plan_error.value.__cause__, httpx.ReadTimeout)
