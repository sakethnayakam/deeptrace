import pytest

from app import config


def test_gemini_configuration_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)
    monkeypatch.delenv("GEMINI_REQUEST_TIMEOUT", raising=False)

    assert config.get_gemini_model() == "gemini-flash-latest"
    assert config.get_gemini_max_output_tokens() is None
    assert config.get_gemini_thinking_budget() is None
    assert config.get_gemini_request_timeout() == 60.0


def test_gemini_configuration_returns_valid_configured_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_MODEL", " gemini-2.5-flash ")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "512")
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "0")
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT", " 12.5 ")

    assert config.get_gemini_model() == "gemini-2.5-flash"
    assert config.get_gemini_max_output_tokens() == 512
    assert config.get_gemini_thinking_budget() == 0
    assert config.get_gemini_request_timeout() == 12.5


@pytest.mark.parametrize(
    ("setting", "value", "getter"),
    [
        ("GEMINI_MAX_OUTPUT_TOKENS", "0", config.get_gemini_max_output_tokens),
        ("GEMINI_MAX_OUTPUT_TOKENS", "not-a-number", config.get_gemini_max_output_tokens),
        ("GEMINI_THINKING_BUDGET", "-2", config.get_gemini_thinking_budget),
        ("GEMINI_THINKING_BUDGET", "not-a-number", config.get_gemini_thinking_budget),
    ],
)
def test_invalid_gemini_numeric_configuration_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str,
    getter: object,
) -> None:
    monkeypatch.setenv(setting, value)

    assert callable(getter)
    assert getter() is None


@pytest.mark.parametrize("value", ["", "0", "-5", "not-a-number"])
def test_invalid_gemini_request_timeout_uses_default(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT", value)

    assert config.get_gemini_request_timeout() == config.DEFAULT_GEMINI_REQUEST_TIMEOUT
