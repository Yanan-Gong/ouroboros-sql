"""Anthropic model routing and text extraction (no network)."""

from types import SimpleNamespace

from ouroboros_sql.models import AnthropicModel, extract_text, flatten_input, resolve_model


def test_resolve_routes_only_claude_with_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert resolve_model("claude-opus-4-8") == "claude-opus-4-8"  # unconfigured -> passthrough
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example/anthropic")
    resolved = resolve_model("claude-opus-4-8")
    assert isinstance(resolved, AnthropicModel)
    assert resolve_model("gpt-5-mini") == "gpt-5-mini"


def test_extract_text_strips_fences():
    blocks = [
        SimpleNamespace(type="text", text='```json\n{"a": 1}\n```'),
        SimpleNamespace(type="thinking", text="ignored"),
    ]
    assert extract_text(blocks) == '{"a": 1}'


def test_flatten_input():
    assert flatten_input("plain") == "plain"
    items = [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
    assert flatten_input(items) == "one\ntwo"


async def test_tools_rejected_loudly(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example/anthropic")
    import pytest

    model = AnthropicModel("claude-opus-4-8")
    with pytest.raises(NotImplementedError):
        await model.get_response(
            None, "x", None, tools=[object()], output_schema=None, handoffs=[], tracing=None
        )
