"""Model routing.

Worker agents run on the default OpenAI-compatible endpoint. Deployments named
``claude-*`` (e.g. the optimizer on Azure AI Foundry's Anthropic endpoint) are
wrapped in a minimal implementation of the Agents SDK ``Model`` interface.

The Anthropic wrapper intentionally supports only what the optimizer-side
agents need — plain text and structured (JSON) output. Agents with tools or
handoffs stay on the primary endpoint; asking this wrapper to run them raises
immediately rather than degrading silently.
"""

import json
import os
import re
from collections.abc import AsyncIterator
from typing import Any

from agents import Model, ModelResponse, Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

ANTHROPIC_MAX_TOKENS = 8192

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def anthropic_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and bool(os.environ.get("ANTHROPIC_BASE_URL"))


def resolve_model(name: str) -> "str | Model":
    """claude-* deployments route to the Anthropic endpoint when configured."""
    if name.startswith("claude") and anthropic_configured():
        return AnthropicModel(name)
    return name


def extract_text(blocks: list[Any]) -> str:
    text = "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text")
    return _FENCE_RE.sub("", text).strip()


def flatten_input(input: Any) -> str:
    """Our optimizer-side agents pass plain strings; tolerate simple item lists."""
    if isinstance(input, str):
        return input
    parts = []
    for item in input:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


class AnthropicModel(Model):
    def __init__(self, deployment: str):
        self.deployment = deployment

    def _client(self):
        from anthropic import AsyncAnthropicFoundry

        return AsyncAnthropicFoundry(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=os.environ["ANTHROPIC_BASE_URL"],
        )

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        *,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    ) -> ModelResponse:
        if tools or handoffs:
            raise NotImplementedError(
                "AnthropicModel backs text/structured-output agents only; "
                "tool-using agents stay on the primary endpoint"
            )
        system = system_instructions or ""
        if output_schema is not None and not output_schema.is_plain_text():
            schema = json.dumps(output_schema.json_schema())
            system += (
                "\n\nRespond with ONLY a single JSON object matching this schema "
                f"(no prose, no markdown fences):\n{schema}"
            )

        message = await self._client().messages.create(
            model=self.deployment,
            system=system,
            messages=[{"role": "user", "content": flatten_input(input)}],
            max_tokens=ANTHROPIC_MAX_TOKENS,
        )
        text = extract_text(message.content)
        usage = Usage(
            requests=1,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            total_tokens=message.usage.input_tokens + message.usage.output_tokens,
        )
        output = ResponseOutputMessage(
            id=message.id,
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )
        return ModelResponse(output=[output], usage=usage, response_id=None)

    async def stream_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        *,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    ) -> AsyncIterator[Any]:
        raise NotImplementedError("AnthropicModel does not stream")
        yield  # pragma: no cover
