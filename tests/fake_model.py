"""A scripted implementation of the Agents SDK `Model` interface.

Lets the offline test suite exercise the *real* Runner orchestration — handoff
resolution, tool invocation and argument parsing, guardrail tripwires, session
memory — with zero network calls. Each `FakeModel` is constructed with a queue
of turns; every `get_response` call pops the next turn addressed to the calling
agent (identified via its system instructions).
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agents import Model, ModelResponse, Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)


@dataclass
class Turn:
    """One scripted model turn.

    Exactly one of `text`, `tool` or `handoff` should describe the action:
    - text: plain assistant message (final output for that agent)
    - tool: (tool_name, args dict) function call
    - handoff: target agent name, e.g. "SchemaLinker" (translated to the
      SDK's transfer_to_<snake_case> tool call)
    """

    text: str | None = None
    tool: tuple[str, dict[str, Any]] | None = None
    handoff: str | None = None
    expect_agent: str | None = None  # optional assertion on who is calling


def _handoff_tool_name(agent_name: str) -> str:
    # Mirrors the SDK's default: transfer_to_<name lowercased, spaces to underscores>
    return f"transfer_to_{agent_name.lower().replace(' ', '_')}"


class ScriptExhaustedError(AssertionError):
    pass


@dataclass
class FakeModel(Model):
    script: list[Turn] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

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
        self.calls.append(
            {
                "system_instructions": system_instructions,
                "input": input,
                "tools": [getattr(t, "name", "?") for t in tools],
                "handoffs": [h.tool_name for h in handoffs],
            }
        )
        if not self.script:
            raise ScriptExhaustedError(
                f"FakeModel script exhausted; called with instructions "
                f"{str(system_instructions)[:80]!r}"
            )
        turn = self.script.pop(0)
        if turn.expect_agent is not None:
            expected_fragment = turn.expect_agent
            assert expected_fragment.lower() in str(system_instructions).lower(), (
                f"Expected call from agent matching {expected_fragment!r}, "
                f"got instructions {str(system_instructions)[:120]!r}"
            )

        if turn.handoff is not None:
            output: Any = ResponseFunctionToolCall(
                type="function_call",
                call_id=f"call_h{len(self.calls)}",
                name=_handoff_tool_name(turn.handoff),
                arguments="{}",
            )
        elif turn.tool is not None:
            name, args = turn.tool
            output = ResponseFunctionToolCall(
                type="function_call",
                call_id=f"call_t{len(self.calls)}",
                name=name,
                arguments=json.dumps(args),
            )
        else:
            output = ResponseOutputMessage(
                id=f"msg_{len(self.calls)}",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(type="output_text", text=turn.text or "", annotations=[])
                ],
            )

        return ModelResponse(
            output=[output],
            usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
            response_id=None,
        )

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
        raise NotImplementedError("FakeModel does not stream")
        yield  # pragma: no cover
