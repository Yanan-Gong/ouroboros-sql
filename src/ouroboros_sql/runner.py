"""Run the pipeline on one question and serialize the full trajectory.

The trajectory — every tool call, handoff, retry, and token count — is data.
The M2 eval harness consumes these records; nothing downstream ever needs to
re-parse model output.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from agents import (
    HandoffCallItem,
    HandoffOutputItem,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    MessageOutputItem,
    ReasoningItem,
    Runner,
    RunResult,
    Session,
    ToolCallItem,
    ToolCallOutputItem,
)

from .agents.tools import QueryContext
from .agents.topology import Pipeline
from .config import settings
from .db.catalog import Catalog


@dataclass
class TrajectoryEvent:
    kind: str  # tool_call | tool_output | handoff | message | reasoning
    agent: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    question: str
    db_id: str
    status: str  # ok | guardrail_refused | max_turns | error
    final_output: str
    events: list[TrajectoryEvent] = field(default_factory=list)
    executed_sql: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    wall_seconds: float = 0.0

    @property
    def final_executed_sql(self) -> str | None:
        for attempt in reversed(self.executed_sql):
            if attempt["ok"]:
                return attempt["sql"]
        return None

    @property
    def handoff_chain(self) -> list[str]:
        chain = []
        for e in self.events:
            if e.kind == "handoff":
                chain.append(e.payload["target"])
        return chain


def _serialize_items(result: RunResult) -> list[TrajectoryEvent]:
    events: list[TrajectoryEvent] = []
    for item in result.new_items:
        agent_name = item.agent.name if item.agent else "?"
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            events.append(
                TrajectoryEvent(
                    kind="tool_call",
                    agent=agent_name,
                    payload={
                        "tool": getattr(raw, "name", "?"),
                        "arguments": getattr(raw, "arguments", None),
                    },
                )
            )
        elif isinstance(item, ToolCallOutputItem):
            output = str(item.output)
            events.append(
                TrajectoryEvent(
                    kind="tool_output",
                    agent=agent_name,
                    payload={
                        "output": output[:2000],
                        "is_error": output.startswith(
                            ("EXECUTION ERROR", "REJECTED", "TIMEOUT", "INVALID", "Error")
                        ),
                    },
                )
            )
        elif isinstance(item, HandoffOutputItem):
            events.append(
                TrajectoryEvent(
                    kind="handoff",
                    agent=agent_name,
                    payload={
                        "source": item.source_agent.name,
                        "target": item.target_agent.name,
                    },
                )
            )
        elif isinstance(item, HandoffCallItem):
            continue  # the paired HandoffOutputItem carries source/target
        elif isinstance(item, MessageOutputItem):
            events.append(TrajectoryEvent(kind="message", agent=agent_name, payload={}))
        elif isinstance(item, ReasoningItem):
            events.append(TrajectoryEvent(kind="reasoning", agent=agent_name, payload={}))
    return events


def _usage_totals(result: RunResult) -> tuple[int, int, int]:
    input_tokens = output_tokens = requests = 0
    for response in result.raw_responses:
        if response.usage:
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            requests += response.usage.requests
    return input_tokens, output_tokens, requests


async def run_one(
    question: str,
    db_id: str,
    pipeline: Pipeline,
    *,
    session: Session | None = None,
    catalog: Catalog | None = None,
    max_turns: int | None = None,
) -> RunRecord:
    catalog = catalog or Catalog(settings.databases_dir)
    context = QueryContext(db_id=db_id, db_path=catalog.path_for(db_id))
    start = time.monotonic()

    status = "ok"
    final_output = ""
    events: list[TrajectoryEvent] = []
    tokens = (0, 0, 0)

    try:
        result = await Runner.run(
            pipeline.orchestrator,
            question,
            context=context,
            session=session,
            max_turns=max_turns or settings.max_turns,
        )
        final_output = str(result.final_output)
        events = _serialize_items(result)
        tokens = _usage_totals(result)
    except InputGuardrailTripwireTriggered as e:
        status = "guardrail_refused"
        verdict = getattr(e.guardrail_result.output, "output_info", None)
        final_output = (
            "I only answer read-only analytics questions about this database. "
            f"({getattr(verdict, 'reason', 'off-topic')})"
        )
    except MaxTurnsExceeded:
        status = "max_turns"
        final_output = "The pipeline exceeded its turn budget before producing an answer."

    return RunRecord(
        question=question,
        db_id=db_id,
        status=status,
        final_output=final_output,
        events=events,
        executed_sql=context.executed_sql,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        requests=tokens[2],
        wall_seconds=time.monotonic() - start,
    )
