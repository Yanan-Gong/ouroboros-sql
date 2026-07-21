"""LLM-as-judge scoring of trajectories — anchored, never authoritative.

The judge scores *process* quality on a rubric. It can never overturn the
deterministic execution match: a record whose SQL result does not match gold
has its overall judge score capped at 0.5. Judge model and version are
recorded with every run; judge-vs-exec agreement is reported so readers can
calibrate how much to trust the process scores.
"""

import json

from agents import Agent, Runner
from pydantic import BaseModel, Field

from .schema import EvalRunRecord


class JudgeScores(BaseModel):
    schema_exploration: float = Field(ge=0, le=1)
    sql_reasoning: float = Field(ge=0, le=1)
    error_recovery: float = Field(ge=0, le=1)
    answer_faithfulness: float = Field(ge=0, le=1)
    rationale: str


JUDGE_INSTRUCTIONS = """\
You are evaluating the internal trajectory of a multi-agent Text-to-SQL system.
You will see: the user question, the database, the sequence of agent handoffs
and tool calls, the SQL attempts, and the final answer. You are also told
whether the final SQL's result matched the gold query's result (EXEC_MATCH).

Score four dimensions, each 0.0-1.0:
- schema_exploration: did the pipeline inspect the right tables before writing
  SQL (not too little, not blindly everything)?
- sql_reasoning: is the final SQL a sound approach for the question (joins,
  filters, aggregation, ordering)?
- error_recovery: if attempts failed, did the retry address the actual error?
  Give 1.0 when there were no errors to recover from.
- answer_faithfulness: does the final natural-language answer state only what
  the executed result supports?

Be strict: reserve scores above 0.8 for genuinely clean work. EXEC_MATCH=false
means something went wrong somewhere — find it and reflect it in the relevant
dimension. Keep rationale to two sentences.
"""


def render_trajectory(record: EvalRunRecord, max_chars: int = 6000) -> str:
    lines = [
        f"QUESTION: {record.question}",
        f"DATABASE: {record.db_id}",
        f"STATUS: {record.status}",
        f"EXEC_MATCH: {record.execution_match}",
        "TRAJECTORY:",
    ]
    for e in record.events:
        kind = e.get("kind")
        payload = e.get("payload", {})
        if kind == "handoff":
            lines.append(f"  handoff {payload.get('source')} -> {payload.get('target')}")
        elif kind == "tool_call":
            args = str(payload.get("arguments", ""))[:300]
            lines.append(f"  [{e.get('agent')}] {payload.get('tool')}({args})")
        elif kind == "tool_output":
            lines.append(f"    -> {str(payload.get('output', ''))[:300]}")
    lines.append("SQL ATTEMPTS:")
    for a in record.executed_sql:
        flag = "ok" if a.get("ok") else f"ERROR: {a.get('error')}"
        lines.append(f"  [{flag}] {a.get('sql', '')[:400]}")
    lines.append(f"FINAL ANSWER: {record.final_output[:800]}")
    text = "\n".join(lines)
    return text[:max_chars]


def build_judge(model: str) -> Agent:
    return Agent(
        name="TrajectoryJudge",
        instructions=JUDGE_INSTRUCTIONS,
        output_type=JudgeScores,
        model=model,
    )


def overall_score(scores: JudgeScores, execution_match: bool | None) -> float:
    mean = (
        scores.schema_exploration
        + scores.sql_reasoning
        + scores.error_recovery
        + scores.answer_faithfulness
    ) / 4
    # Anchor: process score cannot certify a wrong outcome.
    if execution_match is False:
        return min(mean, 0.5)
    return mean


async def judge_record(record: EvalRunRecord, judge: Agent, judge_model: str) -> dict:
    result = await Runner.run(judge, render_trajectory(record))
    scores = result.final_output_as(JudgeScores)
    return {
        "overall": overall_score(scores, record.execution_match),
        **json.loads(scores.model_dump_json()),
        "judge_model": judge_model,
    }
