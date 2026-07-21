"""End-to-end pipeline tests against the real Runner with a scripted FakeModel."""

import json
from pathlib import Path

from fake_model import FakeModel, Turn
from ouroboros_sql.agents.topology import build_pipeline
from ouroboros_sql.db.catalog import Catalog
from ouroboros_sql.runner import run_one


def happy_path_script() -> list[Turn]:
    return [
        Turn(handoff="SchemaLinker", expect_agent="orchestrator"),
        Turn(tool=("list_tables", {}), expect_agent="SchemaLinker"),
        Turn(tool=("describe_table", {"table": "schools"}), expect_agent="SchemaLinker"),
        Turn(handoff="SQLWriter", expect_agent="SchemaLinker"),
        Turn(handoff="Validator", expect_agent="SQLWriter"),
        Turn(
            tool=("execute_sql", {"sql": "SELECT name FROM schools ORDER BY enrollment DESC"}),
            expect_agent="Validator",
        ),
        Turn(handoff="Summarizer", expect_agent="Validator"),
        Turn(
            text="Bay High has the largest enrollment (1,200 students).", expect_agent="Summarizer"
        ),
    ]


async def test_happy_path_full_chain(tiny_db: Path):
    model = FakeModel(script=happy_path_script())
    pipeline = build_pipeline(model=model, with_relevance_guardrail=False)
    record = await run_one(
        "Which school has the largest enrollment?",
        "schools",
        pipeline,
        catalog=Catalog(tiny_db.parent),
    )

    assert record.status == "ok"
    assert "Bay High" in record.final_output
    assert record.handoff_chain == ["SchemaLinker", "SQLWriter", "Validator", "Summarizer"]
    assert record.final_executed_sql == "SELECT name FROM schools ORDER BY enrollment DESC"
    assert record.requests == len(happy_path_script())
    assert record.input_tokens > 0

    tool_calls = [e for e in record.events if e.kind == "tool_call"]
    assert [e.payload["tool"] for e in tool_calls] == [
        "list_tables",
        "describe_table",
        "execute_sql",
    ]
    # Tools ran against the real db: the schema summary reached the model.
    outputs = [e for e in record.events if e.kind == "tool_output"]
    assert "schools(school_id*" in outputs[0].payload["output"]


async def test_retry_loop_after_execution_error(tiny_db: Path):
    script = [
        Turn(handoff="SchemaLinker"),
        Turn(handoff="SQLWriter"),
        Turn(handoff="Validator"),
        # First attempt references a bad column -> execution error
        Turn(tool=("execute_sql", {"sql": "SELECT wrong_column FROM schools"})),
        Turn(handoff="SQLWriter"),  # retry loop back
        Turn(handoff="Validator"),
        Turn(tool=("execute_sql", {"sql": "SELECT name FROM schools"})),
        Turn(handoff="Summarizer"),
        Turn(text="There are three schools."),
    ]
    model = FakeModel(script=script)
    pipeline = build_pipeline(model=model, with_relevance_guardrail=False)
    record = await run_one(
        "List the schools.", "schools", pipeline, catalog=Catalog(tiny_db.parent)
    )

    assert record.status == "ok"
    assert [a["ok"] for a in record.executed_sql] == [False, True]
    assert record.final_executed_sql == "SELECT name FROM schools"
    # The retry ping-pong is visible in the handoff chain.
    assert record.handoff_chain == [
        "SchemaLinker",
        "SQLWriter",
        "Validator",
        "SQLWriter",
        "Validator",
        "Summarizer",
    ]
    error_outputs = [e for e in record.events if e.kind == "tool_output" and e.payload["is_error"]]
    assert len(error_outputs) == 1


async def test_write_attempt_is_rejected_in_code(tiny_db: Path):
    script = [
        Turn(handoff="SchemaLinker"),
        Turn(handoff="SQLWriter"),
        Turn(handoff="Validator"),
        Turn(tool=("execute_sql", {"sql": "DROP TABLE schools"})),
        Turn(text="I could not do that."),
    ]
    model = FakeModel(script=script)
    pipeline = build_pipeline(model=model, with_relevance_guardrail=False)
    record = await run_one(
        "Please drop the schools table.", "schools", pipeline, catalog=Catalog(tiny_db.parent)
    )

    rejected = [a for a in record.executed_sql if not a["ok"]]
    assert len(rejected) == 1
    assert "unsafe" in rejected[0]["error"]
    # And the table is still there.
    from ouroboros_sql.db.introspect import list_tables

    assert "schools" in list_tables(tiny_db)


async def test_guardrail_refuses_offtopic(tiny_db: Path):
    guardrail_model = FakeModel(
        script=[
            Turn(
                text=json.dumps(
                    {"is_analytics_question": False, "reason": "Not a database question."}
                )
            )
        ]
    )
    # The orchestrator may or may not get called before the tripwire cancels
    # it; give it a throwaway turn either way.
    agent_model = FakeModel(script=[Turn(text="(unused)"), Turn(text="(unused)")])
    pipeline = build_pipeline(model=agent_model, guardrail_model=guardrail_model)
    record = await run_one(
        "Write me a poem about databases.",
        "schools",
        pipeline,
        catalog=Catalog(tiny_db.parent),
    )

    assert record.status == "guardrail_refused"
    assert "read-only analytics" in record.final_output
    assert record.executed_sql == []


async def test_guardrail_allows_analytics_question(tiny_db: Path):
    guardrail_model = FakeModel(
        script=[Turn(text=json.dumps({"is_analytics_question": True, "reason": "Analytics."}))]
    )
    agent_model = FakeModel(script=happy_path_script())
    pipeline = build_pipeline(model=agent_model, guardrail_model=guardrail_model)
    record = await run_one(
        "Which school has the largest enrollment?",
        "schools",
        pipeline,
        catalog=Catalog(tiny_db.parent),
    )
    assert record.status == "ok"
    assert record.handoff_chain[-1] == "Summarizer"
