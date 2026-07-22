"""Memory renders into agent instructions without touching frozen sections."""

from ouroboros_sql.agents.memory import MemoryEntry, StrategyMemory
from ouroboros_sql.agents.prompt_loader import render_instructions
from ouroboros_sql.agents.topology import build_pipeline


def make_memory() -> StrategyMemory:
    mem = StrategyMemory()
    mem.upsert(
        MemoryEntry(
            id="w1",
            kind="heuristic",
            scope="sql_writer",
            text="For ratio questions, CAST the numerator AS REAL before dividing.",
            provenance=["bird_mini_dev_0042"],
        )
    )
    mem.upsert(
        MemoryEntry(
            id="w2",
            kind="pitfall",
            scope="sql_writer",
            text="Do not add extra columns the question did not ask for.",
        )
    )
    mem.upsert(
        MemoryEntry(
            id="l1",
            kind="heuristic",
            scope="schema_linker",
            text="Call sample_rows on every column you will filter on.",
        )
    )
    return mem


def test_memory_appears_in_matching_agent_only():
    mem = make_memory()
    writer = render_instructions("sql_writer", memory=mem)
    linker = render_instructions("schema_linker", memory=mem)
    summarizer = render_instructions("summarizer", memory=mem)

    assert "CAST the numerator AS REAL" in writer
    assert "Avoid: Do not add extra columns" in writer
    assert "sample_rows on every column" not in writer

    assert "sample_rows on every column" in linker
    assert "## Learned strategies" in linker

    # No summarizer entries in this memory: none of its text may leak there.
    # (The file's own strategy section may hold optimizer content — that's fine.)
    assert "CAST the numerator AS REAL" not in summarizer
    assert "sample_rows on every column" not in summarizer


def test_no_memory_means_unchanged_instructions():
    assert render_instructions("sql_writer") == render_instructions("sql_writer", memory=None)
    empty = StrategyMemory()
    assert render_instructions("sql_writer", memory=empty) == render_instructions("sql_writer")


def test_frozen_role_text_is_untouched_by_memory():
    mem = make_memory()
    with_mem = render_instructions("sql_writer", memory=mem)
    without = render_instructions("sql_writer")
    # The role section (everything before learned strategies) is identical.
    assert with_mem.startswith(without)


def test_pipeline_builds_with_memory():
    mem = make_memory()
    pipeline = build_pipeline(model="test-model", with_relevance_guardrail=False, memory=mem)
    assert "CAST the numerator AS REAL" in pipeline.sql_writer.instructions
    assert "sample_rows on every column" in pipeline.schema_linker.instructions
    assert "CAST the numerator" not in pipeline.validator.instructions
