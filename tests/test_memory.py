from pathlib import Path

import pytest

from ouroboros_sql.agents.memory import MemoryEntry, StrategyMemory


def entry(
    id: str,
    scope: str = "sql_writer",
    kind: str = "heuristic",
    text: str = "t",
    hit_count: int = 0,
    created: str = "2026-07-01",
) -> MemoryEntry:
    return MemoryEntry(
        id=id, kind=kind, scope=scope, text=text, hit_count=hit_count, created=created
    )


def test_upsert_and_delete():
    mem = StrategyMemory()
    mem.upsert(entry("a", text="first"))
    mem.upsert(entry("a", text="updated"))
    assert len(mem.entries) == 1
    assert mem.get("a").text == "updated"
    assert mem.delete("a") is True
    assert mem.delete("a") is False


def test_upsert_rejects_unknown_scope():
    mem = StrategyMemory()
    with pytest.raises(ValueError, match="Unknown scope"):
        mem.upsert(entry("x", scope="nonexistent_agent"))


def test_created_stamped_on_add():
    mem = StrategyMemory()
    mem.upsert(MemoryEntry(id="n", kind="heuristic", scope="validator", text="x"))
    assert mem.get("n").created  # ISO date auto-filled


def test_scope_isolation():
    mem = StrategyMemory()
    mem.upsert(entry("w1", scope="sql_writer"))
    mem.upsert(entry("l1", scope="schema_linker"))
    assert [e.id for e in mem.entries_for("sql_writer")] == ["w1"]
    assert [e.id for e in mem.entries_for("schema_linker")] == ["l1"]


def test_token_budget_eviction_prefers_hits_then_recency():
    mem = StrategyMemory(token_budget_per_agent=50)
    # ~40 tokens each (160 chars): only one fits the 50-token budget.
    long_text = "x" * 160
    mem.upsert(entry("old_unused", text=long_text, hit_count=0, created="2026-01-01"))
    mem.upsert(entry("hot", text=long_text, hit_count=5, created="2026-01-01"))
    kept = mem.entries_for("sql_writer")
    assert [e.id for e in kept] == ["hot"]

    # With equal hits, older entries win (stability), ties broken by id.
    mem2 = StrategyMemory(token_budget_per_agent=50)
    mem2.upsert(entry("b_newer", text=long_text, created="2026-06-01"))
    mem2.upsert(entry("a_older", text=long_text, created="2026-01-01"))
    assert [e.id for e in mem2.entries_for("sql_writer")] == ["a_older"]


def test_oversized_entry_skipped_but_smaller_kept():
    mem = StrategyMemory(token_budget_per_agent=10)
    mem.upsert(entry("huge", text="x" * 400, hit_count=9))
    mem.upsert(entry("small", text="short tip"))
    assert [e.id for e in mem.entries_for("sql_writer")] == ["small"]


def test_render_sections_split_by_kind():
    mem = StrategyMemory()
    mem.upsert(entry("h", kind="heuristic", text="Always check joins."))
    mem.upsert(entry("p", kind="pitfall", text="COUNT(*) counts NULL rows too."))
    mem.upsert(entry("e", kind="exemplar", text="Q: total sales?\nSQL: SELECT SUM(x) FROM t"))
    strategy, exemplars = mem.render_sections("sql_writer")
    assert "- Always check joins." in strategy
    assert "- Avoid: COUNT(*) counts NULL rows too." in strategy
    assert exemplars.startswith("Q: total sales?")


def test_roundtrip_persistence(tmp_path: Path):
    mem = StrategyMemory()
    mem.upsert(entry("a", text="persist me", hit_count=2))
    path = tmp_path / "mem.json"
    mem.save(path)
    loaded = StrategyMemory.load(path)
    assert loaded.get("a").hit_count == 2
    assert StrategyMemory.load(tmp_path / "missing.json").entries == []


def test_stats():
    mem = StrategyMemory()
    mem.upsert(entry("a", scope="sql_writer"))
    mem.upsert(entry("b", scope="validator"))
    s = mem.stats()
    assert s["total_entries"] == 2
    assert s["by_scope"] == {"sql_writer": 1, "validator": 1}
