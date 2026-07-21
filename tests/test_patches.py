"""The patch layer's hard bounds — the optimizer's cage."""

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from ouroboros_sql.agents.memory import MemoryEntry, StrategyMemory
from ouroboros_sql.agents.prompt_loader import PROMPTS_DIR, load_sections
from ouroboros_sql.optimize.patches import (
    GrowthCapExceeded,
    MemoryUpsertOp,
    PatchSet,
    PromptSectionPatch,
    apply_patchset,
    rollback,
    take_snapshot,
)


@pytest.fixture()
def sandbox(tmp_path: Path) -> tuple[Path, Path]:
    prompts_dir = tmp_path / "prompts"
    shutil.copytree(PROMPTS_DIR, prompts_dir)
    memory_path = tmp_path / "memory.json"
    mem = StrategyMemory()
    mem.upsert(
        MemoryEntry(
            id="keep-me", kind="heuristic", scope="validator", text="stay", provenance=["x"]
        )
    )
    mem.save(memory_path)
    return prompts_dir, memory_path


def test_frozen_sections_unreachable():
    with pytest.raises(ValidationError, match="not mutable"):
        PromptSectionPatch(agent_key="sql_writer", section="role", new_text="be evil")
    with pytest.raises(ValidationError, match="unknown agent_key"):
        PromptSectionPatch(agent_key="judge", section="strategy", new_text="x")


def test_patchset_bounds():
    patches = [
        PromptSectionPatch(agent_key="sql_writer", section="strategy", new_text=f"v{i}")
        for i in range(4)
    ]
    with pytest.raises(ValidationError, match="at most 3"):
        PatchSet(rationale="r", prompt_patches=patches)

    ps = PatchSet(
        rationale="r",
        memory_upserts=[
            MemoryUpsertOp(
                id=f"m{i}", kind="heuristic", scope="sql_writer", text="t", provenance=["p"]
            )
            for i in range(4)
        ],
        memory_deletes=["a", "b"],
    )
    with pytest.raises(ValueError, match="at most 5"):
        ps.validate_memory_budget()

    with pytest.raises(ValidationError):  # provenance required
        MemoryUpsertOp(id="m", kind="heuristic", scope="sql_writer", text="t", provenance=[])


def test_growth_cap(sandbox):
    prompts_dir, memory_path = sandbox
    ps = PatchSet(
        rationale="r",
        prompt_patches=[
            PromptSectionPatch(agent_key="sql_writer", section="strategy", new_text="x" * 5000)
        ],
    )
    with pytest.raises(GrowthCapExceeded):
        apply_patchset(ps, prompts_dir, memory_path)


def test_apply_and_rollback_roundtrip(sandbox):
    prompts_dir, memory_path = sandbox
    original = (prompts_dir / "sql_writer.md").read_text()

    ps = PatchSet(
        rationale="teach ratios",
        prompt_patches=[
            PromptSectionPatch(
                agent_key="sql_writer",
                section="strategy",
                new_text="- Always CAST ratio numerators AS REAL.",
            )
        ],
        memory_upserts=[
            MemoryUpsertOp(
                id="new-tip",
                kind="pitfall",
                scope="validator",
                text="Do not accept empty results silently.",
                provenance=["it1"],
            )
        ],
        memory_deletes=["keep-me"],
    )
    snapshot, diffs = apply_patchset(ps, prompts_dir, memory_path)

    # Patch landed: section replaced, frozen role untouched, markers intact.
    sections = {s.name: s for s in load_sections("sql_writer", prompts_dir)}
    assert sections["strategy"].text == "- Always CAST ratio numerators AS REAL."
    assert sections["role"].frozen
    assert sections["role"].text in original  # frozen body byte-identical

    mem = StrategyMemory.load(memory_path)
    assert mem.get("new-tip") is not None
    assert mem.get("keep-me") is None

    # Diff artifacts exist and look like diffs.
    assert "prompt:sql_writer" in diffs and diffs["prompt:sql_writer"].startswith("---")
    assert "+  - Always CAST" in diffs["prompt:sql_writer"].replace(
        "+- Always CAST", "+  - Always CAST"
    )
    assert "memory" in diffs

    # Rollback restores everything byte-for-byte.
    rollback(snapshot, prompts_dir, memory_path)
    assert (prompts_dir / "sql_writer.md").read_text() == original
    restored = StrategyMemory.load(memory_path)
    assert restored.get("keep-me") is not None
    assert restored.get("new-tip") is None


def test_snapshot_covers_all_agents(sandbox):
    prompts_dir, memory_path = sandbox
    snap = take_snapshot(prompts_dir, memory_path)
    assert set(snap.prompt_files) == {
        "orchestrator",
        "schema_linker",
        "sql_writer",
        "validator",
        "summarizer",
    }


def test_apply_is_atomic_on_late_violation(sandbox):
    prompts_dir, memory_path = sandbox
    original = (prompts_dir / "sql_writer.md").read_text()
    ps = PatchSet(
        rationale="r",
        prompt_patches=[
            PromptSectionPatch(agent_key="sql_writer", section="strategy", new_text="- ok tip"),
            PromptSectionPatch(agent_key="validator", section="strategy", new_text="x" * 9000),
        ],
    )
    with pytest.raises(GrowthCapExceeded):
        apply_patchset(ps, prompts_dir, memory_path)
    # First patch must NOT have been written.
    assert (prompts_dir / "sql_writer.md").read_text() == original


def test_validate_patchset_no_writes(sandbox):
    from ouroboros_sql.optimize.patches import section_budget, validate_patchset

    prompts_dir, _memory_path = sandbox
    budget = section_budget(prompts_dir, "sql_writer", "strategy")
    assert budget >= 1200
    ok = PatchSet(
        rationale="r",
        prompt_patches=[
            PromptSectionPatch(agent_key="sql_writer", section="strategy", new_text="- fine")
        ],
    )
    validate_patchset(ok, prompts_dir)  # no exception, no writes
