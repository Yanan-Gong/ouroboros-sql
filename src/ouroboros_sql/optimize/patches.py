"""Typed, bounded, reversible patches — the only way the optimizer changes the system.

Hard constraints enforced here, not by prompt:
- Prompt patches may touch only the mutable sections (`strategy`, `exemplars`);
  frozen sections are structurally unreachable.
- Per-patch growth cap: a section may not balloon past 1.3x its current size
  (with a floor so empty sections can be seeded).
- At most 3 prompt patches and 5 memory operations per PatchSet.
- Memory upserts must carry provenance.
- `apply` returns a snapshot; `rollback` restores it byte-for-byte. Every
  application writes unified diffs for the iteration record.
"""

import difflib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from ..agents.memory import AGENT_SCOPES, EntryKind, MemoryEntry, StrategyMemory
from ..agents.prompt_loader import MUTABLE_SECTIONS, PROMPTS_DIR

MAX_PROMPT_PATCHES = 3
MAX_MEMORY_OPS = 5
GROWTH_FACTOR = 1.3
MIN_SECTION_BUDGET_CHARS = 900


class PromptSectionPatch(BaseModel):
    agent_key: str
    section: str  # strategy | exemplars
    new_text: str

    @field_validator("agent_key")
    @classmethod
    def _known_agent(cls, v: str) -> str:
        if v not in AGENT_SCOPES:
            raise ValueError(f"unknown agent_key {v!r}")
        return v

    @field_validator("section")
    @classmethod
    def _mutable_section(cls, v: str) -> str:
        if v not in MUTABLE_SECTIONS:
            raise ValueError(f"section {v!r} is not mutable (allowed: {MUTABLE_SECTIONS})")
        return v


class MemoryUpsertOp(BaseModel):
    id: str
    kind: EntryKind
    scope: str
    text: str
    provenance: list[str] = Field(min_length=1)


class PatchSet(BaseModel):
    rationale: str
    prompt_patches: list[PromptSectionPatch] = Field(default_factory=list)
    memory_upserts: list[MemoryUpsertOp] = Field(default_factory=list)
    memory_deletes: list[str] = Field(default_factory=list)

    @field_validator("prompt_patches")
    @classmethod
    def _max_prompt(cls, v: list) -> list:
        if len(v) > MAX_PROMPT_PATCHES:
            raise ValueError(f"at most {MAX_PROMPT_PATCHES} prompt patches per iteration")
        return v

    def validate_memory_budget(self) -> None:
        if len(self.memory_upserts) + len(self.memory_deletes) > MAX_MEMORY_OPS:
            raise ValueError(f"at most {MAX_MEMORY_OPS} memory operations per iteration")

    @property
    def is_empty(self) -> bool:
        return not (self.prompt_patches or self.memory_upserts or self.memory_deletes)


class Snapshot(BaseModel):
    """Byte-exact state before a PatchSet was applied."""

    prompt_files: dict[str, str]  # agent_key -> full file content
    memory_json: str


class GrowthCapExceeded(ValueError):
    pass


def _replace_section(raw: str, section: str, new_text: str) -> str:
    """Replace one section's body in a prompt file, preserving markers."""
    sections = load_sections_from_raw(raw)
    out: list[str] = []
    for name, frozen, marker, body in sections:
        out.append(marker)
        if name == section and not frozen:
            out.append(new_text.strip() + "\n")
        elif body.strip():
            out.append(body.strip() + "\n")
        out.append("\n")
    return "".join(out).rstrip() + "\n"


def load_sections_from_raw(raw: str) -> list[tuple[str, bool, str, str]]:
    """(name, frozen, marker_line, body) preserving original marker text."""
    import re

    marker_re = re.compile(r"<!--\s*SECTION:\s*(?P<name>\w+)(?P<frozen>\s*\(frozen\))?\s*-->")
    matches = list(marker_re.finditer(raw))
    result = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        result.append(
            (m.group("name"), bool(m.group("frozen")), m.group(0) + "\n", raw[m.end() : end])
        )
    return result


def check_growth(old_body: str, new_text: str) -> None:
    budget = max(int(len(old_body) * GROWTH_FACTOR), MIN_SECTION_BUDGET_CHARS)
    if len(new_text) > budget:
        raise GrowthCapExceeded(
            f"section grows to {len(new_text)} chars, budget {budget} "
            f"(1.3x current or {MIN_SECTION_BUDGET_CHARS} floor)"
        )


def take_snapshot(prompts_dir: Path = PROMPTS_DIR, memory_path: Path | None = None) -> Snapshot:
    memory = StrategyMemory.load(memory_path)
    return Snapshot(
        prompt_files={key: (prompts_dir / f"{key}.md").read_text() for key in AGENT_SCOPES},
        memory_json=memory.model_dump_json(indent=2),
    )


def apply_patchset(
    patchset: PatchSet,
    prompts_dir: Path = PROMPTS_DIR,
    memory_path: Path | None = None,
) -> tuple[Snapshot, dict[str, str]]:
    """Apply after validating every bound. Returns (snapshot, diffs by artifact)."""
    patchset.validate_memory_budget()
    snapshot = take_snapshot(prompts_dir, memory_path)
    diffs: dict[str, str] = {}

    for patch in patchset.prompt_patches:
        path = prompts_dir / f"{patch.agent_key}.md"
        old_raw = path.read_text()
        sections = {name: body for name, _f, _m, body in load_sections_from_raw(old_raw)}
        old_body = sections.get(patch.section, "")
        if old_body.strip().startswith("(No learned"):
            old_body = ""
        check_growth(old_body, patch.new_text)
        new_raw = _replace_section(old_raw, patch.section, patch.new_text)
        path.write_text(new_raw)
        diffs[f"prompt:{patch.agent_key}"] = "".join(
            difflib.unified_diff(
                old_raw.splitlines(keepends=True),
                new_raw.splitlines(keepends=True),
                fromfile=f"{patch.agent_key}.md (before)",
                tofile=f"{patch.agent_key}.md (after)",
            )
        )

    if patchset.memory_upserts or patchset.memory_deletes:
        memory = StrategyMemory.load(memory_path)
        before = memory.model_dump_json(indent=2)
        for op in patchset.memory_upserts:
            memory.upsert(
                MemoryEntry(
                    id=op.id,
                    kind=op.kind,
                    scope=op.scope,
                    text=op.text,
                    provenance=op.provenance,
                )
            )
        for entry_id in patchset.memory_deletes:
            memory.delete(entry_id)
        memory.save(memory_path)
        after = memory.model_dump_json(indent=2)
        diffs["memory"] = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="strategy_memory.json (before)",
                tofile="strategy_memory.json (after)",
            )
        )

    return snapshot, diffs


def rollback(
    snapshot: Snapshot,
    prompts_dir: Path = PROMPTS_DIR,
    memory_path: Path | None = None,
) -> None:
    for key, content in snapshot.prompt_files.items():
        (prompts_dir / f"{key}.md").write_text(content)
    memory = StrategyMemory.model_validate_json(snapshot.memory_json)
    memory.save(memory_path)
