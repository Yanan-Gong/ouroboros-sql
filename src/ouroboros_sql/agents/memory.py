"""Strategy memory: the system's evolvable knowledge, separate from its code.

ALMA-flavored design: memory is a first-class, inspectable artifact. Each
entry is a small piece of learned knowledge (a heuristic, a worked exemplar,
or a pitfall to avoid), scoped to one agent, carrying provenance — the failure
ids that motivated it. The optimizer (M4) upserts and deletes entries; humans
can read the JSON and see exactly why each entry exists.

MemAgent-flavored constraint: rendered memory is token-capped per agent. The
store may hold more than fits; rendering evicts deterministically (lowest
hit_count first, then oldest) so the prompt-side memory never grows unbounded.
"""

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..config import settings

EntryKind = Literal["heuristic", "exemplar", "pitfall"]

AGENT_SCOPES = ("orchestrator", "schema_linker", "sql_writer", "validator", "summarizer")

# Rendered budget per agent, in approximate tokens (chars / 4).
DEFAULT_TOKEN_BUDGET = 900


class MemoryEntry(BaseModel):
    id: str
    kind: EntryKind
    scope: str  # one of AGENT_SCOPES
    text: str
    provenance: list[str] = Field(default_factory=list)  # failure/example ids or "manual-seed"
    created: str = ""  # ISO date, set on add
    hit_count: int = 0  # incremented when the optimizer keeps/cites the entry

    def approx_tokens(self) -> int:
        return max(1, len(self.text) // 4)


class StrategyMemory(BaseModel):
    entries: list[MemoryEntry] = Field(default_factory=list)
    token_budget_per_agent: int = DEFAULT_TOKEN_BUDGET

    # -- persistence ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> "StrategyMemory":
        path = path or settings.memory_path
        if not path.is_file():
            return cls()
        return cls.model_validate_json(path.read_text())

    def save(self, path: Path | None = None) -> None:
        path = path or settings.memory_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")

    # -- mutation (used by seeding now, by the optimizer in M4) --------------

    def upsert(self, entry: MemoryEntry) -> None:
        if entry.scope not in AGENT_SCOPES:
            raise ValueError(f"Unknown scope {entry.scope!r}; expected one of {AGENT_SCOPES}")
        if not entry.created:
            entry.created = date.today().isoformat()
        existing = self.get(entry.id)
        if existing is not None:
            self.entries[self.entries.index(existing)] = entry
        else:
            self.entries.append(entry)

    def delete(self, entry_id: str) -> bool:
        entry = self.get(entry_id)
        if entry is None:
            return False
        self.entries.remove(entry)
        return True

    def get(self, entry_id: str) -> MemoryEntry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    # -- rendering -----------------------------------------------------------

    def entries_for(self, scope: str) -> list[MemoryEntry]:
        """Entries for one agent, within the token budget.

        Eviction is deterministic: keep higher hit_count first, then newer
        entries; ties broken by id for reproducibility.
        """
        scoped = [e for e in self.entries if e.scope == scope]
        ranked = sorted(scoped, key=lambda e: (-e.hit_count, e.created, e.id))
        kept: list[MemoryEntry] = []
        budget = self.token_budget_per_agent
        for entry in ranked:
            cost = entry.approx_tokens()
            if cost <= budget:
                kept.append(entry)
                budget -= cost
        return kept

    def render_sections(self, scope: str) -> tuple[str, str]:
        """(strategy_text, exemplars_text) for injection into prompt sections."""
        kept = self.entries_for(scope)
        strategies = [e for e in kept if e.kind in ("heuristic", "pitfall")]
        exemplars = [e for e in kept if e.kind == "exemplar"]
        strategy_lines = []
        for e in strategies:
            prefix = "Avoid: " if e.kind == "pitfall" else ""
            strategy_lines.append(f"- {prefix}{e.text}")
        exemplar_lines = [e.text for e in exemplars]
        return "\n".join(strategy_lines), "\n\n".join(exemplar_lines)

    def stats(self) -> dict:
        by_scope: dict[str, int] = {}
        for e in self.entries:
            by_scope[e.scope] = by_scope.get(e.scope, 0) + 1
        return {
            "total_entries": len(self.entries),
            "by_scope": by_scope,
            "approx_tokens_total": sum(e.approx_tokens() for e in self.entries),
        }


def dump_for_prompt(memory: StrategyMemory) -> str:
    """Compact JSON view of the whole store — given to the optimizer (M4)."""
    return json.dumps([e.model_dump() for e in memory.entries], indent=1, ensure_ascii=False)
