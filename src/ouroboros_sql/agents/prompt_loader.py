"""Sectioned prompt files.

Each agent prompt is a markdown file split into named sections by
`<!-- SECTION: name -->` markers. Sections marked `(frozen)` may never be
modified by the optimizer; `strategy` and `exemplars` are its only mutation
surface. Keeping prompts as diffable files (rather than Python strings) makes
every optimizer patch a reviewable artifact.
"""

import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

_SECTION_RE = re.compile(r"<!--\s*SECTION:\s*(?P<name>\w+)(?P<frozen>\s*\(frozen\))?\s*-->")

MUTABLE_SECTIONS = ("strategy", "exemplars")


@dataclass
class Section:
    name: str
    frozen: bool
    text: str


def parse_sections(raw: str) -> list[Section]:
    sections: list[Section] = []
    matches = list(_SECTION_RE.finditer(raw))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        sections.append(
            Section(
                name=m.group("name"),
                frozen=bool(m.group("frozen")),
                text=raw[m.end() : end].strip(),
            )
        )
    return sections


def load_sections(agent_key: str, prompts_dir: Path = PROMPTS_DIR) -> list[Section]:
    raw = (prompts_dir / f"{agent_key}.md").read_text()
    sections = parse_sections(raw)
    if not sections:
        raise ValueError(f"Prompt file for {agent_key!r} has no SECTION markers")
    return sections


def render_instructions(
    agent_key: str,
    prompts_dir: Path = PROMPTS_DIR,
    memory: object | None = None,
) -> str:
    """Final instruction string for an agent: sections joined, empty ones dropped.

    When a StrategyMemory is given, its rendered entries for this agent are
    appended to the file's own `strategy`/`exemplars` sections — file content
    (written by the optimizer) and memory content compose rather than compete.
    """
    memory_strategy = memory_exemplars = ""
    if memory is not None:
        memory_strategy, memory_exemplars = memory.render_sections(agent_key)  # type: ignore[attr-defined]

    parts = []
    for section in load_sections(agent_key, prompts_dir):
        text = section.text
        if text.startswith("(No learned"):
            text = ""
        if section.name == "strategy":
            text = "\n".join(t for t in (text, memory_strategy) if t)
            if text:
                parts.append("## Learned strategies\n" + text)
        elif section.name == "exemplars":
            text = "\n\n".join(t for t in (text, memory_exemplars) if t)
            if text:
                parts.append("## Worked examples\n" + text)
        elif text:
            parts.append(text)
    return "\n\n".join(parts)
