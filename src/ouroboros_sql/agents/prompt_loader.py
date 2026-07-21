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


def render_instructions(agent_key: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    """Final instruction string for an agent: sections joined, empty ones dropped."""
    parts = []
    for section in load_sections(agent_key, prompts_dir):
        if not section.text or section.text.startswith("(No learned"):
            continue
        if section.name == "strategy":
            parts.append("## Learned strategies\n" + section.text)
        elif section.name == "exemplars":
            parts.append("## Worked examples\n" + section.text)
        else:
            parts.append(section.text)
    return "\n\n".join(parts)
