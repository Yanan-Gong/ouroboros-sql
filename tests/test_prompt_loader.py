from pathlib import Path

import pytest

from ouroboros_sql.agents.prompt_loader import (
    MUTABLE_SECTIONS,
    load_sections,
    parse_sections,
    render_instructions,
)

AGENT_KEYS = ["orchestrator", "schema_linker", "sql_writer", "validator", "summarizer"]


@pytest.mark.parametrize("key", AGENT_KEYS)
def test_all_prompts_have_frozen_role_and_mutable_sections(key: str):
    sections = load_sections(key)
    by_name = {s.name: s for s in sections}
    assert by_name["role"].frozen
    for name in MUTABLE_SECTIONS:
        assert name in by_name
        assert not by_name[name].frozen


@pytest.mark.parametrize("key", AGENT_KEYS)
def test_render_produces_nonempty_instructions(key: str):
    text = render_instructions(key)
    assert len(text) > 100
    assert "SECTION" not in text  # markers never leak into instructions


def test_placeholder_strategy_sections_are_dropped():
    text = render_instructions("sql_writer")
    assert "No learned strategies" not in text
    assert "## Learned strategies" not in text


def test_parse_sections_roundtrip(tmp_path: Path):
    raw = (
        "<!-- SECTION: role (frozen) -->\nBe helpful.\n"
        "<!-- SECTION: strategy -->\nAlways check joins.\n"
        "<!-- SECTION: exemplars -->\nQ: x A: y\n"
    )
    sections = parse_sections(raw)
    assert [s.name for s in sections] == ["role", "strategy", "exemplars"]
    assert sections[0].frozen and not sections[1].frozen
    assert sections[1].text == "Always check joins."

    (tmp_path / "custom.md").write_text(raw)
    text = render_instructions("custom", prompts_dir=tmp_path)
    assert "Be helpful." in text
    assert "## Learned strategies\nAlways check joins." in text
    assert "## Worked examples\nQ: x A: y" in text
