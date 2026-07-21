"""The optimizer agent: reads a failure analysis, proposes one bounded PatchSet.

The agent sees only what it may change (mutable prompt sections + memory) and
what happened (failure analysis + current metrics). It cannot see or mention
tools, topology, guardrail code, or the judge — those are outside its cage,
enforced by the patch layer regardless of what it outputs.
"""

from agents import Agent, Runner

from ..agents.memory import StrategyMemory, dump_for_prompt
from ..agents.prompt_loader import PROMPTS_DIR, load_sections
from ..config import settings
from ..eval.report_agent import EvalReport, FailureAnalysis
from .patches import (
    MAX_MEMORY_OPS,
    MAX_PROMPT_PATCHES,
    PatchSet,
    section_budget,
    validate_patchset,
)

OPTIMIZER_INSTRUCTIONS = f"""\
You improve a multi-agent Text-to-SQL system by editing ONLY two things:
1. The `strategy` / `exemplars` sections of each agent's prompt (agent_key in:
   orchestrator, schema_linker, sql_writer, validator, summarizer).
2. The strategy memory (upsert or delete entries; scope = agent_key).

You receive: current metrics, a failure analysis with per-issue hypotheses and
fix directions, the current mutable prompt sections, and the current memory.

Produce ONE PatchSet:
- At most {MAX_PROMPT_PATCHES} prompt patches and {MAX_MEMORY_OPS} memory operations. Fewer,
  sharper changes beat many speculative ones — each change you make dilutes
  attribution of what worked.
- A prompt patch REPLACES the whole section text: include what should remain.
- Memory upserts must cite provenance (failure taxonomy or example ids).
- Prefer editing/deleting a weak existing entry over adding near-duplicates.
- Target the largest failure classes first; reliability (U90) counts as much
  as accuracy — an instruction that reduces variance (e.g. a deterministic
  procedure instead of a judgment call) is as valuable as one that fixes a
  wrong answer.
- Keep every instruction concrete and testable ("CAST the numerator AS REAL")
  rather than aspirational ("be more careful").
- HARD CHARACTER BUDGETS: each section patch has a maximum size listed in the
  input under SECTION BUDGETS - a longer new_text is rejected outright. When
  guidance will not fit a section budget, put it in memory entries instead
  (memory is budgeted separately per agent at render time).

Set rationale to 2-3 sentences: which issues you targeted and why these
specific edits should move A_mean or U90.
"""


def render_optimizer_input(
    report: EvalReport,
    analysis: FailureAnalysis,
    memory: StrategyMemory,
    prompts_dir=PROMPTS_DIR,
) -> str:
    lines = [
        f"CURRENT METRICS: A_mean={report.a_mean:.3f} U90={report.u90:.3f} "
        f"({report.n_failures}/{report.n_records} records failing)",
        "",
        "FAILURE ANALYSIS:",
        f"summary: {analysis.summary}",
    ]
    for issue in analysis.issues:
        lines.append(
            f"- [{issue.taxonomy} -> {issue.responsible_agent}] {issue.hypothesis} "
            f"| fix direction: {issue.fix_direction}"
        )
    lines.append("\nSECTION BUDGETS (max chars for new_text):")
    for key in ("orchestrator", "schema_linker", "sql_writer", "validator", "summarizer"):
        for section in load_sections(key, prompts_dir):
            if not section.frozen:
                # Advertise headroom below the hard cap so a near-miss
                # (the model rarely counts characters exactly) still fits.
                budget = int(section_budget(prompts_dir, key, section.name) * 0.85)
                lines.append(f"- {key}/{section.name}: {budget}")
    lines.append("\nCURRENT MUTABLE PROMPT SECTIONS:")
    for key in ("orchestrator", "schema_linker", "sql_writer", "validator", "summarizer"):
        for section in load_sections(key, prompts_dir):
            if not section.frozen:
                body = section.text or "(empty)"
                lines.append(f"--- {key} / {section.name} ---\n{body}")
    lines.append("\nCURRENT STRATEGY MEMORY:")
    lines.append(dump_for_prompt(memory))
    return "\n".join(lines)


async def propose_patchset(
    report: EvalReport,
    analysis: FailureAnalysis,
    memory: StrategyMemory,
    prompts_dir=PROMPTS_DIR,
    max_attempts: int = 3,
) -> PatchSet:
    """Propose a PatchSet. On bounds violations, retries with the exact error
    fed back; if still over-limit, returns the last attempt for the caller to
    clamp via normalize_patchset."""
    optimizer = Agent(
        name="Optimizer",
        instructions=OPTIMIZER_INSTRUCTIONS,
        output_type=PatchSet,
        model=settings.optimizer_model,
    )
    base_input = render_optimizer_input(report, analysis, memory, prompts_dir)
    feedback = ""
    patchset: PatchSet | None = None
    for _attempt in range(max_attempts):
        result = await Runner.run(optimizer, base_input + feedback)
        patchset = result.final_output_as(PatchSet)
        try:
            validate_patchset(patchset, prompts_dir)
            return patchset
        except ValueError as e:
            feedback = (
                f"\n\nYOUR PREVIOUS PATCHSET WAS REJECTED: {e}\n"
                "Obey every limit exactly: at most 3 prompt patches; at most 5 "
                "memory operations TOTAL (upserts + deletes combined); each "
                "new_text strictly within its advertised SECTION BUDGET. Cut "
                "content or move it into memory entries. Output a corrected "
                "PatchSet now."
            )
    # Still over-limit after retries: return the last attempt — the caller
    # clamps it with normalize_patchset (every clamp is logged, never silent).
    assert patchset is not None
    return patchset
