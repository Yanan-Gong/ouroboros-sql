"""Turn a finished eval run into a structured failure report.

Two stages:
1. `build_eval_report` (deterministic): aggregates taxonomy counts, attributes
   each failure class to the agent most likely responsible, and picks a few
   exemplar failing trajectories per class.
2. `write_failure_analysis` (LLM): reads the report and produces a structured
   analysis — per-issue hypothesis and fix direction. This is the exact input
   the M4 optimizer consumes; it is advisory, never authoritative.
"""

from pathlib import Path

from agents import Agent, Runner
from pydantic import BaseModel, Field

from ..config import settings
from .judge import render_trajectory
from .metrics import compute_metrics
from .schema import EvalRunRecord, load_split, read_jsonl
from .taxonomy import classify

# Which agent to look at first for each failure class.
ATTRIBUTION = {
    "guardrail_missed": "guardrail",
    "false_refusal": "guardrail",
    "max_turns": "orchestrator",
    "harness_error": "infrastructure",
    "no_sql_executed": "validator",
    "sql_never_recovered": "sql_writer",
    "wrong_tables": "schema_linker",
    "wrong_result": "sql_writer",
}

EXEMPLARS_PER_CLASS = 3


class FailureClassReport(BaseModel):
    taxonomy: str
    count: int
    suspected_agent: str
    exemplar_ids: list[str]
    exemplar_trajectories: list[str]


class EvalReport(BaseModel):
    run_id: str
    split: str
    a_mean: float
    u90: float
    completion_rate: float | None
    n_records: int
    n_failures: int
    failure_classes: list[FailureClassReport]


class Issue(BaseModel):
    taxonomy: str
    responsible_agent: str
    hypothesis: str = Field(description="What is going wrong, grounded in the exemplars")
    fix_direction: str = Field(description="Concrete prompt/memory change to try")


class FailureAnalysis(BaseModel):
    summary: str
    issues: list[Issue]


def build_eval_report(run_dir: Path) -> EvalReport:
    records = read_jsonl(run_dir / "records.jsonl")
    metadata_split = run_dir.name.split("-")[0] if "-" in run_dir.name else "val"
    import json

    metadata_file = run_dir / "metadata.json"
    if metadata_file.is_file():
        metadata_split = json.loads(metadata_file.read_text())["split"]
    examples = load_split(metadata_split)  # type: ignore[arg-type]
    by_id = {ex.id: ex for ex in examples}

    failures: dict[str, list[EvalRunRecord]] = {}
    for record in records:
        label = classify(record, by_id[record.example_id])
        if label is not None:
            failures.setdefault(label, []).append(record)

    class_reports = []
    for label, recs in sorted(failures.items(), key=lambda kv: -len(kv[1])):
        # Prefer diverse exemplars: distinct examples over repeat duplicates.
        seen_examples: set[str] = set()
        exemplars: list[EvalRunRecord] = []
        for r in recs:
            if r.example_id not in seen_examples:
                exemplars.append(r)
                seen_examples.add(r.example_id)
            if len(exemplars) == EXEMPLARS_PER_CLASS:
                break
        class_reports.append(
            FailureClassReport(
                taxonomy=label,
                count=len(recs),
                suspected_agent=ATTRIBUTION.get(label, "unknown"),
                exemplar_ids=[r.example_id for r in exemplars],
                exemplar_trajectories=[render_trajectory(r, max_chars=2500) for r in exemplars],
            )
        )

    n_repeats = max((r.repeat_index for r in records), default=0) + 1
    metrics = compute_metrics(records, examples, metadata_split, n_repeats)
    return EvalReport(
        run_id=run_dir.name,
        split=metadata_split,
        a_mean=metrics.a_mean.value,
        u90=metrics.u90.value,
        completion_rate=metrics.completion_rate.value if metrics.completion_rate else None,
        n_records=len(records),
        n_failures=sum(len(v) for v in failures.values()),
        failure_classes=class_reports,
    )


ANALYST_INSTRUCTIONS = """\
You analyze eval failures of a multi-agent Text-to-SQL system
(Orchestrator -> SchemaLinker -> SQLWriter -> Validator -> Summarizer, with a
relevance guardrail). You receive aggregate metrics, a failure taxonomy with
suspected agents, and exemplar failing trajectories.

For each substantial failure class, produce one issue:
- hypothesis: the specific behavior going wrong, citing evidence from the
  exemplar trajectories (quote SQL fragments or tool calls where useful).
- fix_direction: a concrete, minimal change to that agent's strategy guidance
  or memory (NOT code, NOT topology) that would plausibly reduce this class.

Be specific to what you see. "Write better SQL" is useless; "when the question
asks for a ratio, CAST the numerator AS REAL before dividing — exemplars 1 and
3 used integer division" is the level required. Order issues by impact
(count x plausibility of fix). Summary: three sentences max.
"""


def render_report_for_analyst(report: EvalReport) -> str:
    lines = [
        f"RUN: {report.run_id} split={report.split}",
        f"A_mean={report.a_mean:.3f} U90={report.u90:.3f} completion={report.completion_rate}",
        f"{report.n_failures} failing records of {report.n_records}",
        "",
    ]
    for fc in report.failure_classes:
        lines.append(f"## {fc.taxonomy} (count={fc.count}, suspect={fc.suspected_agent})")
        for i, (ex_id, traj) in enumerate(
            zip(fc.exemplar_ids, fc.exemplar_trajectories, strict=True), 1
        ):
            lines.append(f"--- exemplar {i} ({ex_id}) ---")
            lines.append(traj)
        lines.append("")
    return "\n".join(lines)


async def write_failure_analysis(report: EvalReport) -> FailureAnalysis:
    analyst = Agent(
        name="FailureAnalyst",
        instructions=ANALYST_INSTRUCTIONS,
        output_type=FailureAnalysis,
        model=settings.optimizer_model,
    )
    result = await Runner.run(analyst, render_report_for_analyst(report))
    return result.final_output_as(FailureAnalysis)


def analysis_to_markdown(report: EvalReport, analysis: FailureAnalysis) -> str:
    lines = [
        f"# Failure analysis — {report.run_id}",
        "",
        f"A_mean {report.a_mean:.1%} · U90 {report.u90:.1%} · "
        f"{report.n_failures}/{report.n_records} records failing",
        "",
        analysis.summary,
        "",
    ]
    for issue in analysis.issues:
        lines.extend(
            [
                f"## {issue.taxonomy} → {issue.responsible_agent}",
                f"**Hypothesis**: {issue.hypothesis}",
                "",
                f"**Fix direction**: {issue.fix_direction}",
                "",
            ]
        )
    return "\n".join(lines)
