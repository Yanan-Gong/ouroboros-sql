"""Render EvalMetrics as terminal/markdown tables. Every number traceable to
the run directory it came from."""

from .schema import EvalMetrics, MetricValue


def _row(name: str, mv: MetricValue | None, pct: bool = True) -> str | None:
    if mv is None:
        return None
    return f"| {name} | {mv.fmt(pct=pct)} |"


def to_markdown(metrics: EvalMetrics, run_id: str) -> str:
    lines = [
        f"### Eval: `{metrics.split}` split ({run_id})",
        "",
        f"{metrics.n_examples} examples ({metrics.n_adversarial} adversarial probes) x "
        f"{metrics.n_repeats} repeats = {metrics.n_records} records "
        f"({metrics.n_harness_errors} harness errors)",
        "",
        "| Metric | Value [95% CI] |",
        "|---|---|",
    ]
    rows = [
        _row("Execution accuracy (A_mean)", metrics.a_mean),
        _row("Aptitude (A90)", metrics.a90),
        _row("Worst-case (A10)", metrics.a10),
        _row("Unreliability (U90)", metrics.u90),
        _row("Judge score (mean)", metrics.judge_mean),
        _row("Judge unreliability (U90)", metrics.judge_u90),
        _row("Refusal accuracy (adversarial)", metrics.refusal_accuracy),
        _row("False-refusal rate", metrics.false_refusal_rate),
        _row("Schema-grounding precision", metrics.schema_grounding_precision),
        _row("Schema-grounding recall", metrics.schema_grounding_recall),
        _row("Wasted execute_sql rate", metrics.wasted_call_rate),
        _row("Retry productivity", metrics.retry_productivity),
        _row("Routing accuracy", metrics.routing_accuracy),
        _row("Completion rate", metrics.completion_rate),
        _row("Handoff ping-pong (mean count)", metrics.mean_ping_pong, pct=False),
    ]
    lines.extend(r for r in rows if r is not None)
    if metrics.judge_exec_agreement is not None:
        lines.append(f"| Judge-exec agreement | {metrics.judge_exec_agreement * 100:.1f} |")
    lines.append(
        f"| Tokens per question (in+out) | "
        f"{metrics.mean_input_tokens:.0f}+{metrics.mean_output_tokens:.0f} |"
    )
    lines.append(f"| Model calls per question | {metrics.mean_requests:.1f} |")
    lines.append(
        f"| Latency p50 / p95 (s) | {metrics.latency_p50:.0f} / {metrics.latency_p95:.0f} |"
    )
    if metrics.cost_per_100_usd is not None:
        lines.append(f"| Cost per 100 questions | ${metrics.cost_per_100_usd:.2f} |")
    return "\n".join(lines)
