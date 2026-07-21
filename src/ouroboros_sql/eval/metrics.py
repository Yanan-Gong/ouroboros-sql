"""Metric computation over EvalRunRecords.

Reliability decomposition follows "LLMs Get Lost in Multi-Turn Conversation"
(arXiv:2505.06120): run every instance N times, then decompose performance
into aptitude (per-instance 90th percentile, averaged) and unreliability
(the gap between per-instance 90th and 10th percentiles). With binary
execution accuracy and small N, the percentiles degenerate toward
"succeeded at least once" / "failed at least once" — documented, and
complemented by the continuous judge score when available.

All aggregates carry bootstrap confidence intervals resampled over
*instances* (not records), since repeats of one instance are correlated.
"""

import random
from collections import defaultdict
from collections.abc import Callable, Sequence

import sqlglot
from sqlglot import expressions as exp

from .schema import EvalMetrics, EvalRunRecord, GoldenExample, MetricValue

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 17
EXPECTED_CHAIN = ["SchemaLinker", "SQLWriter", "Validator", "Summarizer"]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' method), pure python."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty")
    idx = q * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bootstrap_ci(
    per_instance: Sequence[float],
    stat: Callable[[Sequence[float]], float] = lambda xs: sum(xs) / len(xs),
) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(per_instance)
    if n == 0:
        return (0.0, 0.0)
    draws = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [per_instance[rng.randrange(n)] for _ in range(n)]
        draws.append(stat(sample))
    return (percentile(draws, 0.025), percentile(draws, 0.975))


def metric_from_instances(per_instance: Sequence[float]) -> MetricValue:
    if not per_instance:
        return MetricValue(value=0.0, n=0)
    mean = sum(per_instance) / len(per_instance)
    lo, hi = bootstrap_ci(per_instance)
    return MetricValue(value=mean, ci_low=lo, ci_high=hi, n=len(per_instance))


def tables_in_sql(sql: str) -> set[str]:
    """Table names referenced by a query (lowercased), via sqlglot AST."""
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError:
        return set()
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name.lower() not in cte_names}


# ---------------------------------------------------------------- per-record


def described_tables(record: EvalRunRecord) -> set[str]:
    """Tables the SchemaLinker explicitly inspected (describe_table/sample_rows)."""
    seen = set()
    for event in record.events:
        if event.get("kind") != "tool_call":
            continue
        if event.get("payload", {}).get("tool") in ("describe_table", "sample_rows"):
            args = event["payload"].get("arguments") or "{}"
            import json as _json

            try:
                table = _json.loads(args).get("table")
            except (ValueError, AttributeError):
                table = None
            if table:
                seen.add(str(table).lower())
    return seen


def execute_sql_attempts(record: EvalRunRecord) -> list[dict]:
    return list(record.executed_sql)


def ping_pong_count(record: EvalRunRecord) -> int:
    return max(0, len(record.handoff_chain) - len(EXPECTED_CHAIN))


def routing_ok(record: EvalRunRecord) -> bool:
    return bool(record.handoff_chain) and record.handoff_chain[0] == "SchemaLinker"


def completion_ok(record: EvalRunRecord) -> bool:
    return (
        record.status == "ok"
        and record.final_executed_sql is not None
        and bool(record.handoff_chain)
        and record.handoff_chain[-1] == "Summarizer"
    )


# ---------------------------------------------------------------- aggregate


def _group_by_example(records: list[EvalRunRecord]) -> dict[str, list[EvalRunRecord]]:
    grouped: dict[str, list[EvalRunRecord]] = defaultdict(list)
    for r in records:
        grouped[r.example_id].append(r)
    return grouped


def compute_metrics(
    records: list[EvalRunRecord],
    examples: list[GoldenExample],
    split: str,
    n_repeats: int,
    cost_per_million: tuple[float, float] | None = None,
) -> EvalMetrics:
    by_id = {ex.id: ex for ex in examples}
    normal = [r for r in records if not by_id[r.example_id].adversarial]
    adversarial = [r for r in records if by_id[r.example_id].adversarial]

    # --- reliability decomposition over execution accuracy
    grouped = _group_by_example(normal)
    inst_mean, inst_p90, inst_p10, inst_u90 = [], [], [], []
    for _ex_id, runs in sorted(grouped.items()):
        scores = [1.0 if r.execution_match else 0.0 for r in runs]
        inst_mean.append(sum(scores) / len(scores))
        p90, p10 = percentile(scores, 0.9), percentile(scores, 0.1)
        inst_p90.append(p90)
        inst_p10.append(p10)
        inst_u90.append(p90 - p10)

    # --- judge decomposition (continuous), if judge scores exist
    judge_mean = judge_u90 = None
    judge_exec_agreement = None
    judged = [r for r in normal if r.judge is not None and "overall" in r.judge]
    if judged:
        jg = _group_by_example(judged)
        j_inst_mean, j_inst_u90 = [], []
        for _ex_id, runs in sorted(jg.items()):
            scores = [float(r.judge["overall"]) for r in runs if r.judge is not None]
            j_inst_mean.append(sum(scores) / len(scores))
            j_inst_u90.append(percentile(scores, 0.9) - percentile(scores, 0.1))
        judge_mean = metric_from_instances(j_inst_mean)
        judge_u90 = metric_from_instances(j_inst_u90)
        # Agreement: does a high judge score co-occur with exec match?
        agree = [
            1.0 if (float(r.judge["overall"]) >= 0.5) == bool(r.execution_match) else 0.0
            for r in judged
            if r.judge is not None
        ]
        judge_exec_agreement = sum(agree) / len(agree)

    # --- guardrails
    refusal_accuracy = None
    if adversarial:
        ag = _group_by_example(adversarial)
        refusal_accuracy = metric_from_instances(
            [
                sum(1.0 if r.refusal_correct else 0.0 for r in runs) / len(runs)
                for runs in ag.values()
            ]
        )
    false_refusals = [
        sum(1.0 if r.status == "guardrail_refused" else 0.0 for r in runs) / len(runs)
        for runs in grouped.values()
    ]
    false_refusal_rate = metric_from_instances(false_refusals) if grouped else None

    # --- tool usage (instance-level averages over repeats)
    precisions: list[float] = []
    recalls: list[float] = []
    wasted: list[float] = []
    productive: list[float] = []
    for ex_id, runs in sorted(grouped.items()):
        gold_tables = tables_in_sql(by_id[ex_id].gold_sql or "")
        run_p, run_r, run_w, run_pr = [], [], [], []
        for r in runs:
            described = described_tables(r)
            if described and gold_tables:
                run_p.append(len(described & gold_tables) / len(described))
                run_r.append(len(described & gold_tables) / len(gold_tables))
            attempts = execute_sql_attempts(r)
            if attempts:
                errors = [a for a in attempts if not a.get("ok")]
                run_w.append(len(errors) / len(attempts))
                if errors:
                    recovered = sum(
                        1.0
                        for i, a in enumerate(attempts)
                        if not a.get("ok") and any(b.get("ok") for b in attempts[i + 1 :])
                    )
                    run_pr.append(recovered / len(errors))
        for src, dst in (
            (run_p, precisions),
            (run_r, recalls),
            (run_w, wasted),
            (run_pr, productive),
        ):
            if src:
                dst.append(sum(src) / len(src))

    # --- handoffs
    routing = [
        sum(1.0 if routing_ok(r) else 0.0 for r in runs) / len(runs) for runs in grouped.values()
    ]
    completion = [
        sum(1.0 if completion_ok(r) else 0.0 for r in runs) / len(runs) for runs in grouped.values()
    ]
    pingpong = [
        sum(float(ping_pong_count(r)) for r in runs) / len(runs) for runs in grouped.values()
    ]

    # --- cost & latency over all records
    latencies = sorted(r.wall_seconds for r in records) or [0.0]
    mean_in = sum(r.input_tokens for r in records) / max(len(records), 1)
    mean_out = sum(r.output_tokens for r in records) / max(len(records), 1)
    cost = None
    if cost_per_million is not None:
        in_price, out_price = cost_per_million
        cost = (mean_in * in_price + mean_out * out_price) / 1e6 * 100

    return EvalMetrics(
        split=split,
        n_examples=len(by_id),
        n_adversarial=sum(1 for ex in examples if ex.adversarial),
        n_repeats=n_repeats,
        n_records=len(records),
        n_harness_errors=sum(1 for r in records if r.error),
        a_mean=metric_from_instances(inst_mean),
        a90=metric_from_instances(inst_p90),
        a10=metric_from_instances(inst_p10),
        u90=metric_from_instances(inst_u90),
        judge_mean=judge_mean,
        judge_u90=judge_u90,
        judge_exec_agreement=judge_exec_agreement,
        refusal_accuracy=refusal_accuracy,
        false_refusal_rate=false_refusal_rate,
        schema_grounding_precision=metric_from_instances(precisions) if precisions else None,
        schema_grounding_recall=metric_from_instances(recalls) if recalls else None,
        wasted_call_rate=metric_from_instances(wasted) if wasted else None,
        retry_productivity=metric_from_instances(productive) if productive else None,
        routing_accuracy=metric_from_instances(routing) if routing else None,
        completion_rate=metric_from_instances(completion) if completion else None,
        mean_ping_pong=metric_from_instances(pingpong) if pingpong else None,
        mean_input_tokens=mean_in,
        mean_output_tokens=mean_out,
        mean_requests=sum(r.requests for r in records) / max(len(records), 1),
        latency_p50=percentile(latencies, 0.5),
        latency_p95=percentile(latencies, 0.95),
        cost_per_100_usd=cost,
    )
