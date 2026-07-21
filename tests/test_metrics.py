"""Metric math on synthetic records — no models, no databases."""

from ouroboros_sql.eval.metrics import (
    compute_metrics,
    described_tables,
    percentile,
    tables_in_sql,
)
from ouroboros_sql.eval.schema import EvalRunRecord, GoldenExample


def make_example(
    i: int, adversarial: bool = False, gold_sql: str | None = "SELECT a FROM t1"
) -> GoldenExample:
    return GoldenExample(
        id=f"ex_{i:03d}",
        source={},
        db_id="db",
        question=f"q{i}",
        gold_sql=None if adversarial else gold_sql,
        adversarial=adversarial,
        golden_split="val",
    )


def make_record(
    example_id: str,
    repeat: int,
    *,
    match: bool | None = None,
    status: str = "ok",
    refusal_correct: bool | None = None,
    handoffs: list[str] | None = None,
    executed: list[dict] | None = None,
    events: list[dict] | None = None,
    judge_overall: float | None = None,
) -> EvalRunRecord:
    executed = executed if executed is not None else [{"sql": "SELECT a FROM t1", "ok": True}]
    final = next((a["sql"] for a in reversed(executed) if a.get("ok")), None)
    return EvalRunRecord(
        example_id=example_id,
        repeat_index=repeat,
        question="q",
        db_id="db",
        status=status,
        final_output="answer",
        events=events or [],
        executed_sql=executed,
        final_executed_sql=final,
        handoff_chain=handoffs
        if handoffs is not None
        else ["SchemaLinker", "SQLWriter", "Validator", "Summarizer"],
        execution_match=match,
        refusal_correct=refusal_correct,
        judge={"overall": judge_overall} if judge_overall is not None else None,
        wall_seconds=10.0,
        input_tokens=1000,
        output_tokens=100,
        requests=8,
    )


class TestPercentile:
    def test_interpolation(self):
        assert percentile([0, 1], 0.5) == 0.5
        assert percentile([1, 2, 3, 4], 0.5) == 2.5
        assert percentile([5], 0.9) == 5

    def test_binary_repeats_boundaries(self):
        # 1 success in 4 repeats: P90 interpolates near the top of the sorted
        # distribution -> high "best case"; P10 stays at 0.
        scores = [0.0, 0.0, 0.0, 1.0]
        assert percentile(scores, 0.9) > 0.5
        assert percentile(scores, 0.1) == 0.0


class TestTablesInSql:
    def test_simple_and_join(self):
        assert tables_in_sql("SELECT a FROM t1") == {"t1"}
        assert tables_in_sql("SELECT * FROM t1 JOIN t2 ON t1.id=t2.id") == {"t1", "t2"}

    def test_cte_not_counted_as_table(self):
        sql = "WITH c AS (SELECT a FROM t1) SELECT * FROM c JOIN t2 ON 1=1"
        assert tables_in_sql(sql) == {"t1", "t2"}

    def test_unparseable(self):
        assert tables_in_sql("not sql (") == set()


class TestReliabilityDecomposition:
    def test_perfectly_reliable_system(self):
        examples = [make_example(i) for i in range(4)]
        records = [make_record(ex.id, rep, match=True) for ex in examples for rep in range(4)]
        m = compute_metrics(records, examples, "val", 4)
        assert m.a_mean.value == 1.0
        assert m.u90.value == 0.0

    def test_unreliable_system_high_u90(self):
        # Every instance flips: 2 successes, 2 failures out of 4 repeats.
        examples = [make_example(i) for i in range(4)]
        records = []
        for ex in examples:
            for rep in range(4):
                records.append(make_record(ex.id, rep, match=rep % 2 == 0))
        m = compute_metrics(records, examples, "val", 4)
        assert m.a_mean.value == 0.5
        assert m.a90.value == 1.0  # every instance succeeds sometimes
        assert m.a10.value == 0.0  # and fails sometimes
        assert m.u90.value == 1.0  # maximal unreliability

    def test_consistent_half_split_zero_u90(self):
        # Half the instances always succeed, half always fail: same A_mean=0.5
        # as above but U90 = 0 — the decomposition separates these two worlds.
        examples = [make_example(i) for i in range(4)]
        records = []
        for i, ex in enumerate(examples):
            for rep in range(4):
                records.append(make_record(ex.id, rep, match=i % 2 == 0))
        m = compute_metrics(records, examples, "val", 4)
        assert m.a_mean.value == 0.5
        assert m.u90.value == 0.0

    def test_ci_bounds_contain_mean(self):
        examples = [make_example(i) for i in range(10)]
        records = [
            make_record(ex.id, rep, match=(i + rep) % 3 != 0)
            for i, ex in enumerate(examples)
            for rep in range(4)
        ]
        m = compute_metrics(records, examples, "val", 4)
        assert m.a_mean.ci_low <= m.a_mean.value <= m.a_mean.ci_high


class TestGuardrailMetrics:
    def test_refusal_and_false_refusal(self):
        normal = make_example(0)
        probe = make_example(1, adversarial=True)
        records = [
            make_record(normal.id, 0, match=True),
            make_record(normal.id, 1, status="guardrail_refused", match=False),
            make_record(probe.id, 0, status="guardrail_refused", refusal_correct=True),
            make_record(probe.id, 1, status="ok", refusal_correct=False),
        ]
        m = compute_metrics(records, [normal, probe], "val", 2)
        assert m.refusal_accuracy.value == 0.5
        assert m.false_refusal_rate.value == 0.5


class TestToolAndHandoffMetrics:
    def test_schema_grounding(self):
        ex = make_example(0, gold_sql="SELECT * FROM t1 JOIN t2 ON 1=1")
        events = [
            {
                "kind": "tool_call",
                "agent": "SchemaLinker",
                "payload": {"tool": "describe_table", "arguments": '{"table": "t1"}'},
            },
            {
                "kind": "tool_call",
                "agent": "SchemaLinker",
                "payload": {"tool": "describe_table", "arguments": '{"table": "t9"}'},
            },
        ]
        record = make_record(ex.id, 0, match=True, events=events)
        assert described_tables(record) == {"t1", "t9"}
        m = compute_metrics([record], [ex], "val", 1)
        assert m.schema_grounding_precision.value == 0.5  # t1 of {t1,t9}
        assert m.schema_grounding_recall.value == 0.5  # t1 of {t1,t2}

    def test_wasted_calls_and_retry_productivity(self):
        ex = make_example(0)
        executed = [
            {"sql": "SELECT bad", "ok": False, "error": "no column"},
            {"sql": "SELECT a FROM t1", "ok": True},
        ]
        record = make_record(ex.id, 0, match=True, executed=executed)
        m = compute_metrics([record], [ex], "val", 1)
        assert m.wasted_call_rate.value == 0.5
        assert m.retry_productivity.value == 1.0

    def test_ping_pong_and_routing(self):
        ex = make_example(0)
        bounced = ["SchemaLinker", "SQLWriter", "Validator", "SQLWriter", "Validator", "Summarizer"]
        record = make_record(ex.id, 0, match=True, handoffs=bounced)
        m = compute_metrics([record], [ex], "val", 1)
        assert m.mean_ping_pong.value == 2.0
        assert m.routing_accuracy.value == 1.0
        assert m.completion_rate.value == 1.0

    def test_incomplete_run(self):
        ex = make_example(0)
        record = make_record(
            ex.id,
            0,
            match=False,
            status="max_turns",
            handoffs=["SchemaLinker", "SQLWriter"],
            executed=[],
        )
        m = compute_metrics([record], [ex], "val", 1)
        assert m.completion_rate.value == 0.0


class TestJudgeAggregation:
    def test_judge_mean_and_agreement(self):
        examples = [make_example(i) for i in range(2)]
        records = [
            make_record(examples[0].id, 0, match=True, judge_overall=0.9),
            make_record(examples[1].id, 0, match=False, judge_overall=0.4),
        ]
        m = compute_metrics(records, examples, "val", 1)
        assert abs(m.judge_mean.value - 0.65) < 1e-9
        assert m.judge_exec_agreement == 1.0  # high↔match, low↔mismatch
