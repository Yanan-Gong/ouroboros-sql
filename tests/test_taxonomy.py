from ouroboros_sql.eval.taxonomy import classify, taxonomy_counts
from test_metrics import make_example, make_record


def test_success_is_unlabeled():
    ex = make_example(0)
    assert classify(make_record(ex.id, 0, match=True), ex) is None


def test_guardrail_missed_on_adversarial():
    ex = make_example(0, adversarial=True)
    assert (
        classify(make_record(ex.id, 0, status="ok", refusal_correct=False), ex)
        == "guardrail_missed"
    )
    assert (
        classify(make_record(ex.id, 0, status="guardrail_refused", refusal_correct=True), ex)
        is None
    )


def test_false_refusal_and_max_turns():
    ex = make_example(0)
    assert (
        classify(make_record(ex.id, 0, match=False, status="guardrail_refused"), ex)
        == "false_refusal"
    )
    assert classify(make_record(ex.id, 0, match=False, status="max_turns"), ex) == "max_turns"


def test_sql_failure_classes():
    ex = make_example(0, gold_sql="SELECT a FROM t1")
    no_attempts = make_record(ex.id, 0, match=False, executed=[])
    assert classify(no_attempts, ex) == "no_sql_executed"

    all_errors = make_record(
        ex.id, 0, match=False, executed=[{"sql": "SELECT bad", "ok": False, "error": "x"}]
    )
    assert classify(all_errors, ex) == "sql_never_recovered"

    wrong_tables = make_record(
        ex.id, 0, match=False, executed=[{"sql": "SELECT b FROM t2", "ok": True}]
    )
    assert classify(wrong_tables, ex) == "wrong_tables"

    wrong_result = make_record(
        ex.id, 0, match=False, executed=[{"sql": "SELECT b FROM t1", "ok": True}]
    )
    assert classify(wrong_result, ex) == "wrong_result"


def test_counts_sorted_desc():
    ex1, ex2 = make_example(1), make_example(2)
    records = [
        make_record(ex1.id, 0, match=False, status="max_turns"),
        make_record(ex1.id, 1, match=False, status="max_turns"),
        make_record(ex2.id, 0, match=False, executed=[]),
        make_record(ex2.id, 1, match=True),
    ]
    counts = taxonomy_counts(records, [ex1, ex2])
    assert counts == {"max_turns": 2, "no_sql_executed": 1}
