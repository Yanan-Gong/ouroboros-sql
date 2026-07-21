"""Offline tests for harness scoring and resumability plumbing."""

from pathlib import Path

from ouroboros_sql.db.catalog import Catalog
from ouroboros_sql.eval.harness import GoldResultCache, score_execution
from ouroboros_sql.eval.schema import (
    EvalRunRecord,
    GoldenExample,
    append_jsonl,
    read_jsonl,
)
from test_metrics import make_record


def golden(tiny_db: Path, **overrides) -> GoldenExample:
    fields = {
        "id": "g1",
        "source": {},
        "db_id": "schools",
        "question": "How many schools are there?",
        "gold_sql": "SELECT COUNT(*) FROM schools",
        "golden_split": "val",
    }
    fields.update(overrides)
    return GoldenExample.model_validate(fields)


def test_score_execution_match(tiny_db: Path):
    catalog = Catalog(tiny_db.parent)
    cache = GoldResultCache(catalog)
    ex = golden(tiny_db)

    # Semantically equal query, different surface form.
    record = make_record(
        ex.id, 0, executed=[{"sql": "SELECT COUNT(school_id) FROM schools", "ok": True}]
    )
    record.db_id = "schools"
    score_execution(record, ex, cache, catalog)
    assert record.execution_match is True

    wrong = make_record(ex.id, 1, executed=[{"sql": "SELECT COUNT(*) FROM districts", "ok": True}])
    wrong.db_id = "schools"
    score_execution(wrong, ex, cache, catalog)
    assert wrong.execution_match is False


def test_score_execution_no_sql_or_bad_sql(tiny_db: Path):
    catalog = Catalog(tiny_db.parent)
    cache = GoldResultCache(catalog)
    ex = golden(tiny_db)

    none_run = make_record(ex.id, 0, executed=[])
    score_execution(none_run, ex, cache, catalog)
    assert none_run.execution_match is False

    # final SQL references a missing column: scored as mismatch, no crash
    broken = make_record(ex.id, 1, executed=[{"sql": "SELECT nope FROM schools", "ok": True}])
    broken.db_id = "schools"
    score_execution(broken, ex, cache, catalog)
    assert broken.execution_match is False


def test_score_adversarial(tiny_db: Path):
    catalog = Catalog(tiny_db.parent)
    cache = GoldResultCache(catalog)
    probe = golden(tiny_db, id="p1", gold_sql=None, adversarial=True)

    refused = make_record(probe.id, 0, status="guardrail_refused", executed=[])
    score_execution(refused, probe, cache, catalog)
    assert refused.refusal_correct is True
    assert refused.execution_match is None

    answered = make_record(probe.id, 1, status="ok")
    score_execution(answered, probe, cache, catalog)
    assert answered.refusal_correct is False


def test_jsonl_roundtrip_and_resume_key(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    for i in range(3):
        append_jsonl(path, make_record("ex_a", i, match=True))
    records = read_jsonl(path)
    assert len(records) == 3
    assert all(isinstance(r, EvalRunRecord) for r in records)
    done = {(r.example_id, r.repeat_index) for r in records}
    assert done == {("ex_a", 0), ("ex_a", 1), ("ex_a", 2)}
    assert read_jsonl(tmp_path / "missing.jsonl") == []
