"""Deterministic half of the report pipeline (no model calls)."""

import json
from pathlib import Path

from ouroboros_sql.eval.report_agent import (
    ATTRIBUTION,
    build_eval_report,
    render_report_for_analyst,
)
from ouroboros_sql.eval.schema import append_jsonl
from ouroboros_sql.eval.taxonomy import FAILURE_CLASSES
from test_metrics import make_example, make_record


def test_attribution_covers_all_failure_classes():
    assert set(ATTRIBUTION) == set(FAILURE_CLASSES)


def test_build_eval_report(tmp_path: Path, monkeypatch):
    # Fake a run dir with mixed outcomes over the real val split's first ids.
    from ouroboros_sql.eval import schema as schema_mod

    examples = [make_example(i) for i in range(4)]
    monkeypatch.setattr(schema_mod, "load_split", lambda split: examples)
    import ouroboros_sql.eval.report_agent as ra

    monkeypatch.setattr(ra, "load_split", lambda split: examples)

    run_dir = tmp_path / "myrun"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps({"split": "val"}))

    records = [
        make_record(examples[0].id, 0, match=True),
        make_record(examples[0].id, 1, match=False, status="max_turns"),
        make_record(examples[1].id, 0, match=False, status="max_turns"),
        make_record(
            examples[2].id,
            0,
            match=False,
            executed=[{"sql": "SELECT b FROM t2", "ok": True}],
        ),
    ]
    for r in records:
        append_jsonl(run_dir / "records.jsonl", r)

    report = build_eval_report(run_dir)
    assert report.n_records == 4
    assert report.n_failures == 3
    classes = {fc.taxonomy: fc for fc in report.failure_classes}
    assert classes["max_turns"].count == 2
    assert classes["max_turns"].suspected_agent == "orchestrator"
    assert classes["wrong_tables"].count == 1
    # Exemplars are distinct examples, not repeats of one.
    assert len(set(classes["max_turns"].exemplar_ids)) == 2

    text = render_report_for_analyst(report)
    assert "## max_turns (count=2" in text
    assert "exemplar 1" in text
