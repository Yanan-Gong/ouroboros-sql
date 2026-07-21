"""The eval harness: golden set x N repeats, resumable, error-contained.

Design points:
- Every (example, repeat) pair is independent and written to JSONL as soon as
  it finishes — a crashed or interrupted run resumes by skipping pairs already
  on disk.
- A harness-level exception in one pair becomes a record with `error` set; it
  never kills the run.
- Gold SQL results are executed once per example and cached; predicted SQL is
  re-executed per record. Comparison is the same normalized-result match used
  everywhere else.
"""

import asyncio
import datetime
import uuid
from pathlib import Path

from .. import __version__
from ..agents.memory import StrategyMemory
from ..agents.topology import Pipeline, build_pipeline
from ..config import settings
from ..db.catalog import Catalog
from ..db.executor import QueryResult, execute_sql, results_match
from ..runner import run_one
from .judge import build_judge, judge_record
from .metrics import compute_metrics
from .schema import (
    EvalMetrics,
    EvalRunRecord,
    GoldenExample,
    RunMetadata,
    append_jsonl,
    load_split,
    read_jsonl,
)


class GoldResultCache:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self._cache: dict[str, QueryResult] = {}

    def get(self, example: GoldenExample) -> QueryResult:
        if example.id not in self._cache:
            assert example.gold_sql is not None
            self._cache[example.id] = execute_sql(
                self.catalog.path_for(example.db_id),
                example.gold_sql,
                timeout_seconds=settings.sql_timeout_seconds,
                row_limit=settings.sql_row_limit,
            )
        return self._cache[example.id]


def score_execution(
    record: EvalRunRecord, example: GoldenExample, gold_cache: GoldResultCache, catalog: Catalog
) -> None:
    if example.adversarial:
        record.refusal_correct = record.status == "guardrail_refused"
        return
    if record.final_executed_sql is None:
        record.execution_match = False
        return
    try:
        predicted = execute_sql(
            catalog.path_for(example.db_id),
            record.final_executed_sql,
            timeout_seconds=settings.sql_timeout_seconds,
            row_limit=settings.sql_row_limit,
        )
        record.execution_match = results_match(
            predicted, gold_cache.get(example), order_matters=example.order_matters
        )
    except Exception:
        record.execution_match = False


async def _run_pair(
    example: GoldenExample,
    repeat_index: int,
    pipeline: Pipeline,
    catalog: Catalog,
) -> EvalRunRecord:
    try:
        rr = await run_one(
            example.question_with_evidence(), example.db_id, pipeline, catalog=catalog
        )
        return EvalRunRecord(
            example_id=example.id,
            repeat_index=repeat_index,
            question=example.question,
            db_id=example.db_id,
            status=rr.status,
            final_output=rr.final_output,
            events=[{"kind": e.kind, "agent": e.agent, "payload": e.payload} for e in rr.events],
            executed_sql=rr.executed_sql,
            final_executed_sql=rr.final_executed_sql,
            handoff_chain=rr.handoff_chain,
            input_tokens=rr.input_tokens,
            output_tokens=rr.output_tokens,
            requests=rr.requests,
            wall_seconds=rr.wall_seconds,
        )
    except Exception as e:
        return EvalRunRecord(
            example_id=example.id,
            repeat_index=repeat_index,
            question=example.question,
            db_id=example.db_id,
            status="error",
            final_output="",
            error=f"{type(e).__name__}: {e}",
        )


async def run_eval(
    split: str,
    *,
    repeats: int = 8,
    concurrency: int = 8,
    limit: int | None = None,
    with_judge: bool = False,
    with_memory: bool = True,
    run_id: str | None = None,
    progress: bool = True,
) -> tuple[EvalMetrics, Path]:
    examples = load_split(split)  # type: ignore[arg-type]
    if limit is not None:
        examples = examples[:limit]

    run_id = run_id or f"{split}-{datetime.date.today().isoformat()}-{uuid.uuid4().hex[:6]}"
    run_dir = settings.runs_dir / run_id
    records_path = run_dir / "records.jsonl"

    existing = {(r.example_id, r.repeat_index) for r in read_jsonl(records_path)}
    pending = [(ex, i) for ex in examples for i in range(repeats) if (ex.id, i) not in existing]

    catalog = Catalog(settings.databases_dir)
    gold_cache = GoldResultCache(catalog)
    memory = StrategyMemory.load() if with_memory else None
    pipeline = build_pipeline(memory=memory)
    judge = build_judge(settings.judge_model) if with_judge else None

    metadata = RunMetadata(
        run_id=run_id,
        split=split,
        n_repeats=repeats,
        limit=limit,
        agent_model=str(settings.agent_model),
        judge_model=str(settings.judge_model) if with_judge else None,
        base_url_host=_base_url_host(),
        seed=0,
        memory_entries=len(memory.entries) if memory else 0,
        started_at=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        package_version=__version__,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2))

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    done_count = 0

    async def worker(ex: GoldenExample, i: int) -> None:
        nonlocal done_count
        async with semaphore:
            record = await _run_pair(ex, i, pipeline, catalog)
            score_execution(record, ex, gold_cache, catalog)
            if judge is not None and not ex.adversarial and record.error is None:
                try:
                    record.judge = await judge_record(record, judge, str(settings.judge_model))
                except Exception as e:
                    record.judge = {"error": f"{type(e).__name__}: {e}"}
            async with write_lock:
                append_jsonl(records_path, record)
                done_count += 1
                if progress and done_count % 10 == 0:
                    print(f"  {done_count}/{len(pending)} records", flush=True)

    await asyncio.gather(*(worker(ex, i) for ex, i in pending))

    records = read_jsonl(records_path)
    metrics = compute_metrics(
        records, examples, split, repeats, cost_per_million=settings.cost_per_million
    )
    (run_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    return metrics, run_dir


def _base_url_host() -> str | None:
    import os
    from urllib.parse import urlparse

    url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE_URL")
    return urlparse(url).netloc if url else None
