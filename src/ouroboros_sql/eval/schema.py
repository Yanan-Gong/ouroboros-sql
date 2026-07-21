"""Data contracts for the evaluation layer.

Everything the harness writes and the metrics/report layers read is one of
these models — no ad-hoc dicts crossing module boundaries.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config import settings


class GoldenExample(BaseModel):
    id: str
    source: dict[str, Any]
    db_id: str
    question: str
    evidence: str | None = None
    gold_sql: str | None = None
    difficulty: str = "unknown"
    order_matters: bool = False
    adversarial: bool = False
    golden_split: str

    def question_with_evidence(self) -> str:
        """BIRD's `evidence` field is domain knowledge the question assumes;
        benchmarks provide it to the system, so we append it to the input."""
        if self.evidence:
            return f"{self.question}\n(Domain note: {self.evidence})"
        return self.question


def load_split(split: Literal["train", "val", "holdout"]) -> list[GoldenExample]:
    path = settings.golden_dir / f"{split}.json"
    with path.open() as f:
        return [GoldenExample.model_validate(ex) for ex in json.load(f)]


class EvalRunRecord(BaseModel):
    """One (example, repeat) execution of the pipeline, fully serialized."""

    example_id: str
    repeat_index: int
    question: str
    db_id: str
    status: str
    final_output: str
    events: list[dict[str, Any]] = Field(default_factory=list)
    executed_sql: list[dict[str, Any]] = Field(default_factory=list)
    final_executed_sql: str | None = None
    handoff_chain: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    wall_seconds: float = 0.0
    # Filled by scoring:
    execution_match: bool | None = None  # None for adversarial examples
    refusal_correct: bool | None = None  # None for non-adversarial examples
    judge: dict[str, Any] | None = None
    error: str | None = None  # harness-level exception, if any


class MetricValue(BaseModel):
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int = 0

    def fmt(self, pct: bool = True) -> str:
        scale = 100 if pct else 1
        s = f"{self.value * scale:.1f}"
        if self.ci_low is not None and self.ci_high is not None:
            s += f" [{self.ci_low * scale:.1f}, {self.ci_high * scale:.1f}]"
        return s


class EvalMetrics(BaseModel):
    """Aggregated metrics for one eval run."""

    split: str
    n_examples: int
    n_adversarial: int
    n_repeats: int
    n_records: int
    n_harness_errors: int

    # Reliability decomposition over execution accuracy (binary per repeat).
    a_mean: MetricValue
    a90: MetricValue  # aptitude: per-instance 90th percentile, averaged
    a10: MetricValue
    u90: MetricValue  # unreliability: a90 - a10

    # Judge-score decomposition (continuous), present when the judge ran.
    judge_mean: MetricValue | None = None
    judge_u90: MetricValue | None = None
    judge_exec_agreement: float | None = None

    # Guardrails (adversarial probes only).
    refusal_accuracy: MetricValue | None = None
    false_refusal_rate: MetricValue | None = None  # non-adversarial refused

    # Tool usage (non-adversarial records).
    schema_grounding_precision: MetricValue | None = None
    schema_grounding_recall: MetricValue | None = None
    wasted_call_rate: MetricValue | None = None
    retry_productivity: MetricValue | None = None

    # Handoffs (non-adversarial records).
    routing_accuracy: MetricValue | None = None
    completion_rate: MetricValue | None = None
    mean_ping_pong: MetricValue | None = None

    # Cost & latency.
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    mean_requests: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    cost_per_100_usd: float | None = None


class RunMetadata(BaseModel):
    run_id: str
    split: str
    n_repeats: int
    limit: int | None
    agent_model: str
    judge_model: str | None
    base_url_host: str | None
    seed: int
    memory_entries: int = 0
    started_at: str
    package_version: str


def append_jsonl(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[EvalRunRecord]:
    if not path.is_file():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EvalRunRecord.model_validate_json(line))
    return records
