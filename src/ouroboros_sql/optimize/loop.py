"""The outer self-improvement loop.

One iteration:
    eval(train) -> failure report -> analysis -> optimizer PatchSet ->
    apply -> eval(val) -> accept / rollback

Accept rule (decided before any experiment ran, never tuned afterwards):
    accept iff  A_mean(val) improves by >= +1.0 point
           or   (A_mean drops <= 0.5 point AND U90 improves by >= 2.0 points)
Safety brake: reject outright if the false-refusal rate more than doubles
and worsens by > 5 points — a "fix" that refuses questions is not a fix.

Everything an iteration did — patchset, diffs, decision, metrics, memory
snapshot — lands in iterations/<k>/ so the whole learning curve is auditable.
"""

import datetime
import json
from pathlib import Path

from pydantic import BaseModel

from ..agents.memory import StrategyMemory
from ..config import REPO_ROOT
from ..eval.harness import run_eval
from ..eval.report_agent import build_eval_report, write_failure_analysis
from ..eval.schema import EvalMetrics
from .optimizer_agent import propose_patchset
from .patches import GrowthCapExceeded, apply_patchset, rollback

ITERATIONS_DIR = REPO_ROOT / "iterations"

ACCEPT_A_MEAN_GAIN = 0.010
ACCEPT_A_MEAN_TOLERANCE = 0.005
ACCEPT_U90_GAIN = 0.020
FALSE_REFUSAL_FACTOR = 2.0
FALSE_REFUSAL_ABS = 0.05


class Decision(BaseModel):
    accepted: bool
    reason: str
    a_mean_ref: float
    a_mean_new: float
    u90_ref: float
    u90_new: float


def decide(ref: EvalMetrics, candidate: EvalMetrics) -> Decision:
    d_a = candidate.a_mean.value - ref.a_mean.value
    d_u = ref.u90.value - candidate.u90.value  # positive = more reliable

    ref_fr = ref.false_refusal_rate.value if ref.false_refusal_rate else 0.0
    new_fr = candidate.false_refusal_rate.value if candidate.false_refusal_rate else 0.0
    if new_fr > ref_fr * FALSE_REFUSAL_FACTOR and new_fr - ref_fr > FALSE_REFUSAL_ABS:
        return Decision(
            accepted=False,
            reason=f"safety brake: false-refusal rate {ref_fr:.1%} -> {new_fr:.1%}",
            a_mean_ref=ref.a_mean.value,
            a_mean_new=candidate.a_mean.value,
            u90_ref=ref.u90.value,
            u90_new=candidate.u90.value,
        )

    if d_a >= ACCEPT_A_MEAN_GAIN:
        accepted, reason = True, f"A_mean +{d_a * 100:.1f} pts"
    elif d_a >= -ACCEPT_A_MEAN_TOLERANCE and d_u >= ACCEPT_U90_GAIN:
        accepted, reason = True, f"U90 -{d_u * 100:.1f} pts at A_mean {d_a * 100:+.1f}"
    else:
        accepted, reason = False, f"A_mean {d_a * 100:+.1f}, U90 {-d_u * 100:+.1f} — below bar"
    return Decision(
        accepted=accepted,
        reason=reason,
        a_mean_ref=ref.a_mean.value,
        a_mean_new=candidate.a_mean.value,
        u90_ref=ref.u90.value,
        u90_new=candidate.u90.value,
    )


class LoopConfig(BaseModel):
    max_iterations: int = 3
    stop_after_rejections: int = 2
    train_limit: int | None = 40
    train_repeats: int = 2
    val_repeats: int = 4
    concurrency: int = 8


async def run_loop(
    config: LoopConfig,
    ref_metrics: EvalMetrics,
    *,
    start_iteration: int = 1,
    iterations_dir: Path = ITERATIONS_DIR,
    progress: bool = True,
) -> list[Decision]:
    """Run up to max_iterations. `ref_metrics` is the current-state val metrics
    (iteration 0); it advances only on accepted iterations."""
    stamp = datetime.date.today().isoformat()
    decisions: list[Decision] = []
    consecutive_rejections = 0

    for k in range(start_iteration, start_iteration + config.max_iterations):
        it_dir = iterations_dir / f"{k:02d}"
        it_dir.mkdir(parents=True, exist_ok=True)
        log = (lambda msg: print(f"[iter {k}] {msg}", flush=True)) if progress else (lambda _: None)

        # 1. Observe: fresh failures from train under the current state.
        log(f"train eval ({config.train_limit} x {config.train_repeats})...")
        _, train_dir = await run_eval(
            "train",
            repeats=config.train_repeats,
            concurrency=config.concurrency,
            limit=config.train_limit,
            with_judge=False,
            run_id=f"loop-{stamp}-it{k:02d}-train",
            progress=False,
        )

        # 2. Diagnose.
        log("failure analysis...")
        report = build_eval_report(train_dir)
        analysis = await write_failure_analysis(report)
        (it_dir / "report.json").write_text(report.model_dump_json(indent=2))
        (it_dir / "analysis.json").write_text(analysis.model_dump_json(indent=2))

        # 3. Propose.
        log("optimizer proposing patchset...")
        memory = StrategyMemory.load()
        patchset = await propose_patchset(report, analysis, memory)
        (it_dir / "patchset.json").write_text(patchset.model_dump_json(indent=2))
        if patchset.is_empty:
            log("optimizer proposed no changes; stopping.")
            break

        # 4. Apply (bounded), keeping the snapshot for rollback.
        try:
            snapshot, diffs = apply_patchset(patchset)
        except (GrowthCapExceeded, ValueError) as e:
            log(f"patchset rejected by bounds: {e}")
            (it_dir / "decision.json").write_text(
                json.dumps({"accepted": False, "reason": f"bounds: {e}"}, indent=2)
            )
            consecutive_rejections += 1
            if consecutive_rejections >= config.stop_after_rejections:
                break
            continue
        for name, diff in diffs.items():
            safe = name.replace(":", "_")
            (it_dir / f"diff_{safe}.patch").write_text(diff)

        # 5. Gate on val.
        log(f"val eval (full x {config.val_repeats})...")
        candidate, val_dir = await run_eval(
            "val",
            repeats=config.val_repeats,
            concurrency=config.concurrency,
            with_judge=False,
            run_id=f"loop-{stamp}-it{k:02d}-val",
            progress=False,
        )
        (it_dir / "val_metrics.json").write_text(candidate.model_dump_json(indent=2))

        # 6. Decide.
        decision = decide(ref_metrics, candidate)
        (it_dir / "decision.json").write_text(decision.model_dump_json(indent=2))
        decisions.append(decision)
        log(f"{'ACCEPTED' if decision.accepted else 'REJECTED'}: {decision.reason}")

        if decision.accepted:
            ref_metrics = candidate
            consecutive_rejections = 0
            (it_dir / "memory_after.json").write_text(
                StrategyMemory.load().model_dump_json(indent=2)
            )
        else:
            rollback(snapshot)
            consecutive_rejections += 1
            if consecutive_rejections >= config.stop_after_rejections:
                log(f"{consecutive_rejections} consecutive rejections; stopping.")
                break

    return decisions
