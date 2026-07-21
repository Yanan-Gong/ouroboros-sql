"""The accept/reject gate — pure decision logic, exhaustively testable."""

from ouroboros_sql.eval.schema import EvalMetrics, MetricValue
from ouroboros_sql.optimize.loop import decide


def metrics(a_mean: float, u90: float, false_refusal: float = 0.0) -> EvalMetrics:
    mv = lambda v: MetricValue(value=v, n=60)  # noqa: E731
    return EvalMetrics(
        split="val",
        n_examples=60,
        n_adversarial=3,
        n_repeats=4,
        n_records=252,
        n_harness_errors=0,
        a_mean=mv(a_mean),
        a90=mv(a_mean),
        a10=mv(a_mean),
        u90=mv(u90),
        false_refusal_rate=mv(false_refusal),
    )


REF = metrics(a_mean=0.538, u90=0.232)


def test_accept_on_accuracy_gain():
    d = decide(REF, metrics(a_mean=0.550, u90=0.240))
    assert d.accepted and "A_mean +1.2" in d.reason


def test_reject_below_accuracy_bar():
    assert not decide(REF, metrics(a_mean=0.545, u90=0.232)).accepted  # +0.7 < +1.0


def test_accept_on_reliability_gain_with_flat_accuracy():
    d = decide(REF, metrics(a_mean=0.535, u90=0.205))  # -0.3 A_mean, -2.7 U90
    assert d.accepted and "U90" in d.reason


def test_reject_reliability_gain_with_too_much_accuracy_loss():
    assert not decide(REF, metrics(a_mean=0.525, u90=0.190)).accepted  # -1.3 A_mean


def test_reject_pure_regression():
    d = decide(REF, metrics(a_mean=0.500, u90=0.260))
    assert not d.accepted and "below bar" in d.reason


def test_safety_brake_on_false_refusals():
    # Accuracy improved, but the system started refusing real questions.
    d = decide(REF, metrics(a_mean=0.580, u90=0.200, false_refusal=0.12))
    assert not d.accepted and "safety brake" in d.reason


def test_small_false_refusal_change_does_not_trip_brake():
    d = decide(
        metrics(a_mean=0.538, u90=0.232, false_refusal=0.02),
        metrics(a_mean=0.560, u90=0.232, false_refusal=0.03),
    )
    assert d.accepted
