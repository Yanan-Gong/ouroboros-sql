#!/usr/bin/env python3
"""Render the learning curve as a dependency-free SVG.

Reads val metrics from named run directories (single source of truth — the
same committed metrics.json files behind the README tables) and draws
A_mean with CI whiskers across system states, marking rejected optimizer
candidates as open points. Output: docs/results/learning_curve.svg
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).parents[1]

# (label, run dir under runs/ or docs/results/, accepted?)
STAGES: list[tuple[str, str, bool]] = [
    ("baseline\n(no memory)", "baseline-val-v0", True),
    ("+ seeded memory\n(human)", "ablation-val-memory-v1", True),
    ("+ optimizer iter 2\n(claude, accepted)", "loop-2026-07-21-it02-val-63775d9f41", True),
]
REJECTED: list[tuple[str, str]] = [
    # (label, run dir) — drawn as open points at their sequence position
]

W, H = 760, 420
ML, MR, MT, MB = 70, 30, 40, 90
PLOT_W, PLOT_H = W - ML - MR, H - MT - MB
Y_MIN, Y_MAX = 0.40, 0.65


def load_metrics(run: str) -> dict:
    for base in (REPO / "runs", REPO / "docs" / "results"):
        p = base / run / "metrics.json"
        if p.is_file():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"metrics.json not found for {run}")


def x_at(i: int, n: int) -> float:
    return ML + PLOT_W * (i + 0.5) / n


def y_at(v: float) -> float:
    return MT + PLOT_H * (1 - (v - Y_MIN) / (Y_MAX - Y_MIN))


def main() -> None:
    stages = [(label, load_metrics(run), accepted) for label, run, accepted in STAGES]
    n = len(stages)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui" font-size="13">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{ML}" y="24" font-size="16" font-weight="600">'
        "Execution accuracy (A_mean) on val across system states</text>",
    ]
    # y grid
    v = Y_MIN
    while v <= Y_MAX + 1e-9:
        y = y_at(v)
        parts.append(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#6b7280">'
            f"{v * 100:.0f}%</text>"
        )
        v += 0.05

    # connecting line through accepted stages
    points = []
    for i, (_label, m, _acc) in enumerate(stages):
        points.append((x_at(i, n), y_at(m["a_mean"]["value"])))
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    parts.append(f'<path d="{path}" fill="none" stroke="#2563eb" stroke-width="2.5"/>')

    for i, (label, m, accepted) in enumerate(stages):
        x = x_at(i, n)
        a = m["a_mean"]
        y = y_at(a["value"])
        # CI whisker
        if a.get("ci_low") is not None:
            parts.append(
                f'<line x1="{x:.1f}" y1="{y_at(a["ci_low"]):.1f}" '
                f'x2="{x:.1f}" y2="{y_at(a["ci_high"]):.1f}" '
                'stroke="#93c5fd" stroke-width="4" stroke-linecap="round" opacity="0.8"/>'
            )
        fill = "#2563eb" if accepted else "white"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{fill}" '
            'stroke="#2563eb" stroke-width="2.5"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y - 14:.1f}" text-anchor="middle" '
            f'font-weight="600">{a["value"] * 100:.1f}</text>'
        )
        for j, line in enumerate(label.split("\n")):
            parts.append(
                f'<text x="{x:.1f}" y="{H - MB + 24 + j * 16}" text-anchor="middle" '
                f'fill="#374151">{line}</text>'
            )

    parts.append(
        f'<text x="{ML}" y="{H - 12}" fill="#9ca3af" font-size="11">'
        "Whiskers: bootstrap 95% CI over instances. Same protocol throughout: "
        "val split, N=4 repeats. Rejected optimizer candidates not shown; see iterations/.</text>"
    )
    parts.append("</svg>")

    out = REPO / "docs" / "results" / "learning_curve.svg"
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
