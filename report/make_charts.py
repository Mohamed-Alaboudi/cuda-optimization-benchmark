#!/usr/bin/env python3
"""Build assets/per_task.png and assets/results.gif. Numbers are parsed from
results/avg_vs_reference.md (itself generated from the grader logs), never typed by hand.

  uv run --with matplotlib python report/make_charts.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
AVG = ROOT / "results" / "avg_vs_reference.md"
OUT = ROOT / "assets"

MODELS = ["gpt-6.1-sol", "claude-opus-5-5", "kimi-k3"]
COLORS = {"gpt-6.1-sol": "#2E9E6B", "claude-opus-5-5": "#D97757", "kimi-k3": "#7C6BD6"}


def parse():
    """Parse '| task | REFx | SOLx (P%) | OPUSx (P%) | KIMIx (P%) |' rows."""
    tasks, refs, pct = [], {}, {m: [] for m in MODELS}
    pat = re.compile(r"^\| (\w+) \| ([\d.]+)x \| [\d.]+x \((\d+)%\) \| [\d.]+x \((\d+)%\) \| [\d.]+x \((\d+)%\) \|")
    for line in AVG.read_text().splitlines():
        m = pat.match(line)
        if m:
            tasks.append(m.group(1))
            refs[m.group(1)] = float(m.group(2))
            for model, g in zip(MODELS, [3, 4, 5]):
                pct[model].append(int(m.group(g)))
    assert len(tasks) == 6, f"expected 6 task rows, got {tasks}"
    return tasks, refs, pct


def build(tasks, refs, pct, scale):
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=100)
    x = range(len(tasks))
    w = 0.26
    bars = []
    for i, model in enumerate(MODELS):
        b = ax.bar([v + (i - 1) * w for v in x], [p * scale for p in pct[model]], w,
                   label=model, color=COLORS[model])
        bars.append((b, pct[model]))
    ax.axhline(100, color="#111", lw=1.2, ls="--")
    ax.text(len(tasks) - 0.45, 102, "expert reference kernel", fontsize=9, ha="right")
    ax.set_xticks(list(x), [f"{t}\nref: {refs[t]:g}x PyTorch" for t in tasks], fontsize=8.5)
    ax.set_ylabel("speed as % of the expert reference kernel")
    ax.set_ylim(0, 115)
    ax.set_title("All 54 attempts passed every correctness and memory check. The gap is speed.", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, bars


def main():
    tasks, refs, pct = parse()
    OUT.mkdir(exist_ok=True)
    fig, _ = build(tasks, refs, pct, 1.0)
    fig.savefig(OUT / "per_task.png")
    plt.close(fig)

    fig, bars = build(tasks, refs, pct, 0.0)
    grow, hold = 25, 10

    def frame(f):
        s = min(f / (grow - 1), 1.0)
        s = 1 - (1 - s) ** 2  # ease-out
        for b, vals in bars:
            for rect, v in zip(b, vals):
                rect.set_height(v * s)
        return [r for b, _ in bars for r in b]

    anim = FuncAnimation(fig, frame, frames=grow + hold, blit=False)
    anim.save(OUT / "results.gif", writer=PillowWriter(fps=15))
    print("wrote", OUT / "per_task.png", "and", OUT / "results.gif")


if __name__ == "__main__":
    main()
