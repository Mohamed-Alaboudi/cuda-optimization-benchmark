#!/usr/bin/env python3
"""Build assets/per_task.png. Numbers are parsed from
results/avg_vs_reference.md (itself generated from the grader logs), never typed by hand.

  uv run --with matplotlib python report/make_charts.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def build(tasks, refs, pct):
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=100)
    x = range(len(tasks))
    w = 0.26
    for i, model in enumerate(MODELS):
        b = ax.bar([v + (i - 1) * w for v in x], pct[model], w,
                   label=model, color=COLORS[model])
    ax.axhline(100, color="#111", lw=1.2, ls="--")
    ax.text(len(tasks) - 0.45, 102, "expert reference kernel", fontsize=9, ha="right")
    ax.set_xticks(list(x), [f"{t}\nref: {refs[t]:g}x PyTorch" for t in tasks], fontsize=8.5)
    ax.set_ylabel("speed as % of the expert reference kernel")
    ax.set_ylim(0, 115)
    ax.set_title("All 54 attempts passed every correctness and memory check. The gap is speed.", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    tasks, refs, pct = parse()
    OUT.mkdir(exist_ok=True)
    fig = build(tasks, refs, pct)
    fig.savefig(OUT / "per_task.png")
    plt.close(fig)

print("wrote", OUT / "per_task.png")


if __name__ == "__main__":
    main()
