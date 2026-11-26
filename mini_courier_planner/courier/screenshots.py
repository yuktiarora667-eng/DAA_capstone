"""Render console output as terminal-style PNG images (evidence for the report)."""
import os
from typing import List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:                   # pragma: no cover
    HAVE_MPL = False


def text_to_png(text: str, path: str, title: str = "Mini-Courier Planner", max_lines: int = 60) -> str:
    lines: List[str] = text.strip("\n").split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines in the .txt trace)"]
    width = max(len(l) for l in lines)
    fig_w = max(8.0, min(width * 0.095 + 0.6, 16))
    fig_h = len(lines) * 0.148 + 0.95
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0f172a")
    fig.text(0.012, 1 - 0.30 / fig_h, "o o o   " + title, color="#94a3b8", family="monospace", fontsize=8, va="center")
    fig.text(0.012, 1 - 0.75 / fig_h, "\n".join(lines), color="#e2e8f0", family="monospace",
             fontsize=8, va="top", linespacing=1.25)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
