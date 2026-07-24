"""Regenerate the paper framework figure used in the README."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

FIG_W, FIG_H = 11.0, 3.25
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 13.0)
ax.set_ylim(0, 4.15)
ax.axis("off")

C_IO    = "#ECECEC"   # inputs
C_STAGE = "#C9DCEA"   # VLM / LLM stages
C_KNOW  = "#F2DFAE"   # knowledge repository
C_EDIT  = "#C3DEC3"   # editing backbone
EDGE    = "#2F2F2F"

def box(x, y, w, h, lines, color, sizes=None, weights=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor=color, edgecolor=EDGE, linewidth=1.1, zorder=3))
    n = len(lines)
    total = sum((sizes or [9.5] * n))
    yc = y + h / 2
    step = h / (n + 0.6)
    start = yc + step * (n - 1) / 2
    for i, txt in enumerate(lines):
        ax.text(x + w / 2, start - i * step, txt,
                ha="center", va="center", zorder=4,
                fontsize=(sizes or [9.5] * n)[i],
                fontweight=(weights or ["normal"] * n)[i])
    return (x, y, w, h)

def arrow(p1, p2, style="-|>", rad=0.0, lw=1.4, color="#4A4A4A"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=13,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw, color=color, zorder=2,
        shrinkA=0, shrinkB=0))

# ── nodes ────────────────────────────────────────────────────────────────
IMG   = box(0.20, 2.38, 1.75, 1.05, ["Input image", "$I$"], C_IO,
            sizes=[9.5, 10.5], weights=["normal", "bold"])
REQ   = box(0.20, 0.62, 1.75, 1.05, ["User request $q$", "“make it rainy”"], C_IO,
            sizes=[9.5, 8.3], weights=["bold", "normal"])

VLM   = box(2.70, 2.38, 2.45, 1.05,
            ["Stage 1", "Vision–language model", "scene description $s$"], C_STAGE,
            sizes=[8.6, 9.3, 9.3], weights=["bold", "normal", "normal"])
REPO  = box(2.70, 0.62, 2.45, 1.05,
            ["Stage 2", "Structured knowledge", "cue retrieval $\\rightarrow$ $C$"], C_KNOW,
            sizes=[8.6, 9.3, 9.3], weights=["bold", "normal", "normal"])

LLM   = box(6.30, 1.50, 2.55, 1.05,
            ["Stage 3", "LLM instruction", "generation"], C_STAGE,
            sizes=[8.6, 9.3, 9.3], weights=["bold", "normal", "normal"])

EDITR = box(10.00, 1.50, 2.60, 1.05,
            ["Editing backbone", "FLUX Kontext / IP2P"], C_EDIT,
            sizes=[9.5, 8.5], weights=["bold", "normal"])

# ── edges ────────────────────────────────────────────────────────────────
arrow((1.95, 2.905), (2.70, 2.905))                    # I -> VLM
arrow((1.95, 1.145), (2.70, 1.145))                    # q -> repo
arrow((5.15, 2.75), (6.30, 2.32), rad=-0.06)           # s -> LLM
arrow((5.15, 1.30), (6.30, 1.73), rad=0.06)            # C -> LLM
arrow((8.85, 2.025), (10.00, 2.025))                   # LLM -> editor

# q -> LLM, routed underneath so nothing crosses
ax.plot([1.075, 1.075, 7.575], [0.62, 0.22, 0.22],
        color="#4A4A4A", linewidth=1.4, zorder=1,
        solid_capstyle="round")
arrow((7.575, 0.22), (7.575, 1.50))
ax.text(4.3, 0.05, "user request also conditions instruction generation",
        ha="center", va="bottom", fontsize=7.6, color="#5A5A5A", style="italic")

# edge labels
ax.text(5.72, 2.63, "$s$", fontsize=9, color="#3A3A3A")
ax.text(5.72, 1.30, "$C$", fontsize=9, color="#3A3A3A")
ax.text(9.42, 2.13, "editing\ninstruction", fontsize=7.8, ha="center",
        va="bottom", color="#3A3A3A")

ax.text(6.5, 3.90, "All stages run at inference time — no component is fine-tuned",
        ha="center", fontsize=9, style="italic", color="#5A5A5A")

plt.tight_layout(pad=0.25)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("assets/framework.png"))
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Wrote {args.output}")
