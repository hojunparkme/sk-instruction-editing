"""Fig. 2 - shared-reference CLIPdir bar charts (PNG 300 dpi + vector PDF)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

FLUX = {"Simple": (0.158, 0.001), "LLM-only": (0.158, 0.003),
        "SK+Filter": (0.181, 0.002), "SK+LLM\n(ours)": (0.183, 0.001)}
IP2P = {"Simple": 0.088, "MGIE": 0.095, "LLM-only": 0.099,
        "SK+Filter": 0.091, "SK+LLM\n(ours)": 0.095}
C_FLUX = ["#B0B0B0", "#88A5C0", "#E0A860", "#4A7BA6"]
C_IP2P = ["#B0B0B0", "#C08888", "#88A5C0", "#E0A860", "#4A7BA6"]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

ax = axes[0]
b = ax.bar(list(FLUX), [v for v, _ in FLUX.values()],
           yerr=[e for _, e in FLUX.values()], capsize=4,
           color=C_FLUX, edgecolor="#333", linewidth=0.9)
ax.set_title("(a) FLUX Kontext", fontsize=12, fontweight="bold")
for bar, (v, _) in zip(b, FLUX.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006, f"{v:.3f}",
            ha="center", fontsize=9.5)
ax.axhline(y=list(FLUX.values())[0][0], color="#999", ls="--", lw=0.8, alpha=0.6)

ax = axes[1]
b = ax.bar(list(IP2P), list(IP2P.values()), color=C_IP2P,
           edgecolor="#333", linewidth=0.9)
ax.set_title("(b) InstructPix2Pix", fontsize=12, fontweight="bold")
for bar, v in zip(b, IP2P.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006, f"{v:.3f}",
            ha="center", fontsize=9)
ax.axhline(y=list(IP2P.values())[0], color="#999", ls="--", lw=0.8, alpha=0.6)
ax.tick_params(labelsize=9)

for a in axes:
    a.set_ylabel("Common CLIPdir", fontsize=11)
    a.set_ylim(0, 0.22)

plt.tight_layout(pad=0.4)
fig.savefig("fig2_clipdir.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("fig2_clipdir.pdf", bbox_inches="tight", facecolor="white")
print("wrote fig2_clipdir.png / .pdf")
