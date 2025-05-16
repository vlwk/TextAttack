import matplotlib.pyplot as plt
import numpy as np

budgets = [0, 1, 5]
perturbations = ["homoglyphs", "invisible", "deletions", "reorderings", "clean"]
models = ["Bhadresh", "GPT-4o", "Gemini 2F", "Bhadresh-ft", "Bhadresh-ft-enc", "T5-Bhadresh"]
colours = {
    "Bhadresh": "#1f77b4",
    "GPT-4o": "#ff7f0e",
    "Gemini 2F": "#2ca02c",
    "Bhadresh-ft": "#9467bd",
    "Bhadresh-ft-enc": "#d62728",
    "T5-Bhadresh": "#8c564b"
}

data = {
    "homoglyphs": [0.742, 0.578, 0.5285, 0.8095, 0.8030, 0.7875],
    "invisible":  [0.7695, 0.5795, 0.5320, 0.8435, 0.8340, 0.8140],
    "deletions":  [0.7695, 0.5755, 0.4845, 0.8435, 0.8340, 0.8140],
    "reorderings": [0.7565, 0.5695, 0.5190, 0.8510, 0.8390, 0.7740],
    "clean": [0.9270, 0.5760, 0.5425, 0.9245, 0.9250, 0.9270]
}

x = np.arange(len(models))  # model positions
bar_width = 0.2

fig, axs = plt.subplots(1, 5, figsize=(22, 6), sharey=True)

for i, perturb in enumerate(perturbations):
    ax = axs[i]
    values = data[perturb]
    ax.bar(x, values, color=[colours[m] for m in models])
    ax.set_title(perturb.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    if i == 0:
        ax.set_ylabel("Percentage classified correctly")
    ax.grid(axis='y')

fig.suptitle("Percentage of examples classified correctly by Perturbation Type")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("all_sc_o.png", dpi=300, bbox_inches='tight')