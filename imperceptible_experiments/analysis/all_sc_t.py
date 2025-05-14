import matplotlib.pyplot as plt
import numpy as np

# Define data
budgets = [0, 1, 5]
perturbations = ["homoglyphs", "invisible", "deletions", "reorderings"]
models = ["Bhadresh", "GPT-4o", "Gemini 2F", "Bhadresh-ft", "Bhadresh-ft-enc", "T5-Bhadresh"]
colours = {
    "Bhadresh": "#1f77b4",        # muted blue
    "GPT-4o": "#ff7f0e",          # orange
    "Gemini 2F": "#2ca02c",       # green
    "Bhadresh-ft": "#9467bd",     # purple
    "Bhadresh-ft-enc": "#d62728", # red
    "T5-Bhadresh": "#8c564b"      # brown
}

# Raw counts: number of adversarial successes (out of 50)
data = {
    "homoglyphs": [
        [16, 40, 49],  # Bhadresh
        [13, 15, 14],  # GPT-4o
        [13.5, 15, 16],  # Gemini,
        [16, 23, 41], # Bhadresh-ft
        [16, 32, 46], # Bhadresh-ft-enc
        [18, 32, 48] # T5-Bhadresh
    ],
    "invisible": [
        [16, 16, 16],
        [13, 14, 15],
        [13.5, 15, 17],
        [16, 16, 16],
        [16, 16, 16],
        [18, 35, 47]
    ],
    "deletions": [
        [16, 41, 50],
        [13, 14, 17],
        [14, 16, 20],
        [18, 29, 44],
        [18, 35, 48],
        [18, 39, 48]
    ],
    "reorderings": [
        [16, 39, 49],
        [13, 14, 20],
        [11.5, 16, 22],
        [16, 25, 45],
        [16, 33, 47],
        [18, 35, 50]
    ]
}

# Convert to attack success rates (0.0 to 1.0)
data_rate = {
    perturb: [
        [v / 50 for v in model_data] for model_data in data[perturb]
    ] for perturb in perturbations
}

# Plotting
fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, perturb in enumerate(perturbations):
    ax = axs[i]
    for j, model in enumerate(models):
        values = data_rate[perturb][j]
        ax.plot(budgets, values, marker='o', label=model, color=colours[model])
    ax.set_title(perturb.capitalize())
    ax.set_xticks(budgets)
    ax.set_xlabel("Budget")
    if i == 0:
        ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1.1)
    ax.grid(True)

fig.suptitle("Attack Success Rate by Perturbation Type and Budget")
fig.legend(models, loc='upper left', ncol=3, bbox_to_anchor=(0, 1))
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("all_sc_t.png", dpi=300, bbox_inches='tight')