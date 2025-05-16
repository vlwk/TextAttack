import matplotlib.pyplot as plt
import numpy as np

# Define data
budgets = [0, 1, 5]
perturbations = ["homoglyphs", "invisible", "deletions", "reorderings"]
models = ["Bhadresh", "GPT-4o", "Gemini 2F"]
colours = {
    "Bhadresh": "#1f77b4",
    "GPT-4o": "#ff7f0e",
    "Gemini 2F": "#2ca02c"
}

# Raw counts: number of adversarial successes (out of 50)
data = {
    "homoglyphs": [
        [16, 40, 49],  # Bhadresh
        [13, 15, 14],  # GPT-4o
        [13.5, 15, 16]  # Gemini
    ],
    "invisible": [
        [16, 16, 16],
        [13, 14, 15],
        [13.5, 15, 17]
    ],
    "deletions": [
        [16, 41, 50],
        [13, 14, 17],
        [14, 16, 20]
    ],
    "reorderings": [
        [16, 39, 49],
        [13, 14, 20],
        [11.5, 16, 22]
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
fig.legend(models, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()