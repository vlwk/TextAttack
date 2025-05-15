import matplotlib.pyplot as plt
import numpy as np

# Define data
budgets = [0, 1, 5]
perturbations = ["homoglyphs", "invisible", "deletions", "reorderings"]
models = ["Fairseq (en-fr)", "GPT-4o", "Gemini 2F"]
colours = {
    "Fairseq (en-fr)": "#1f77b4",
    "GPT-4o": "#ff7f0e",
    "Gemini 2F": "#2ca02c"
}

# Raw Levenshtein distance values (not rates)
data = {
    "homoglyphs": [
        [45.72, 69.96, 102.28],  # Fairseq (en-fr)
        [(104.56+103.34)/2, 113.18, 113.28],  # GPT-4o
        [(46.24+46.58)/2, 54.88, 74.56]  # Gemini
    ],
    "invisible": [
        [45.72, 76.18, 132.5],
        [(105.68+106.84)/2, 113.2, 113.52],
        [(45.96+45.04)/2, 57.28, 61.64]
    ],
    "deletions": [
        [45.72, 73.72, 104.38],
        [(104.32+105.84)/2, 113.22, 113.52],
        [(46.06+44.78)/2, 57.94, 192.38]
    ],
    "reorderings": [
        [45.72, 108.24, 449.94],
        [(106.42+106.02)/2, 113.22, 113.30],
        [(45.76+47.18)/2, 57.66, 143.90]
    ]
}

# Plotting
fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, perturb in enumerate(perturbations):
    ax = axs[i]
    for j, model in enumerate(models):
        values = data[perturb][j]
        ax.plot(budgets, values, marker='o', label=model, color=colours[model])
    ax.set_title(perturb.capitalize())
    ax.set_xticks(budgets)
    ax.set_xlabel("Budget")
    if i == 0:
        ax.set_ylabel("Average Levenshtein Distance")
    ax.grid(True)

fig.suptitle("Average Levenshtein Distance by Perturbation Type and Budget")
fig.legend(models, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
plt.tight_layout(rect=[0, 0, 1, 0.98])
# plt.show()
plt.savefig("levenshtein_plot.png", dpi=300, bbox_inches='tight')