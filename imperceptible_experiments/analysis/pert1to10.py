import matplotlib.pyplot as plt
import numpy as np

# Budgets (x-axis)
budgets = np.arange(1, 11)

# Define models and their accuracy data for each perturbation type from budget 1 to 10
perturbation_types = ["homoglyphs", "invisible", "deletions", "reorderings"]
models = ["Bhadresh", "Bhadresh-ft", "Bhadresh-ft-enc", "T5-Bhadresh", "GPT-4o", "Gemini 2F"]
colours = {
    "Bhadresh": "#1f77b4",
    "Bhadresh-ft": "#9467bd",
    "Bhadresh-ft-enc": "#d62728",
    "T5-Bhadresh": "#8c564b",
    "GPT-4o": "#ff7f0e",
    "Gemini 2F": "#2ca02c"
}

# Accuracy data from logs
accuracy_per_perturbation = {
    "homoglyphs": {
        "Bhadresh": [0.8700, 0.8400, 0.8100, 0.7850, 0.6950, 0.7400, 0.7150, 0.7000, 0.6400, 0.6250],
        "Bhadresh-ft": [0.8800, 0.8550, 0.8250, 0.8500, 0.8350, 0.7800, 0.7650, 0.7600, 0.7500, 0.7950],
        "Bhadresh-ft-enc": [0.9200, 0.8600, 0.8300, 0.8700, 0.8050, 0.7250, 0.7550, 0.7900, 0.7250, 0.7500],
        "T5-Bhadresh": [0.7875]*10,
        "GPT-4o": [0.5650, 0.5700, 0.5200, 0.5700, 0.5900, 0.6350, 0.5300, 0.5700, 0.5700, 0.6600],
        "Gemini 2F": [0.5800, 0.5000, 0.4850, 0.5250, 0.5200, 0.5700, 0.4650, 0.5350, 0.5050, 0.6000]
    },
    "invisible": {
        "Bhadresh": [0.9300, 0.9450, 0.8850, 0.9150, 0.9350, 0.9250, 0.9500, 0.9350, 0.9100, 0.9400],
        "Bhadresh-ft": [0.9200, 0.9350, 0.8800, 0.9300, 0.9250, 0.9350, 0.9450, 0.9100, 0.9250, 0.9400],
        "Bhadresh-ft-enc": [0.9250, 0.9300, 0.9050, 0.9300, 0.9200, 0.9200, 0.9200, 0.9400, 0.9100, 0.9500],
        "T5-Bhadresh": [0.8140]*10,
        "GPT-4o": [0.5950, 0.5550, 0.5250, 0.5800, 0.5900, 0.6200, 0.5100, 0.6150, 0.5550, 0.6500],
        "Gemini 2F": [0.5650, 0.5050, 0.4800, 0.5300, 0.5200, 0.5450, 0.5100, 0.5450, 0.4850, 0.6350]
    },
    "deletions": {
        "Bhadresh": [0.9000, 0.8500, 0.7950, 0.8250, 0.8000, 0.7400, 0.7250, 0.7250, 0.6600, 0.6750],
        "Bhadresh-ft": [0.9100, 0.9050, 0.8600, 0.8650, 0.8600, 0.8050, 0.8100, 0.8400, 0.7950, 0.7850],
        "Bhadresh-ft-enc": [0.9100, 0.8900, 0.8750, 0.8600, 0.8700, 0.7950, 0.7700, 0.8200, 0.7300, 0.8200],
        "T5-Bhadresh": [0.8140]*10,
        "GPT-4o": [0.5950, 0.5350, 0.5450, 0.5800, 0.5750, 0.6050, 0.5300, 0.6000, 0.5550, 0.6350],
        "Gemini 2F": [0.5650, 0.4850, 0.5000, 0.5050, 0.4600, 0.4850, 0.4850, 0.4300, 0.4300, 0.5000]
    },
    "reorderings": {
        "Bhadresh": [0.9050, 0.8400, 0.7600, 0.8150, 0.7250, 0.7450, 0.7000, 0.7300, 0.6550, 0.6900],
        "Bhadresh-ft": [0.9000, 0.9050, 0.8500, 0.8900, 0.8300, 0.8300, 0.8400, 0.8250, 0.8100, 0.8300],
        "Bhadresh-ft-enc": [0.9250, 0.9050, 0.8350, 0.8800, 0.8650, 0.8100, 0.8100, 0.7750, 0.8250, 0.7600],
        "T5-Bhadresh": [0.7740]*10,
        "GPT-4o": [0.5800, 0.5650, 0.5250, 0.5500, 0.5400, 0.6350, 0.5500, 0.5400, 0.5350, 0.6750],
        "Gemini 2F": [0.5600, 0.5100, 0.5150, 0.5000, 0.5400, 0.5650, 0.4700, 0.5200, 0.4650, 0.5450]
    }
}

# Create subplots for each perturbation type
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axs = axs.flatten()

for i, perturb in enumerate(perturbation_types):
    ax = axs[i]
    for model in models:
        acc = accuracy_per_perturbation[perturb][model]
        ax.plot(budgets, acc, marker='o', label=model, color=colours[model])
    ax.set_title(perturb.capitalize())
    ax.set_xticks(budgets)
    ax.set_xlabel("Perturbation Budget")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True)
    if i % 2 == 0:
        ax.set_ylabel("Accuracy")

fig.suptitle("Model Accuracy Across Budgets for Each Perturbation Type", fontsize=16)
fig.legend(models, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

