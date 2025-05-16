import matplotlib.pyplot as plt

# Data
budgets = [0, 1, 2, 3, 4, 5]
homoglyphs = [45.72, 69.96, 83.32, 92.82, 101.08, 102.28]
invisible = [45.72, 76.18, 89.38, 107.86, 121.82, 132.5]
deletions = [45.72, 73.72, 89.94, 93.74, 100.42, 104.38]
reorderings = [45.72, 108.24, 170.22, 259.26, 327.42, 449.94]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(budgets, homoglyphs, marker='^', color='green', label='Homoglyphs')
plt.plot(budgets, invisible, marker='o', color='blue', label='Invisible')
plt.plot(budgets, deletions, marker='s', color='red', label='Deletions')
plt.plot(budgets, reorderings, marker='v', color='orange', label='Reorderings')

# Labels and formatting
plt.title("Machine Translation Integrity Attack:\nFacebook Fairseq Levenshtein Distance")
plt.xlabel("Perturbation Budget")
plt.ylabel("Levenshtein Distance to Reference Translation")
plt.xticks(budgets)
plt.ylim(0, 600)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("levenshtein_fairseq_mine.png", dpi=300, bbox_inches='tight')
plt.show()
