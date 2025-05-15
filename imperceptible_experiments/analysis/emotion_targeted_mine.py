import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Raw data: number of non-toxic classifications
budgets = [0, 1, 2, 3, 4, 5]
total = 50
homoglyphs = [16, 40, 46, 48, 49, 49]
invisible = [16, 16, 16, 16, 16, 16]
deletions = [16, 41, 48, 49, 49, 50]
reorderings = [16, 39, 46, 49, 48, 49]

# Convert to percentage
homoglyphs = [x / total * 100 for x in homoglyphs]
invisible = [x / total * 100 for x in invisible]
deletions = [x / total * 100 for x in deletions]
reorderings = [x / total * 100 for x in reorderings]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(budgets, homoglyphs, marker='^', color='green', label='Homoglyphs')
plt.plot(budgets, invisible, marker='o', color='blue', label='Invisible')
plt.plot(budgets, deletions, marker='s', color='red', label='Deletions')
plt.plot(budgets, reorderings, marker='v', color='orange', label='Reorderings')

plt.title("Emotion Model Targeted Attacks")
plt.xlabel("Perturbation Budget")
plt.ylabel("Attack Success Rate")
plt.xticks(budgets)
plt.ylim(10, 100)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("emotion_targeted_mine.png", dpi=300, bbox_inches='tight')
plt.show()
