import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Data for MNLI attack
budgets = [0, 1, 2, 3, 4, 5]
total = 50
homoglyphs = [19, 21, 21, 22, 23, 24]
invisible = [19, 22, 25, 26, 28, 30]
deletions = [19, 22, 27, 30, 30, 35]
reorderings = [19, 24, 34, 36, 44, 43]

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

plt.title("Textual Entailment Targeted Attack: Facebook Fairseq MNLI")
plt.xlabel("Perturbation Budget")
plt.ylabel("Predicted Target Class")
plt.xticks(budgets)
plt.ylim(0, 100)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mnli_targeted_mine.png", dpi=300, bbox_inches='tight')
plt.show()
