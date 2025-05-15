import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Data
budgets = [0, 1, 2, 3, 4, 5]
homoglyphs_raw = [22, 25, 31, 35, 39, 37]
invisible_raw = [22, 22, 22, 22, 22, 22]
deletions_raw = [22, 32, 40, 42, 47, 47]
reorderings_raw = [22, 26, 32, 36, 40, 42]
total = 50

# Convert to success rates
homoglyphs = [x / total * 100 for x in homoglyphs_raw]
invisible = [x / total * 100 for x in invisible_raw]
deletions = [x / total * 100 for x in deletions_raw]
reorderings = [x / total * 100 for x in reorderings_raw]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(budgets, homoglyphs, marker='^', color='green', label='Homoglyphs')
plt.plot(budgets, invisible, marker='o', color='blue', label='Invisible')
plt.plot(budgets, deletions, marker='s', color='red', label='Deletions')
plt.plot(budgets, reorderings, marker='v', color='orange', label='Reorderings')

# Labels and formatting
plt.title("NER Model Targeted Attacks")
plt.xlabel("Perturbation Budget")
plt.ylabel("Attack Success Rate")
plt.xticks(budgets)
plt.ylim(30, 100)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ner_targeted_mine.png", dpi=300, bbox_inches='tight')
plt.show()
