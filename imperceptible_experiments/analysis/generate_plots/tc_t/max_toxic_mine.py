import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Data
budgets = [0, 1, 2, 3, 4, 5]
homoglyphs_raw = [5, 12, 24, 28, 34, 38]
invisible_raw = [5, 5, 5, 5, 5, 5]
deletions_raw = [5, 17, 24, 30, 35, 38]
reorderings_raw = [5, 16, 25, 35, 39, 38]
total = 50

# Convert to success rates
homoglyphs = [(total - x) / total * 100 for x in homoglyphs_raw]
invisible = [(total - x) / total * 100 for x in invisible_raw]
deletions = [(total - x) / total * 100 for x in deletions_raw]
reorderings = [(total - x) / total * 100 for x in reorderings_raw]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(budgets, homoglyphs, marker='^', color='green', label='Homoglyphs')
plt.plot(budgets, invisible, marker='o', color='blue', label='Invisible')
plt.plot(budgets, deletions, marker='s', color='red', label='Deletions')
plt.plot(budgets, reorderings, marker='v', color='orange', label='Reorderings')

# Labels and formatting
plt.title("Toxic Content Classification\nIBM Toxic Content Classifier")
plt.xlabel("Perturbation Budget")
plt.ylabel("Percentage Toxic Examples Classified Toxic")
plt.xticks(budgets)
plt.ylim(0, 100)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("max_toxic_mine.png", dpi=300, bbox_inches='tight')
plt.show()
