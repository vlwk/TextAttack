import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--popsize", type=int, required=True)
    parser.add_argument("--maxiter", type=int, required=True)
    parser.add_argument("--base_dir", default="results", help="Base directory for logs")
    parser.add_argument("--output_dir", default="imperceptible_experiments/figures/mt_u", help="Where to save the plot")
    return parser.parse_args()


def calculate_confidence_interval(data):
    data = np.array(data)
    if len(data) == 0:
        return 0.0, 0.0
    result = bootstrap((data,), np.mean, n_resamples=10000)
    return result.confidence_interval.low, result.confidence_interval.high


def main():
    args = parse_args()

    perturbations = ["homoglyphs", "invisible", "deletions", "reorderings"]
    markers = {
        "homoglyphs": "^",
        "invisible": "o",
        "deletions": "s",
        "reorderings": "v"
    }
    colours = {
        "homoglyphs": "green",
        "invisible": "blue",
        "deletions": "red",
        "reorderings": "orange"
    }
    budgets = range(0, 6)
    model = args.model_name

    plt.figure(figsize=(10, 6))
    for perturb in perturbations:
        means = []
        ci_lows = []
        ci_highs = []

        original_scores = []
        # Load original scores from budget 1 (to extract clean scores)
        orig_subdir = os.path.join(
            args.dataset_name,
            f"num{args.num_examples}",
            model,
            f"pop{args.popsize}_iter{args.maxiter}",
            perturb,
            f"pert1"
        )
        orig_log_path = os.path.join(args.base_dir, "mt_u", orig_subdir, "log.csv")
        if os.path.exists(orig_log_path):
            with open(orig_log_path, newline="") as orig_csvfile:
                reader = csv.DictReader(orig_csvfile)
                for row in reader:
                    try:
                        original_scores.append(float(row["original_score"]))
                    except ValueError:
                        continue
        else:
            print(f"Warning: Could not find original score file for {perturb}. Skipping.")
            means.extend([np.nan] * len(budgets))
            ci_lows.extend([np.nan] * len(budgets))
            ci_highs.extend([np.nan] * len(budgets))
            continue

        for budget in budgets:
            distances = []

            if budget == 0:
                distances = original_scores
            else:
                subdir = os.path.join(
                    args.dataset_name,
                    f"num{args.num_examples}",
                    model,
                    f"pop{args.popsize}_iter{args.maxiter}",
                    perturb,
                    f"pert{budget}"
                )
                log_path = os.path.join(args.base_dir, "mt_u", subdir, "log.csv")

                if not os.path.exists(log_path):
                    print(f"Log file not found: {log_path}")
                    means.append(np.nan)
                    ci_lows.append(np.nan)
                    ci_highs.append(np.nan)
                    continue

                with open(log_path, newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        try:
                            perturbed_score = float(row["perturbed_score"])
                            distances.append(perturbed_score)
                        except ValueError:
                            continue

            if distances:
                mean = np.mean(distances)
                ci_low, ci_high = calculate_confidence_interval(distances)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
            else:
                means.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)

        plt.plot(budgets, means, marker=markers[perturb], color=colours[perturb], label=perturb.capitalize())
        plt.fill_between(budgets, ci_lows, ci_highs, alpha=0.2, color=colours[perturb])

    plt.title(f"Levenshtein Distance vs Budget ({model})")
    plt.xlabel("Perturbation Budget")
    plt.ylabel("Levenshtein Distance (Lower is Better)")
    plt.xticks(budgets)
    plt.ylim(0, 600)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"levenshtein_{model}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()