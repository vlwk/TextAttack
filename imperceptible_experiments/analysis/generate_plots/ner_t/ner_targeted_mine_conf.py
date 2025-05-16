import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import bootstrap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--popsize", type=int, required=True)
    parser.add_argument("--maxiter", type=int, required=True)
    parser.add_argument("--target_suffix", type=str, required=True)
    parser.add_argument("--base_dir", default="results", help="Base directory for logs")
    parser.add_argument("--output_dir", default="figures/ner_t", help="Where to save the plot")
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

        # For budget 0, we need to read from any perturbation type's original scores
        orig_subdir = os.path.join(
            args.dataset_name,
            f"num{args.num_examples}",
            model,
            f"pop{args.popsize}_iter{args.maxiter}",
            perturb,
            f"target{args.target_suffix}",
            f"pert1"
        )
        orig_log_path = os.path.join(args.base_dir, "ner_t", orig_subdir, "log.csv")
        if not os.path.exists(orig_log_path):
            print(f"Warning: Could not find original score file for {perturb}. Skipping.")
            means.extend([np.nan] * len(budgets))
            ci_lows.extend([np.nan] * len(budgets))
            ci_highs.extend([np.nan] * len(budgets))
            continue

        for budget in budgets:
            successes = []

            if budget == 0:
                # For budget 0, success is when original_score > 0.0
                with open(orig_log_path, newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        try:
                            successes.append(float(float(row["original_score"]) > 0.0))
                        except ValueError:
                            successes.append(0.0)
            else:
                subdir = os.path.join(
                    args.dataset_name,
                    f"num{args.num_examples}",
                    model,
                    f"pop{args.popsize}_iter{args.maxiter}",
                    perturb,
                    f"target{args.target_suffix}",
                    f"pert{budget}"
                )
                log_path = os.path.join(args.base_dir, "ner_t", subdir, "log.csv")

                if not os.path.exists(log_path):
                    print(f"Log file not found: {log_path}")
                    means.append(np.nan)
                    ci_lows.append(np.nan)
                    ci_highs.append(np.nan)
                    continue

                # For non-zero budgets, success is when result_type is "Successful"
                with open(log_path, newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        result_type = row.get("result_type", "").strip()
                        successes.append(float(result_type == "Successful"))

            if successes:
                # Convert to percentage
                successes = np.array(successes) * 100
                mean = np.mean(successes)
                ci_low, ci_high = calculate_confidence_interval(successes)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
            else:
                means.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)

        plt.plot(budgets, means, marker=markers[perturb], color=colours[perturb], label=perturb.capitalize())
        plt.fill_between(budgets, ci_lows, ci_highs, alpha=0.2, color=colours[perturb])

    plt.title(f"NER Model Targeted Attacks (with 95% CI)")
    plt.xlabel("Perturbation Budget")
    plt.ylabel("Attack Success Rate")
    plt.xticks(budgets)
    plt.ylim(30, 100)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"ner_targeted_{model}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()