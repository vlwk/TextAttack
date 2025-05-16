import argparse
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import bootstrap
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="mnli", help="Dataset name")
    parser.add_argument("--num_examples", type=int, default=50, help="Number of examples")
    parser.add_argument("--popsize", type=int, default=5, help="Population size")
    parser.add_argument("--maxiter", type=int, default=3, help="Max iterations")
    parser.add_argument("--base_dir", default="results", help="Base directory for results")
    return parser.parse_args()

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval using bootstrapping."""
    data = np.array(data)
    bootstrap_results = bootstrap((data,), np.mean, n_resamples=10000)
    return (
        bootstrap_results.confidence_interval.low,
        bootstrap_results.confidence_interval.high
    )

def generate_plot(base_dir, output_dir, dataset_name, num_examples, popsize, maxiter):
    """Generate plot with confidence intervals."""
    perturbations = ["homoglyphs", "invisible", "deletions", "reorderings"]
    colors = {
        "homoglyphs": "green",
        "invisible": "blue",
        "deletions": "red",
        "reorderings": "orange"
    }
    markers = {
        "homoglyphs": "^",
        "invisible": "o",
        "deletions": "s",
        "reorderings": "v"
    }
    budgets = [0, 1, 2, 3, 4, 5]

    plt.figure(figsize=(10, 6))

    # For each perturbation type
    for perturb in perturbations:
        means = []
        ci_lows = []
        ci_highs = []
        
        # For each budget
        for budget in budgets:
            # Construct path and read data
            if budget == 0:
                # For budget 0, we need to read from any perturbation type's original scores
                subdir = os.path.join(
                    dataset_name,
                    f"num{num_examples}",
                    "fairseq_mnli",
                    f"pop{popsize}_iter{maxiter}",
                    perturbations[0],  # Use first perturbation type
                    f"pert1"  # Use any budget folder
                )
                log_path = os.path.join(base_dir, "te_t", subdir, "log.csv")
                if os.path.exists(log_path):
                    df = pd.read_csv(log_path)
                    # For TE-T, success is when predicted class matches target class
                    successes = (df['predicted_class'] == df['target_class']).astype(float)
                else:
                    print(f"Warning: File not found: {log_path}")
                    continue
            else:
                subdir = os.path.join(
                    dataset_name,
                    f"num{num_examples}",
                    "fairseq_mnli",
                    f"pop{popsize}_iter{maxiter}",
                    perturb,
                    f"pert{budget}"
                )
                log_path = os.path.join(base_dir, "te_t", subdir, "log.csv")
                if os.path.exists(log_path):
                    df = pd.read_csv(log_path)
                    successes = (df['predicted_class'] == df['target_class']).astype(float)
                else:
                    print(f"Warning: File not found: {log_path}")
                    continue

            # Calculate statistics
            success_rate = np.mean(successes) * 100  # Convert to percentage
            ci_low, ci_high = calculate_confidence_interval(successes)
            ci_low *= 100  # Convert to percentage
            ci_high *= 100  # Convert to percentage
            
            means.append(success_rate)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)
        
        if means:  # Only plot if we have data
            plt.plot(budgets, means, marker=markers[perturb], color=colors[perturb], 
                    label=perturb.capitalize())
            plt.fill_between(budgets, ci_lows, ci_highs, alpha=0.2, color=colors[perturb])

    # Labels and formatting
    plt.title("Textual Entailment Targeted Attack: Facebook Fairseq MNLI (with 95% CI)")
    plt.xlabel("Perturbation Budget")
    plt.ylabel("Predicted Target Class")
    plt.xticks(budgets)
    plt.ylim(0, 100)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'mnli_targeted_mine.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # Generate plot
    output_dir = os.path.join("figures", "te_t")
    os.makedirs(output_dir, exist_ok=True)
    
    generate_plot(
        args.base_dir,
        output_dir,
        args.dataset_name,
        args.num_examples,
        args.popsize,
        args.maxiter
    )
    print(f"Generated plot in {os.path.join(output_dir, 'mnli_targeted_mine.png')}")

if __name__ == "__main__":
    main() 