import argparse
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="wmt14enfr", help="Dataset name")
    parser.add_argument("--num_examples", type=int, default=50, help="Number of examples")
    parser.add_argument("--popsize", type=int, default=5, help="Population size")
    parser.add_argument("--maxiter", type=int, default=3, help="Max iterations")
    parser.add_argument("--base_dir", default="imperceptible_experiments/results", help="Base directory for results")
    return parser.parse_args()

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval using bootstrapping."""
    data = np.array(data)
    bootstrap_results = bootstrap((data,), np.mean, n_resamples=10000)
    return (
        bootstrap_results.confidence_interval.low,
        bootstrap_results.confidence_interval.high
    )

def generate_comparison_plots(base_dir, output_dir, dataset_name, num_examples, popsize, maxiter):
    """Generate comparison plots with confidence intervals across models and perturbation types."""
    perturbations = ["homoglyphs", "invisible", "deletions", "reorderings"]
    models = ["fairseq_en_fr", "gpt4o", "gemini2f"]
    model_display_names = {
        "fairseq_en_fr": "Fairseq (en-fr)",
        "gpt4o": "GPT-4o",
        "gemini2f": "Gemini 2F"
    }
    colours = {
        "fairseq_en_fr": "#1f77b4",
        "gpt4o": "#ff7f0e",
        "gemini2f": "#2ca02c"
    }
    budgets = [0, 1, 5]

    # Create figure
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    
    # For each perturbation type
    for i, perturb in enumerate(perturbations):
        ax = axs[i]
        
        # For each model
        for model in models:
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
                        model,
                        f"pop{popsize}_iter{maxiter}",
                        perturbations[0],  # Use first perturbation type
                        f"pert1"  # Use any budget folder
                    )
                    log_path = os.path.join(base_dir, "mt_u", subdir, "log.csv")
                    if os.path.exists(log_path):
                        df = pd.read_csv(log_path)
                        scores = df['original_score'].astype(float)
                    else:
                        print(f"Warning: File not found: {log_path}")
                        continue
                else:
                    subdir = os.path.join(
                        dataset_name,
                        f"num{num_examples}",
                        model,
                        f"pop{popsize}_iter{maxiter}",
                        perturb,
                        f"pert{budget}"
                    )
                    log_path = os.path.join(base_dir, "mt_u", subdir, "log.csv")
                    if os.path.exists(log_path):
                        df = pd.read_csv(log_path)
                        scores = df['perturbed_score'].astype(float)
                    else:
                        print(f"Warning: File not found: {log_path}")
                        continue

                # Calculate statistics
                mean = np.mean(scores)
                ci_low, ci_high = calculate_confidence_interval(scores)
                
                means.append(mean)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
            
            if means:  # Only plot if we have data
                display_name = model_display_names[model]
                ax.plot(budgets, means, marker='o', label=display_name, color=colours[model])
                ax.fill_between(budgets, ci_lows, ci_highs, alpha=0.2, color=colours[model])
        
        ax.set_title(perturb.capitalize())
        ax.set_xticks(budgets)
        ax.set_xlabel("Budget")
        if i == 0:
            ax.set_ylabel("Average Levenshtein Distance")
        ax.grid(True)

    fig.suptitle("Average Levenshtein Distance by Perturbation Type and Budget (with 95% CI)")
    # fig.legend(model_display_names.values(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'levenshtein_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # Generate comparison plot
    output_dir = os.path.join("figures", "mt_u")  # Simplified path
    os.makedirs(output_dir, exist_ok=True)
    
    generate_comparison_plots(
        args.base_dir,
        output_dir,  # Pass the simplified path
        args.dataset_name,
        args.num_examples,
        args.popsize,
        args.maxiter
    )
    print(f"Generated comparison plot in {os.path.join(output_dir, 'levenshtein_plot.png')}")

if __name__ == "__main__":
    main() 