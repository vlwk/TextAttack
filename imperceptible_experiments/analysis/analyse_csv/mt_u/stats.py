import argparse
import os
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--popsize", type=int, required=True)
    parser.add_argument("--maxiter", type=int, required=True)
    parser.add_argument("--perturbation_type", required=True)
    parser.add_argument("--budget", type=int, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    subdir = os.path.join(
        args.dataset_name,
        f"num{args.num_examples}",
        args.model_name,
        f"pop{args.popsize}_iter{args.maxiter}",
        args.perturbation_type,
        f"pert{args.budget}"
    )

    log_path = os.path.join("results", "mt_u", subdir, "log.csv")
    output_path = os.path.join("analysis", "mt_u", subdir, "log.txt")

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    total_original_score = 0.0
    total_perturbed_score = 0.0
    count = 0

    with open(log_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                total_original_score += float(row["original_score"])
                total_perturbed_score += float(row["perturbed_score"])
                count += 1
            except ValueError:
                print(f"Skipping row with invalid score: {row}")

    if count == 0:
        print("No valid rows found.")
        return

    avg_original = total_original_score / count
    avg_perturbed = total_perturbed_score / count

    result_text = (
        f"File: {log_path}\n"
        f"Examples: {count}\n"
        f"Average original score:  {avg_original:.4f}\n"
        f"Average perturbed score: {avg_perturbed:.4f}\n"
    )

    print(result_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(result_text)

if __name__ == "__main__":
    main()
