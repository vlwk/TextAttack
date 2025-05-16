import argparse
import os
import csv
import ast

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--popsize", type=int, required=True)
    parser.add_argument("--maxiter", type=int, required=True)
    parser.add_argument("--perturbation_type", required=True)
    parser.add_argument("--target_class", type=int, required=True)
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
        f"target{args.target_class}",
        f"pert{args.budget}"
    )

    log_path = os.path.join("results", "sc_t", subdir, "log.csv")
    output_path = os.path.join("analysis", "sc_t", subdir, "log.txt")

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    total_successful = 0
    total_failed = 0
    original_equals_target = 0
    count = 0

    with open(log_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            result_type = row.get("result_type", "").strip()
            if result_type == "Successful":
                total_successful += 1
            else:
                total_failed += 1

            try:
                original_output = ast.literal_eval(row["original_output"].replace("tensor", ""))
                predicted_label = original_output.index(max(original_output))
                if predicted_label == args.target_class:
                    original_equals_target += 1
            except Exception as e:
                print(f"Warning: Skipping row due to error: {e}")

            count += 1

    result_text = (
        f"File: {log_path}\n"
        f"Examples: {count}\n"
        f"Total successful:          {total_successful}\n"
        f"Total failed:              {total_failed}\n"
        f"Original == target class:  {original_equals_target} / {count} "
        f"({(original_equals_target / count * 100):.2f}% of inputs)\n"
    )

    print(result_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(result_text)

if __name__ == "__main__":
    main()
