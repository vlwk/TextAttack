import os
import json
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--perturbation_type", required=True)
    return parser.parse_args()

def argmax_index(lst):
    return max(range(len(lst)), key=lambda i: lst[i])

def get_prediction(entry):
    if "raw_prediction" in entry:
        return argmax_index(entry["raw_prediction"])
    elif "probs" in entry:
        return argmax_index(entry["probs"])
    elif "logits" in entry:
        return argmax_index(entry["logits"])
    else:
        raise ValueError("No valid prediction field found.")

def evaluate_file(filepath, with_levels=True):
    with open(filepath, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    total = len(lines)
    overall_correct = 0

    if with_levels:
        chunk_size = total // 10
        results_by_level = defaultdict(lambda: {"correct": 0, "total": 0})

        for i, ex in enumerate(lines):
            try:
                true_label = ex["true_label"]
                predicted = get_prediction(ex)
                level = min((i // chunk_size) + 1, 10)
                results_by_level[level]["total"] += 1
                if predicted == true_label:
                    results_by_level[level]["correct"] += 1
                    overall_correct += 1
            except Exception as e:
                print(f"Skipping row {i}: {e}")

        return results_by_level, total, overall_correct

    else:
        for ex in lines:
            try:
                true_label = ex["true_label"]
                predicted = get_prediction(ex)
                if predicted == true_label:
                    overall_correct += 1
            except Exception as e:
                print(f"Skipping row: {e}")
        return None, total, overall_correct

def log_results(name, results_by_level, total, correct, with_levels):
    output = [f"=== {name.upper()} ==="]

    if with_levels:
        for level in range(1, 11):
            r = results_by_level[level]
            acc = r["correct"] / r["total"] if r["total"] > 0 else 0.0
            output.append(f"Perturbation {level}: {acc:.4f} ({r['correct']} / {r['total']})")

    total_acc = correct / total if total > 0 else 0.0
    output.append(f"TOTAL ACCURACY: {total_acc:.4f} ({correct} / {total})")
    return "\n".join(output)

def main():
    args = parse_args()

    input_path = os.path.join(
        "results", "sentiment_classification", args.dataset_name,
        "oneshot", args.model_name, f"logits_{args.perturbation_type}.jsonl"
    )

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with_levels = args.perturbation_type != "clean"
    results_by_level, total, correct = evaluate_file(input_path, with_levels=with_levels)
    output_str = log_results(args.perturbation_type, results_by_level, total, correct, with_levels)

    print("\n" + output_str)

    output_path = os.path.join(
        "analysis", "sentiment_classification", args.dataset_name,
        "oneshot", args.model_name, f"{args.perturbation_type}.txt"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(output_str + "\n")

if __name__ == "__main__":
    main()
