from concurrent.futures import ThreadPoolExecutor
import textattack
import json
import os
import time

def process_pair_threaded(idx, input_text, expected_output, attack):
    try:
        start_time = time.time()
        attack_result = attack.attack(input_text, expected_output).perturbed_result
        elapsed_time = time.time() - start_time
        return {
            "index": idx,
            "elapsed_time": elapsed_time,
            "input_text": input_text,
            "correct_output": expected_output,
            "perturbed_text": attack_result.attacked_text.text,
            "perturbed_output": attack_result.output,
            "score": attack_result.score,
            "goal_status": attack_result.goal_status
        }
    except Exception as e:
        print(f"[!] Failed on index {idx}: {e}")
        return None

def run_textattack_pairs_thread(
    attack,
    data_pairs,
    start_idx,
    results_path,
    max_workers=None
):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # data_pairs = data_pairs[start_idx:end_idx]
    indexed = [(i + start_idx, input_text, output_text) for i, (input_text, output_text) in enumerate(data_pairs)]

    if max_workers is None:
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pair_threaded, idx, x, y, attack) for idx, x, y in indexed]
        results = [f.result() for f in futures if f and f.result() is not None]

    with open(results_path, "a", encoding="utf8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {results_path}")
