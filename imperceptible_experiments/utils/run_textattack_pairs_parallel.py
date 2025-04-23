import textattack
from textattack import Attack
import json
import os
import time
import multiprocessing as mp

def _attack_worker(index, input_text, expected_output, attack):
    try:
        start_time = time.time()
        result = attack.attack(input_text, expected_output).perturbed_result
        elapsed_time = time.time() - start_time
        return {
            "index": index,
            "elapsed_time": elapsed_time,
            "input_text": input_text,
            "correct_output": expected_output,
            "perturbed_text": result.attacked_text.text,
            "perturbed_output": result.output,
            "score": result.score,
            "goal_status": result.goal_status
        }
    except Exception as e:
        print(f"[!] Failed on index {index}: {e}")
        return None

def run_textattack_pairs_parallel(
    attack: textattack.Attack,
    data_pairs,
    start_idx,
    results_path: str,
    num_workers: int = None
):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    indexed_data = [(i + start_idx, inp, out, attack) for i, (inp, out) in enumerate(data_pairs)]
    
    if num_workers is None:
        num_workers = mp.cpu_count()

    import torch
    torch.set_num_threads(1)

    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(_attack_worker, indexed_data)

    results = [r for r in results if r is not None]

    with open(results_path, "a", encoding="utf8") as f:
        for result_entry in results:
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(results)} results to {results_path}")
