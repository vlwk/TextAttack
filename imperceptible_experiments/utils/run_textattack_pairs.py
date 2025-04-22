import textattack
from textattack import Attack

import json
import os
import time

def run_textattack_pairs(
    attack: textattack.Attack,
    data_pairs,
    results_path: str
):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "a", encoding="utf8") as results:
        for index, (input_text, expected_output) in enumerate(data_pairs):
            start_time = time.time()
            try:
                attack_result = attack.attack(input_text, expected_output).perturbed_result
            except Exception as e:
                print(f"[!] Failed on index {index}: {e}")
                continue
            elapsed_time = time.time() - start_time
            result_entry = {
                "index": index,
                "elapsed_time": elapsed_time,
                "input_text": input_text,
                "correct_output": expected_output,
                "perturbed_text": attack_result.attacked_text.text,
                "perturbed_output": attack_result.output,
                "score": attack_result.score,
                "goal_status": attack_result.goal_status
            }
            results.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            print(f"Processed: {index + 1} / {len(data_pairs)}")
