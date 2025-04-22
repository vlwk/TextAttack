import json
import os
import time
import functools
import sacrebleu

from textattack.shared import AttackedText


def run_baseline_pairs(model_wrapper, score_fn, data_pairs, results_path: str):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "a", encoding="utf8") as results:
        for index, (input_text, expected_output) in enumerate(data_pairs):
            start_time = time.time()
            try:
                model_output = model_wrapper([input_text])[0]
                bleu_score = score_fn(model_output, expected_output)
            except Exception as e:
                print(f"[!] Failed on index {index}: {e}")
                continue

            elapsed_time = time.time() - start_time
            result_entry = {
                "index": index,
                "elapsed_time": elapsed_time,
                "input_text": input_text,
                "correct_output": expected_output,
                "perturbed_text": input_text,       # no perturbation
                "perturbed_output": model_output,   # model output from clean input
                "score": bleu_score,
                "goal_status": 1  # Always success for baseline
            }
            results.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            print(f"[✓] Baseline {index + 1}/{len(data_pairs)}")
