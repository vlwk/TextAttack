import os
from datetime import datetime
import textattack
from textattack.search_methods import ImperceptibleDE

from imperceptible_experiments.utils.run_textattack_pairs_thread import run_textattack_pairs_thread
from imperceptible_experiments.utils.run_baseline_pairs import run_baseline_pairs

def run_experiment_thread(
    model_wrapper,
    pairs,
    start_idx,
    end_idx,
    score_fn,
    goal_function,
    transformation,
    out_dir_base="results",
    experiment_name="experiment",
    popsize=10,
    maxiter=5,
    perturb_range=range(0, 6),
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_dir_base, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    for max_perturbs in perturb_range:
        results_path = os.path.join(out_dir, f"{experiment_name}_p{max_perturbs}.jsonl")

        if max_perturbs == 0:
            run_baseline_pairs(model_wrapper, score_fn, pairs, start_idx, end_idx, results_path)
        else:
            search_method = ImperceptibleDE(
                popsize=popsize,
                maxiter=maxiter,
                verbose=True,
                max_perturbs=max_perturbs,
            )
            attack = textattack.Attack(goal_function, [], transformation, search_method)
            run_textattack_pairs_thread(
                attack=attack,
                data_pairs=pairs,
                start_idx=start_idx,
                end_idx=end_idx,
                results_path=results_path,
                max_workers=4
            )
