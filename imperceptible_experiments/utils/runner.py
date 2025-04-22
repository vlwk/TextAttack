import os
from datetime import datetime
import textattack
from textattack.search_methods import ImperceptibleDE

from imperceptible_experiments.utils.run_textattack_pairs import run_textattack_pairs
from imperceptible_experiments.utils.run_baseline_pairs import run_baseline_pairs

def run_experiment(
    model_wrapper,
    pairs,
    score_fn,
    goal_function,
    transformation,
    out_dir_base="results",
    experiment_name="experiment",
    popsize=10,
    maxiter=5,
    num_examples=5,
    perturb_range=range(0, 6),
):
    pairs = sorted(pairs, key=lambda x: len(x[0]))[:num_examples]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_dir_base, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    for max_perturbs in perturb_range:
        results_path = os.path.join(out_dir, f"{experiment_name}_p{max_perturbs}.jsonl")

        if max_perturbs == 0:
            run_baseline_pairs(model_wrapper, score_fn, pairs, results_path)
        else:
            search_method = ImperceptibleDE(
                popsize=popsize,
                maxiter=maxiter,
                verbose=True,
                max_perturbs=max_perturbs,
            )
            attack = textattack.Attack(goal_function, [], transformation, search_method)
            run_textattack_pairs(
                attack=attack,
                data_pairs=pairs,
                results_path=results_path
            )
