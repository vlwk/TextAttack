import argparse
import os
import torch
from dotenv import load_dotenv

from textattack import Attacker, Attack, AttackArgs
from textattack.datasets import Dataset
from textattack.goal_functions import MaximizeLevenshtein
from textattack.search_methods import ImperceptibleDE
from textattack.transformations import (
    WordSwapHomoglyphSwap,
    WordSwapInvisibleCharacters,
    WordSwapReorderings,
    WordSwapDeletions,
)
from textattack.models.wrappers import MarianWrapper
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from imperceptible_experiments.translation.load_data import load_en_fr_data

# -----------------------
# CLI args
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("start_row", type=int)
parser.add_argument("end_row", type=int)
args = parser.parse_args()

# -----------------------
# Setup
# -----------------------
if __name__ == "__main__":
    load_dotenv()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cur_dir)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    print("task_id:", task_id)

    # -----------------------
    # Load model + wrapper
    # -----------------------
    print("Loading mBART model...")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    model = model.to("cuda")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")

    model_wrapper = MarianWrapper(model, tokenizer)

    # -----------------------
    # Load and slice data
    # -----------------------
    print("Loading and slicing data...")
    pairs = load_en_fr_data()
    pairs = pairs[args.start_row : args.end_row]
    dataset = Dataset(pairs)

    # -----------------------
    # Map SLURM task_id to transformation/perturbation
    # -----------------------
    all_transforms = ["homo", "invis", "reord", "del"]
    perturb_levels = list(range(1, 6))
    grid = [(t, p) for t in all_transforms for p in perturb_levels]  # 20 jobs
    transformation_name, perturb_level = grid[task_id]
    print(f"[🧠] Task {task_id} → Transformation: {transformation_name}, Perturb level: {perturb_level}")

    # -----------------------
    # Setup transformation
    # -----------------------
    transformations = {
        "homo": WordSwapHomoglyphSwap(),
        "invis": WordSwapInvisibleCharacters(),
        "reord": WordSwapReorderings(),
        "del": WordSwapDeletions(),
    }
    transformation = transformations[transformation_name]

    # -----------------------
    # Attack setup
    # -----------------------
    goal_function = MaximizeLevenshtein(model_wrapper)
    search_method = ImperceptibleDE()
    attack = Attack(goal_function, [], transformation, search_method)

    # -----------------------
    # AttackArgs
    # -----------------------
    log_prefix = f"marian_lev_{transformation_name}_{args.start_row}_{args.end_row - 1}_p{perturb_level}"

    attack_args = AttackArgs(
        num_examples=len(pairs),
        disable_stdout=True,
        parallel=True,  # GPU mode
        log_to_csv=f"logs/{log_prefix}.csv",
        checkpoint_dir=f"checkpoints/{log_prefix}",
        checkpoint_interval=50,
        random_seed=42
    )

# -----------------------
# Run the attack
# -----------------------

    attacker = Attacker(attack, dataset, attack_args)
    device = next(model.parameters()).device
    print(f"Model running on device: {device}")
    attacker.attack_dataset()
