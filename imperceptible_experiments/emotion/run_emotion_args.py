import argparse
import os
import torch
from dotenv import load_dotenv

from textattack.goal_functions import Emotion
from textattack.transformations import (
    WordSwapHomoglyphSwap,
    WordSwapInvisibleCharacters,
    WordSwapReorderings,
    WordSwapDeletions
    
)
from textattack.models.wrappers import EmotionWrapper

from imperceptible_experiments.utils.runner_parallel import run_experiment_parallel
from imperceptible_experiments.emotion.load import load_emotion
from imperceptible_experiments.emotion.load_data import load_emotion_data_pairs
from imperceptible_experiments.utils.scoring import get_lev_score

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
load_dotenv()
cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)

# -----------------------
# Load slurm info
# -----------------------
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
print("task_id:", task_id)

# -----------------------
# Load model
# -----------------------
print("Loading model...")
model = load_emotion()
model_wrapper = EmotionWrapper(model)



# -----------------------
# Load and slice data
# -----------------------
print("Loading and slicing data...")
pairs = load_emotion_data_pairs()
pairs = pairs[args.start_row:args.end_row]

# -----------------------
# Set transformation, perturbs
# -----------------------

# Map SLURM array ID to (transformation, perturb)
all_transforms = ["homo", "invis", "reord", "del"]
perturb_levels = list(range(1, 6))  # perturbation levels 1–5
grid = [(t, p) for t in all_transforms for p in perturb_levels]  # total 20 combinations
transformation_name, perturb_level = grid[task_id]
print(f"[🧠] Task {task_id} → Transformation: {transformation_name}, Perturb: {perturb_level}")

transformations = {
    "homo": WordSwapHomoglyphSwap(),
    "invis": WordSwapInvisibleCharacters(),
    "reord": WordSwapReorderings(),
    "del": WordSwapDeletions()
}
transformation = transformations[transformation_name]

# -----------------------
# Goal function
# -----------------------
goal_function = Emotion(model_wrapper)

# -----------------------
# Run experiment
# -----------------------
print(f"[🚀] Running experiment: {transformation_name} from {args.start_row} to {args.end_row}...")
run_experiment_parallel(
    model_wrapper=model_wrapper,
    pairs=pairs,
    start_idx=args.start_row,
    score_fn=get_lev_score,
    goal_function=goal_function,
    transformation=transformation,
    experiment_name=f"emotion_{transformation_name}_{args.start_row}_{args.end_row - 1}",
    perturb_range=range(perturb_level, perturb_level + 1)
)
