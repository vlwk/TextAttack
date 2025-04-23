import argparse
import os
import torch
from dotenv import load_dotenv

from textattack.goal_functions import MaximizeLevenshtein
from textattack.transformations import (
    WordSwapHomoglyphSwap,
    WordSwapInvisibleCharacters,
    WordSwapReorderings,
    WordSwapDeletions
    
)
from textattack.models.wrappers import FairseqTranslationWrapper

from imperceptible_experiments.utils.runner_thread import run_experiment_thread
from imperceptible_experiments.translation.load_data import load_en_fr_data
from imperceptible_experiments.utils.scoring import get_lev_score

# -----------------------
# CLI args
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("start_row", type=int)
parser.add_argument("end_row", type=int)
parser.add_argument("transformation", choices=["homo", "invis", "reord", "del"])
parser.add_argument("start_perturb", type=int)
parser.add_argument("end_perturb", type=int)
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
num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
print("task_id:", task_id)
print("num_tasks:", num_tasks)

# -----------------------
# Load model
# -----------------------
print("Loading model...")
model = torch.hub.load(
    'pytorch/fairseq',
    'transformer.wmt14.en-fr',
    tokenizer='moses',
    bpe='subword_nmt',
    verbose=False
).eval()
model_wrapper = FairseqTranslationWrapper(model)

# -----------------------
# Load and slice data
# -----------------------
print("Loading and slicing data...")
pairs = load_en_fr_data()
pairs = sorted(pairs, key=lambda x: len(x[0]))
chunk_size = (args.end_row - args.start_row + num_tasks - 1) // num_tasks
start_idx = args.start_row + task_id * chunk_size
end_idx = min(start_idx + chunk_size, args.end_row)
pairs = pairs[start_idx:end_idx]

# -----------------------
# Set transformation
# -----------------------
transformations = {
    "homo": WordSwapHomoglyphSwap(),
    "invis": WordSwapInvisibleCharacters(),
    "reord": WordSwapReorderings(),
    "del": WordSwapDeletions()
}
transformation = transformations[args.transformation]

# -----------------------
# Goal function
# -----------------------
goal_function = MaximizeLevenshtein(model_wrapper)

# -----------------------
# Run experiment
# -----------------------
print(f"[🚀] Running experiment: {args.transformation} from {start_idx} to {end_idx}...")
run_experiment_thread(
    model_wrapper=model_wrapper,
    pairs=pairs,
    start_idx=start_idx,
    score_fn=get_lev_score,
    goal_function=goal_function,
    transformation=transformation,
    experiment_name=f"fairseq_translation_{args.transformation}_lev_{start_idx}_{end_idx - 1}",
    perturb_range=range(args.start_perturb, args.end_perturb)
)
