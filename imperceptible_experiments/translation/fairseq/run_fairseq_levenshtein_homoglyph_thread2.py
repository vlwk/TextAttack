from textattack.goal_functions import MaximizeLevenshtein
from textattack.transformations import WordSwapHomoglyphSwap
from textattack.models.wrappers import FairseqTranslationWrapper

from imperceptible_experiments.utils.runner_thread import run_experiment_thread
from imperceptible_experiments.translation.load_data import load_en_fr_data
from imperceptible_experiments.utils.scoring import get_lev_score
import torch

from dotenv import load_dotenv
import os


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

# -----------------------
# Load model
# -----------------------
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
TOTAL_ROWS = 100
pairs = load_en_fr_data()
pairs = sorted(pairs, key=lambda x: len(x[0]))
pairs = pairs[:TOTAL_ROWS]
chunk_size = (len(pairs) + num_tasks - 1) // num_tasks  # ceil division
start_idx = task_id * chunk_size
end_idx = min(start_idx + chunk_size, len(pairs))

# -----------------------
# Goal function
# -----------------------
goal_function = MaximizeLevenshtein(model_wrapper)

# -----------------------
# Transformation
# -----------------------
transformation = WordSwapHomoglyphSwap()

# -----------------------
# Run experiment
# -----------------------

print(f"[SLURM] Task {task_id}/{num_tasks} running on slice {start_idx}:{end_idx}")
print(f"[Model] Fairseq transformer.wmt14.en-fr loaded.")
print(f"[Data] Total pairs: {len(pairs)} | This slice: {end_idx - start_idx}")

run_experiment_thread(
    model_wrapper=model_wrapper,
    pairs=pairs,
    start_idx=start_idx,
    end_idx=end_idx,
    score_fn=get_lev_score,
    goal_function=goal_function,
    transformation=transformation,
    experiment_name=f"fairseq_translation_homo_lev_{start_idx}_{end_idx - 1}",
    perturb_range=range(3, 4)
)
