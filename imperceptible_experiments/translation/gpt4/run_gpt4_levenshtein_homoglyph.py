from textattack.goal_functions import MaximizeLevenshtein
from textattack.transformations import WordSwapHomoglyphSwap
from textattack.models.wrappers import GPT4Wrapper

from imperceptible_experiments.utils.runner import run_experiment
from imperceptible_experiments.translation.load_data import load_en_fr_data
from imperceptible_experiments.utils.scoring import get_lev_score

from dotenv import load_dotenv
import os

# -----------------------
# Setup
# -----------------------
load_dotenv()
cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)

# -----------------------
# Load model
# -----------------------
model_wrapper = GPT4Wrapper(system_prompt=(
    "You are a French translation assistant. Translate any English sentence into French. "
    "Only respond with the translation. Do not include explanations or extra formatting."
))

# -----------------------
# Load data
# -----------------------
pairs = load_en_fr_data()

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
run_experiment(
    model_wrapper=model_wrapper,
    pairs=pairs,
    score_fn=get_lev_score,
    goal_function=goal_function,
    transformation=transformation,
    experiment_name="gpt4_translation_homo_lev",
    num_examples=5,
    perturb_range=range(0, 2)
)
