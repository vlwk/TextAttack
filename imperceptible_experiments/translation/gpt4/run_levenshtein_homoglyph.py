from textattack.goal_functions import MaximizeLevenshtein
from textattack.transformations import WordSwapHomoglyphSwap
from textattack.models.wrappers import GPT4Wrapper

from imperceptible_experiments.utils.runner import run_experiment
from imperceptible_experiments.translation.load_data import load_en_fr_data
from imperceptible_experiments.utils.scoring import get_lev_score

from dotenv import load_dotenv
import os

load_dotenv()

model_wrapper = GPT4Wrapper(system_prompt=(
    "You are a French translation assistant. Translate any English sentence into French. "
    "Only respond with the translation. Do not include explanations or extra formatting."
))

pairs = load_en_fr_data()
goal_function = MaximizeLevenshtein(model_wrapper)
transformation = WordSwapHomoglyphSwap()

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
