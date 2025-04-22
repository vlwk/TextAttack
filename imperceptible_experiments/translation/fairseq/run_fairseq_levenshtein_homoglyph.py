from textattack.goal_functions import MaximizeLevenshtein
from textattack.transformations import WordSwapHomoglyphSwap
from textattack.models.wrappers import FairseqTranslationWrapper

from imperceptible_experiments.utils.runner import run_experiment
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
    experiment_name="fairseq_translation_homo_lev",
    num_examples=5,
    perturb_range=range(0, 2)
)
