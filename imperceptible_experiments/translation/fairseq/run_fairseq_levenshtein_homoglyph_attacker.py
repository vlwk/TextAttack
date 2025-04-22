from textattack.goal_functions import MaximizeLevenshtein
from textattack.transformations import WordSwapHomoglyphSwap
from textattack.models.wrappers import FairseqTranslationWrapper
from textattack import Attack, Attacker, AttackArgs

from imperceptible_experiments.translation.load_data import load_en_fr_dataset
import torch
import os
from dotenv import load_dotenv


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
dataset = load_en_fr_dataset()  # already returns textattack.datasets.Dataset

# -----------------------
# Build Attack
# -----------------------
goal_function = MaximizeLevenshtein(model_wrapper)
transformation = WordSwapHomoglyphSwap()

from textattack.search_methods import ImperceptibleDE

search_method = ImperceptibleDE(
    popsize=10,
    maxiter=5,
    verbose=True,
    max_perturbs=1
)

attack = Attack(goal_function, [], transformation, search_method)

# -----------------------
# Setup attack args
# -----------------------
attack_args = AttackArgs(
    num_examples=5,
    disable_stdout=False,
    log_to_csv="results/fairseq_attacker_log.csv",
    checkpoint_interval=5,
    checkpoint_dir="checkpoints/fairseq_attacker",
    parallel=False  # Set to True only if you're on GPU & want parallel GPU attacks
)

# -----------------------
# Run Attacker
# -----------------------
attacker = Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
