import argparse
import textattack
from imperceptible_experiments.model_wrappers.mt_u.fairseq_en_fr import FairseqEnFrWrapper
from imperceptible_experiments.datasets.mt_u.wmt14enfr.load import download_en_fr_dataset, load_en_fr_dataset
import torch

# --------------------------
# Argument Parsing
# --------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--perturbs_start_incl", type=int, required=True)
parser.add_argument("--perturbs_end_excl", type=int, required=True)
parser.add_argument("--perturbation_type", type=str, required=True, choices=["homoglyphs", "invisible", "deletions", "reorderings"])
parser.add_argument("--popsize", type=int, default=5)
parser.add_argument("--num_examples", type=int, default=50)
args = parser.parse_args()

# --------------------------
# Model Loading
# --------------------------

# Pre-load the model in the main process
torch.serialization.add_safe_globals([argparse.Namespace])
model = torch.hub.load(
    'pytorch/fairseq',
    'transformer.wmt14.en-fr',
    tokenizer='moses',
    bpe='subword_nmt',
    weights_only=False,
    verbose=False
).eval()

model_wrapper = FairseqEnFrWrapper(model)

# --------------------------
# Dataset Loading
# --------------------------

download_en_fr_dataset()
dataset = load_en_fr_dataset()

# --------------------------
# Attack Configuration
# --------------------------

# Configure the attack
if args.perturbation_type == "homoglyphs":
    transformation = textattack.transformations.WordSwapHomoglyphSwap()
elif args.perturbation_type == "invisible":
    transformation = textattack.transformations.WordSwapInvisible()
elif args.perturbation_type == "deletions":
    transformation = textattack.transformations.WordSwapDeletions()
elif args.perturbation_type == "reorderings":
    transformation = textattack.transformations.WordSwapReorderings()

goal_function = textattack.goal_functions.text.MaximizeLevenshtein()

search_method = textattack.search_methods.DifferentialEvolution(
    pop_size=args.popsize,
    max_iters=3,
    max_perturbs=1,
    verbose=False
)

attack = textattack.Attack(
    goal_function=goal_function,
    transformation=transformation,
    search_method=search_method,
    model_wrapper=model_wrapper,
)

# --------------------------
# Run Attack
# --------------------------

# Configure output directory
checkpoint_dir = f"results/mt_u/wmt14enfr/num{args.num_examples}/fairseq_en_fr/pop{args.popsize}_iter3{args.perturbation_type}/pert{args.perturbs_start_incl}"
log_to_csv = f"{checkpoint_dir}/log.csv"

# Configure attack arguments
attack_args = textattack.AttackArgs(
    num_examples=args.num_examples,
    checkpoint_interval=10,
    checkpoint_dir=checkpoint_dir,
    log_to_csv=log_to_csv
)

# Run the attack
attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
