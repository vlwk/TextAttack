import argparse
import textattack
from imperceptible_experiments.model_wrappers.machine_translation.fairseq_en_fr import FairseqEnFrWrapper
from imperceptible_experiments.datasets.machine_translation.newstest2024_en_fr.load import download_en_fr_dataset, load_en_fr_dataset
import torch

# --------------------------
# Argument Parsing
# --------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--perturbs_start_incl", type=int, required=True)
parser.add_argument("--perturbs_end_excl", type=int, required=True)
parser.add_argument("--perturbation_type", type=str, required=True, choices=["homoglyphs", "invisible", "deletions", "reorderings"])
parser.add_argument("--target_distance", type=float, default=0.1)
parser.add_argument("--popsize", type=int, default=5)
parser.add_argument("--maxiter", type=int, default=3)
parser.add_argument("--num_examples", type=int, required=True)
args = parser.parse_args()

# --------------------------
# Argument Checks
# --------------------------

assert args.perturbs_start_incl < args.perturbs_end_excl, "perturbs_start_incl must be less than perturbs_end_excl"
assert args.popsize > 0, "popsize must be positive"
assert args.maxiter > 0, "maxiter must be positive"
assert args.num_examples > 0, "num_examples must be positive"

# --------------------------
# Model & Dataset Setup
# --------------------------

model = torch.hub.load(
    'pytorch/fairseq',
    'transformer.wmt14.en-fr',
    tokenizer='moses',
    bpe='subword_nmt',
    verbose=False
).eval()
model_wrapper = FairseqEnFrWrapper(model)

download_en_fr_dataset()
dataset = load_en_fr_dataset()

assert args.num_examples <= len(dataset), f"num_examples must be ≤ {len(dataset)}"

# --------------------------
# Run Attacks
# --------------------------

for pert in range(args.perturbs_start_incl, args.perturbs_end_excl):
    attack = textattack.attack_recipes.BadCharacters2021.build(
        model_wrapper=model_wrapper,
        goal_function_type="maximize_levenshtein",
        perturbation_type=args.perturbation_type,
        perturbs=pert,
        popsize=args.popsize,
        maxiter=args.maxiter,
    )

    checkpoint_dir = (
        f"results/machine_translation/newstest2024_en_fr/"
        f"num{args.num_examples}/fairseq_en_fr/"
        f"pop{args.popsize}_iter{args.maxiter}"
        f"{args.perturbation_type}/pert{pert}/"
    )

    log_to_csv = f"{checkpoint_dir}/log.csv"

    attack_args = textattack.AttackArgs(
        num_examples=args.num_examples,
        checkpoint_interval=10,
        checkpoint_dir=checkpoint_dir,
        log_to_csv=log_to_csv
    )

    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()
