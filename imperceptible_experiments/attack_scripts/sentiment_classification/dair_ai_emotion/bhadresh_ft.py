import textattack
from imperceptible_experiments.model_wrappers.sentiment_classification.bhadresh_distilbert_base_uncased_emotion import BhadreshDistilbertBaseUncasedEmotionWrapper
import argparse
from transformers import pipeline
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--perturbs_start_incl", type=int, required=True)
parser.add_argument("--perturbs_end_excl", type=int, required=True)
parser.add_argument("--perturbation_type", type=str, required=True, choices=["homoglyphs", "invisible", "deletions", "reorderings"])
parser.add_argument("--target_class", type=int, required=True, choices=range(6))  # 0–5 inclusive
parser.add_argument("--popsize", type=int, default=5)
parser.add_argument("--maxiter", type=int, default=3)
parser.add_argument("--num_examples", type=int, required=True)

args = parser.parse_args()

assert args.perturbs_start_incl < args.perturbs_end_excl, f"perturbs_start_incl must be less than perturbs_end_excl"
assert args.popsize > 0, f"popsize must be positive"
assert args.maxiter > 0, f"maxiter must be positive"
assert args.num_examples > 0, f"num_examples must be positive"

device = 0 if torch.cuda.is_available() else -1
model = pipeline("text-classification", model="models/bhadresh_ft/checkpoints", tokenizer="models/bhadresh_ft/checkpoints", return_all_scores=True, device=device)
model_wrapper = BhadreshDistilbertBaseUncasedEmotionWrapper(model)

for pert in range(args.perturbs_start_incl, args.perturbs_end_excl):

    attack = textattack.attack_recipes.BadCharacters2021.build(
        model_wrapper=model_wrapper, 
        goal_function_type="targeted_bonus",
        perturbation_type=args.perturbation_type,
        target_class=args.target_class,
        perturbs=pert,
        popsize=args.popsize,
        maxiter=args.maxiter
    )
    dataset = textattack.datasets.HuggingFaceDataset("dair-ai/emotion", split="test")

    assert args.num_examples <= len(dataset), f"num_examples must be ≤ {len(dataset)}"

    checkpoint_dir = (
        f"results/sentiment_classification/dair_ai_emotion/"
        f"num{args.num_examples}/bhadresh_ft/"
        f"pop{args.popsize}_iter{args.maxiter}/"
        f"{args.perturbation_type}/target{args.target_class}/pert{pert}"
    )
    
    log_to_csv = (
        f"{checkpoint_dir}/log.csv"
    )

    attack_args = textattack.AttackArgs(
        num_examples=args.num_examples,
        checkpoint_interval=10,
        checkpoint_dir=checkpoint_dir,
        log_to_csv=log_to_csv
    )

    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()