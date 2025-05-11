import os
import argparse
import textattack
import torch
from transformers import pipeline
from string import punctuation
from datasets import load_dataset
from typing import List
from textattack.datasets import Dataset
from imperceptible_experiments.model_wrappers.named_entity_recognition.dbmdz_bert_large_cased_finetuned_conll03_english import NERModelWrapper

# --------------------------
# Argument Parsing
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--perturbs_start_incl", type=int, required=True)
parser.add_argument("--perturbs_end_excl", type=int, required=True)
parser.add_argument("--perturbation_type", type=str, required=True, choices=["homoglyphs", "invisible", "deletions", "reorderings"])
parser.add_argument("--target_suffix", type=str, required=True, choices=["PER", "ORG", "LOC", "MISC"])
parser.add_argument("--popsize", type=int, default=5)
parser.add_argument("--maxiter", type=int, default=3)
parser.add_argument("--num_examples", type=int, required=True)
args = parser.parse_args()

assert args.perturbs_start_incl < args.perturbs_end_excl, f"perturbs_start_incl must be less than perturbs_end_excl"
assert args.popsize > 0, f"popsize must be positive"
assert args.maxiter > 0, f"maxiter must be positive"
assert args.num_examples > 0, f"num_examples must be positive"

# --------------------------
# Model Setup
# --------------------------
device = 0 if torch.cuda.is_available() else -1
model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=device)
model_wrapper = NERModelWrapper(model)

# --------------------------
# Dataset Setup
# --------------------------
def detokenize(tokens: List[str]) -> str:
    output = ""
    for index, token in enumerate(tokens):
        if (len(token) == 1 and token in punctuation) or index == 0:
            output += token
        else:
            output += ' ' + token
    return output

raw_dataset = load_dataset("conll2003", split="test", trust_remote_code=True)
examples = [(detokenize(ex["tokens"]), "NER") for ex in raw_dataset]
dataset = Dataset(examples[:args.num_examples])  # Slice to num_examples early

# --------------------------
# Attack Loop
# --------------------------
for pert in range(args.perturbs_start_incl, args.perturbs_end_excl):
    print(f"Running attack with perturbs = {pert}")

    attack = textattack.attack_recipes.BadCharacters2021.build(
        model_wrapper=model_wrapper,
        goal_function_type="named_entity_recognition",
        perturbation_type=args.perturbation_type,
        target_suffix=args.target_suffix,
        perturbs=pert,
        popsize=args.popsize,
        maxiter=args.maxiter,
    )

    checkpoint_dir = (
        f"results/named_entity_recognition/conll2003/"
        f"num{args.num_examples}/dbmdz/"
        f"pop{args.popsize}_iter{args.maxiter}/"
        f"{args.perturbation_type}/target{args.target_suffix}/pert{pert}"
    )
    log_to_csv = os.path.join(checkpoint_dir, "log.csv")

    attack_args = textattack.AttackArgs(
        num_examples=args.num_examples,
        checkpoint_interval=10,
        checkpoint_dir=checkpoint_dir,
        log_to_csv=log_to_csv,
        disable_stdout=True,
        disable_log_color=True
    )

    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()
