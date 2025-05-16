import os
import shutil
import tarfile
import requests
import importlib
import subprocess
import argparse
import sys

import pandas as pd
import numpy as np
from typing import List
import zipfile
from io import BytesIO
import json

import textattack
from textattack.datasets import Dataset

from imperceptible_experiments.model_wrappers.te_t.fairseq_roberta_mnli import FairseqMnliWrapper
import torch  

# --------------------------
# Argument Parsing
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--perturbs_start_incl", type=int, required=True)
parser.add_argument("--perturbs_end_excl", type=int, required=True)
parser.add_argument("--perturbation_type", type=str, required=True, choices=["homoglyphs", "invisible", "deletions", "reorderings"])
parser.add_argument("--popsize", type=int, default=5)
parser.add_argument("--maxiter", type=int, default=3)
parser.add_argument("--num_examples", type=int, required=True)
args = parser.parse_args()
if __name__=='__main__':
    assert args.perturbs_start_incl < args.perturbs_end_excl, "perturbs_start_incl must be less than perturbs_end_excl"
    assert args.popsize > 0, "popsize must be positive"
    assert args.maxiter > 0, "maxiter must be positive"
    assert args.num_examples > 0, "num_examples must be positive"
    model = torch.hub.load('pytorch/fairseq',
        'roberta.large.mnli').eval()
    model_wrapper = FairseqMnliWrapper(model)
    url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    os.makedirs("temp/mnli", exist_ok=True)  
    response = requests.get(url)
    response.raise_for_status() 
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("temp/mnli")
    label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    pairs = []
    with open("temp/mnli/multinli_1.0/multinli_1.0_dev_matched.jsonl", 'r') as f:
        for line in f:
            sample = json.loads(line)
            if sample['gold_label'] not in label_map:
                continue

            premise = sample['sentence1'].strip()
            hypothesis = sample['sentence2'].strip()
            label = label_map[sample['gold_label']]
            item = ((premise, hypothesis), label)
            pairs.append(item)
    dataset = textattack.datasets.Dataset(pairs, input_columns=["premise", "hypothesis"])
    gpu_available = True if torch.cuda.is_available() else False
    for pert in range(args.perturbs_start_incl, args.perturbs_end_excl):
        print(f"Running attack with perturbs = {pert}")
        attack = textattack.attack_recipes.BadCharacters2021.build(
            model_wrapper, 
            goal_function_type="targeted_strict", 
            perturbation_type=args.perturbation_type,
            perturbs=pert,
            popsize=args.popsize,
            maxiter=args.maxiter,
        )
        checkpoint_dir = (
            f"results/te_t/mnli/"
            f"num{args.num_examples}/fairseq_roberta_mnli/"
            f"pop{args.popsize}_iter{args.maxiter}/"
            f"{args.perturbation_type}/pert{pert}"
        )
        log_to_csv = os.path.join(checkpoint_dir, "log.csv")
        attack_args = textattack.AttackArgs(
            num_examples=args.num_examples,
            checkpoint_interval=10,
            checkpoint_dir=checkpoint_dir,
            log_to_csv=log_to_csv,
            parallel=gpu_available,
            disable_stdout=True,
        )
        attacker = textattack.Attacker(attack, dataset, attack_args)
        attacker.attack_dataset()