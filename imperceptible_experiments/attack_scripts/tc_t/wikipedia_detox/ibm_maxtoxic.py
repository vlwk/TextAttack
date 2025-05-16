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

import textattack
from textattack.datasets import Dataset

from imperceptible_experiments.model_wrappers.tc_t.ibm_maxtoxic import IBMMAXToxicWrapper
import torch  # Only needed if your model wrapper internally depends on it



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

if __name__ == '__main__':
    # --------------------------
    # Sanity Checks
    # --------------------------
    assert args.perturbs_start_incl < args.perturbs_end_excl, "perturbs_start_incl must be less than perturbs_end_excl"
    assert args.popsize > 0, "popsize must be positive"
    assert args.maxiter > 0, "maxiter must be positive"
    assert args.num_examples > 0, "num_examples must be positive"

    # Download model

    def run(cmd):
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    os.makedirs("temp/toxic", exist_ok=True)
    shutil.rmtree(os.path.join("temp/toxic", "assets"), ignore_errors=True)
    shutil.rmtree(os.path.join("temp/toxic", "toxic"), ignore_errors=True)
    os.makedirs(os.path.join("temp/toxic", "assets"), exist_ok=True)
    url = "https://codait-cos-max.s3.us.cloud-object-storage.appdomain.cloud/max-toxic-comment-classifier/1.0.0/assets.tar.gz"
    tar_path = os.path.join("temp/toxic", "assets/assets.tar.gz")
    print(f"Downloading {url}")
    response = requests.get(url)
    with open(tar_path, "wb") as f:
        f.write(response.content)
    print("Extracting assets...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall("temp/toxic/assets")
    os.remove(tar_path)
    run("git clone https://github.com/IBM/MAX-Toxic-Comment-Classifier.git")
    shutil.move("MAX-Toxic-Comment-Classifier", os.path.join("temp/toxic", "toxic"))
    requirements_path = os.path.join("temp/toxic", "toxic/requirements.txt")
    with open(requirements_path, "r") as f:
        lines = f.readlines()
    with open(requirements_path, "w") as f:
        for line in lines:
            f.write(line.split("==")[0].strip() + "\n")
    run(f"pip install -r {requirements_path}")
    run("pip install maxfw")
    model_py_path = os.path.join("temp/toxic", "toxic/core/model.py")
    with open(model_py_path, "r") as f:
        content = f.read()
    content = content.replace("from config", "from ..config")
    content = content.replace("from core.", "from .")
    with open(model_py_path, "w") as f:
        f.write(content)
    with open("temp/toxic/toxic/config.py", "r") as f:
        content = f.read()
    content = content.replace("assets", "temp/toxic/assets")
    with open("temp/toxic/toxic/config.py", "w") as f:
        f.write(content)
    importlib.invalidate_caches()

    def load_model_wrapper():
        module_name = "imperceptible_experiments.temp.toxic.toxic.core.model"
        
        # Remove the module if it's already loaded
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)
        
        module = sys.modules[module_name]
        return module.ModelWrapper()

    model = load_model_wrapper()
    model_wrapper = IBMMAXToxicWrapper(model)

    # Get data
    if (os.path.exists("temp/toxic/toxicity_annotated_comments.tsv") == False) or (os.path.exists("temp/toxic/toxicity_annotations.tsv") == False): 
        data_urls = {
            "temp/toxic/toxicity_annotated_comments.tsv": "https://ndownloader.figshare.com/files/7394542",
            "temp/toxic/toxicity_annotations.tsv": "https://ndownloader.figshare.com/files/7394539",
        }
        for filename, url in data_urls.items():
            print(f"Downloading {filename}")
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)

    comments = pd.read_csv('temp/toxic/toxicity_annotated_comments.tsv', sep = '\t', index_col = 0)
    annotations = pd.read_csv('temp/toxic/toxicity_annotations.tsv',  sep = '\t')
    # labels a comment as toxic if the majority of annoatators did so
    labels = annotations.groupby('rev_id')['toxicity'].mean() > 0.5
    # join labels and comments
    comments['toxicity'] = labels
    # remove newline and tab tokens
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
    test_comments = comments.query("split=='test'").query("toxicity==True")
    examples = test_comments.reset_index().to_dict('records')

    pairs = [(row["comment"], 0) for row in examples]

    dataset = textattack.datasets.Dataset(pairs)
    # --------------------------
    # Attack Loop
    # --------------------------
    gpu_available = True if torch.cuda.is_available() else False

    for pert in range(args.perturbs_start_incl, args.perturbs_end_excl):
        print(f"Running attack with perturbs = {pert}")

        attack = textattack.attack_recipes.BadCharacters2021.build(
            model_wrapper=model_wrapper,
            goal_function_type="logit_sum",
            perturbation_type=args.perturbation_type,
            perturbs=pert,
            popsize=args.popsize,
            maxiter=args.maxiter,
        )

        checkpoint_dir = (
            f"results/tc_t/wikipedia_detox/"
            f"num{args.num_examples}/ibm_maxtoxic/"
            f"pop{args.popsize}_iter{args.maxiter}/"
            f"{args.perturbation_type}/pert{pert}"
        )
        log_to_csv = os.path.join(checkpoint_dir, "log.csv")

        attack_args = textattack.AttackArgs(
            num_examples=args.num_examples,
            checkpoint_interval=10,
            checkpoint_dir=checkpoint_dir,
            log_to_csv=log_to_csv,
            parallel=gpu_available
        )

        attacker = textattack.Attacker(attack, dataset, attack_args)
        attacker.attack_dataset()