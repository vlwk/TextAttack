

import textattack
from textattack.goal_functions import MinimizeBleu2, MaximizeLevenshtein
from textattack.search_methods import ImperceptibleDE
from textattack.transformations import WordSwapHomoglyphSwap

from imperceptible_experiments.utils.run_textattack_pairs import run_textattack_pairs
from imperceptible_experiments.utils.run_baseline_pairs import run_baseline_pairs
from textattack.models.wrappers import GPT4Wrapper

from bs4 import BeautifulSoup
import os
from datetime import datetime

from Levenshtein import distance as levenshtein_distance

# -----------------------
# Setup
# -----------------------
cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)
num_examples = 5

from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env into environment

# -----------------------
# Load GPT-4 model wrapper
# -----------------------
prompt_instruction = (
    "You are a French translation assistant. Translate any English sentence into French. "
    "Only respond with the translation. Do not include explanations or extra formatting."
)

model_wrapper = GPT4Wrapper(system_prompt=prompt_instruction)

# -----------------------
# Load source/target data
# -----------------------
source_path = "../data/newstest2014-fren-src.en.sgm"
target_path = "../data/newstest2014-fren-ref.fr.sgm"

source_doc = BeautifulSoup(open(source_path, 'r'), 'html.parser')
target_doc = BeautifulSoup(open(target_path, 'r'), 'html.parser')

pairs = []

for doc in source_doc.find_all('doc'):
    docid = str(doc['docid'])
    for seg in doc.find_all('seg'):
        segid = str(seg['id'])
        src = str(seg.string)
        tgt_node = target_doc.select_one(f'doc[docid="{docid}"] > seg[id="{segid}"]')
        if tgt_node:
            tgt = str(tgt_node.string)
            # Only use the raw input for perturbation
            pairs.append((src, tgt))

# Sort by length and trim to N examples
pairs.sort(key=lambda x: len(x[0]))
pairs = pairs[:num_examples]

# -----------------------
# Attack setup
# -----------------------
# goal_function = MinimizeBleu2(model_wrapper)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = os.path.join("results", timestamp)
os.makedirs(out_dir, exist_ok=True)

goal_function = MaximizeLevenshtein(model_wrapper)
constraints = []
transformation = WordSwapHomoglyphSwap()

for max_perturbs in range(0, 1):

    results_path = os.path.join(out_dir, f"gpt4_translation_homo_lev_p{max_perturbs}.jsonl")

    def get_score(model_output, ground_truth_output):
        """
        model_output (str): Stores the output of the text-to-text model.
        ground_truth_output: The expected output.

        When used with ImperceptibleDE, this method maximizes the Levenshtein distance between model_output and ground_truth_output.
        """
        distance = levenshtein_distance(model_output, ground_truth_output)

        return -distance

    if (max_perturbs == 0):
        run_baseline_pairs(model_wrapper, get_score, pairs, results_path)
    else:
    
        search_method = ImperceptibleDE(
            popsize=10,
            maxiter=5,
            verbose=True,
            max_perturbs=max_perturbs
        )
        attack = textattack.Attack(goal_function, constraints, transformation, search_method)

        # -----------------------
        # Run attack
        # -----------------------
        # results_path = "results/gpt4_translation_homo_bleu.jsonl"
        
        run_textattack_pairs(
            attack=attack,
            data_pairs=pairs,
            results_path=results_path
        )

