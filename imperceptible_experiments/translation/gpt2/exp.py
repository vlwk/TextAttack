import textattack
from textattack.goal_functions import MinimizeBleu2
from textattack.search_methods import ImperceptibleDE
from textattack.transformations import WordSwapHomoglyphSwap

from imperceptible_experiments.utils.run_textattack_pairs import run_textattack_pairs
from textattack.models.wrappers import GPT2PromptWrapper

from bs4 import BeautifulSoup
import os

# -----------------------
# Setup
# -----------------------
cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)
num_examples = 5

# -----------------------
# Load GPT2 model wrapper
# -----------------------
model_wrapper = GPT2PromptWrapper(model_name="gpt2", max_new_tokens=30)

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
            # Add prompt for GPT2
            # prompt = f"Translate to French: {src}"
            prompt_template = (
                "Translate the following English sentences into French.\n\n"
                "English: I love you.\n"
                "French: Je t’aime.\n\n"
                "English: He died.\n"
                "French: Il est mort.\n\n"
                "English: Make no mistake.\n"
                "French: Ne nous trompons surtout pas.\n\n"
                "English: {src}\n"
                "French:"
            )

            prompt = prompt_template.format(src=src)

            pairs.append((prompt, tgt))

# Sort by length and trim to N examples
pairs.sort(key=lambda x: len(x[0]))
pairs = pairs[:num_examples]

# -----------------------
# Attack setup
# -----------------------
goal_function = MinimizeBleu2(model_wrapper)
constraints = []
transformation = WordSwapHomoglyphSwap()
search_method = ImperceptibleDE(
    popsize=10,
    maxiter=5,
    verbose=True,
    max_perturbs=1
)
attack = textattack.Attack(goal_function, constraints, transformation, search_method)

# -----------------------
# Run attack
# -----------------------
results_path = "results/gpt2_translation_homo_bleu.jsonl"
run_textattack_pairs(
    attack=attack,
    data_pairs=pairs,
    results_path=results_path
)
