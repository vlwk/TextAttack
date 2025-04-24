from textattack import Trainer, TrainingArgs
from textattack.datasets import Dataset
from textattack.models.wrappers import MarianWrapper
from textattack.goal_functions import MaximizeLevenshtein
from textattack.search_methods import ImperceptibleDE
from textattack import Attack
from textattack.transformations import WordSwapHomoglyphSwap

# Load model
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
model_wrapper = MarianWrapper(model.to("cuda"), tokenizer)

# Slice your data (assume pairs are loaded)
pairs = load_en_fr_data()[1000:1001]
train_dataset = Dataset(pairs)

# Define attack
goal_function = MaximizeLevenshtein(model_wrapper)
transformation = WordSwapHomoglyphSwap()  # or any you like
search_method = ImperceptibleDE()
attack = Attack(goal_function, [], transformation, search_method)

# TrainingArgs
training_args = TrainingArgs(
    num_epochs=3,
    num_clean_epochs=0,
    num_train_adv_examples=-1,  # Use all examples
    parallel=True,
    attack_num_workers_per_device=1,
    per_device_train_batch_size=8,
    output_dir="marian_adv_training",
    log_to_tb=False,
    save_last=True,
    load_best_model_at_end=True,
)

# Initialise Trainer
trainer = Trainer(
    model_wrapper=model_wrapper,
    task_type="regression",  # <- not really regression, but you're using Levenshtein
    attack=attack,
    train_dataset=train_dataset,
    eval_dataset=None,  # optional
    training_args=training_args,
)

# Start training
trainer.train()
