from textattack.transformations import (
    WordSwapHomoglyphSwap,
    WordSwapDeletions,
    WordSwapInvisibleCharacters,
    WordSwapReorderings
)
from imperceptible_experiments.datasets.sc_o.emotion_perturbed.perturbation_data_generator import EmotionPerturbationDatasetGenerator
from datasets import load_dataset

def index_to_transformation(p):
    return {
        1: WordSwapHomoglyphSwap,
        2: WordSwapDeletions,
        3: WordSwapInvisibleCharacters,
        4: WordSwapReorderings
    }[p]

PERTURBATION_NAMES = {
    1: "homoglyphs",
    2: "deletions",
    3: "invisible",
    4: "reorderings"
}

def generate_datasets(split: str, perturbation_depth: int, base_pct: float):
    dataset = load_dataset("dair-ai/emotion", split=split)
    base_texts = [(ex["text"], ex["label"]) for ex in dataset]
    gen = EmotionPerturbationDatasetGenerator(base_texts, index_to_transformation=index_to_transformation)

    gen.generate_control_dataset(f"datasets/sc_o/emotion_perturbed/files/{split}/clean_full_{split}.csv")

    for p in [1, 2, 3, 4]:
        perturbation_allocation = {
            tuple([p] * n): base_pct for n in range(1, perturbation_depth + 1)
        }
        gen.generate_full_dataset(
            perturbation_allocation,
            f"datasets/sc_o/emotion_perturbed/files/{split}/{PERTURBATION_NAMES[p]}_full_1to{perturbation_depth}_{split}.csv"
        )

# Generate TRAIN with 1 to 5 perturbations, 20% each
generate_datasets(split="train", perturbation_depth=5, base_pct=0.2)

# Generate TEST with 1 to 10 perturbations, 10% each
generate_datasets(split="test", perturbation_depth=10, base_pct=0.1)
