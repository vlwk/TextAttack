from transformers import AutoTokenizer
from imperceptible_experiments.datasets.word_index_annotator import WordIndexAnnotator
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
annotator = WordIndexAnnotator(tokenizer)

# CLEAN TRAIN + TEST
for mode in ["train", "test"]:
    input_path = f"datasets/sc_o/emotion_perturbed_test/files/{mode}/clean_full_{mode}.csv"
    output_path = f"datasets/sc_o/emotion_perturbed_test/files/{mode}_annotated/clean_full_{mode}_annotated.csv"

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Annotating clean_full_{mode}...")
    df = annotator.annotate(df, verbose=False)

    print(f"Saving to: {output_path}\n")
    df.to_csv(output_path, index=False)

# PERTURBED TRAIN
print("Annotating TRAIN datasets...\n")
for perturbation in ["homoglyphs", "deletions", "invisible", "reorderings"]:
    input_path = f"datasets/sc_o/emotion_perturbed_test/files/train/{perturbation}_full_1to5_train.csv"
    output_path = f"datasets/sc_o/emotion_perturbed_test/files/train_annotated/{perturbation}_full_1to5_train_annotated.csv"
    
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Annotating perturbation type: {perturbation}")
    df = annotator.annotate(df, verbose=False)

    print(f"Saving to: {output_path}\n")
    df.to_csv(output_path, index=False)

# PERTURBED TEST
print("\nAnnotating TEST datasets...\n")
for perturbation in ["homoglyphs", "deletions", "invisible", "reorderings"]:
    input_path = f"datasets/sc_o/emotion_perturbed_test/files/test/{perturbation}_full_1to10_test.csv"
    output_path = f"datasets/sc_o/emotion_perturbed_test/files/test_annotated/{perturbation}_full_1to10_test_annotated.csv"

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Annotating perturbation type: {perturbation}")
    df = annotator.annotate(df, verbose=False)

    print(f"Saving to: {output_path}\n")
    df.to_csv(output_path, index=False)



print("Done annotating all datasets.")
