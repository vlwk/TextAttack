from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np

paths = [
    "datasets/sc_o/emotion_perturbed/files/train_annotated/clean_full_train_annotated.csv",
    "datasets/sc_o/emotion_perturbed/files/train_annotated/homoglyphs_full_1to5_train_annotated.csv",
    "datasets/sc_o/emotion_perturbed/files/train_annotated/deletions_full_1to5_train_annotated.csv",
    "datasets/sc_o/emotion_perturbed/files/train_annotated/invisible_full_1to5_train_annotated.csv",
    "datasets/sc_o/emotion_perturbed/files/train_annotated/reorderings_full_1to5_train_annotated.csv",
]

tokenizer = DistilBertTokenizerFast.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")

lengths = []

for path in paths:
    df = pd.read_csv(path)
    for text in df["input"]:
        tokens = tokenizer.encode(text, truncation=False)
        lengths.append(len(tokens))

# Compute and print stats
print(f"Number of samples: {len(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Min length: {min(lengths)}")
print(f"Mean length: {np.mean(lengths):.2f}")
print(f"05th percentile length: {np.percentile(lengths, 95):.2f}")