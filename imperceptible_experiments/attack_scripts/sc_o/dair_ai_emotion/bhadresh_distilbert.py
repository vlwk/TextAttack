import os
import json
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# === CONFIG ===
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
TEST_FILES = {
    "clean": "datasets/sc_o/emotion_perturbed_test/files/test_annotated/clean_full_test_annotated.csv",
    "deletions": "datasets/sc_o/emotion_perturbed_test/files/test_annotated/deletions_full_1to10_test_annotated.csv",
    "homoglyphs": "datasets/sc_o/emotion_perturbed_test/files/test_annotated/homoglyphs_full_1to10_test_annotated.csv",
    "invisible": "datasets/sc_o/emotion_perturbed_test/files/test_annotated/invisible_full_1to10_test_annotated.csv",
    "reorderings": "datasets/sc_o/emotion_perturbed_test/files/test_annotated/reorderings_full_1to10_test_annotated.csv",
}
BATCH_SIZE = 32
OUTPUT_DIR = os.path.join("results/sc_o/emotion_perturbed_test", "bhadresh_distilbert")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === DATASET ===
class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["input"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        tokens = self.tokenizer(
            self.texts[i], truncation=True, padding="max_length",
            max_length=128, return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i]),
            "raw_text": self.texts[i]
        }

# === EVALUATION ===
def evaluate(model, loader, device):
    model.eval()
    all_results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].tolist()
            raw_texts = batch["raw_text"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            for i in range(len(labels)):
                all_results.append({
                    "input": raw_texts[i],
                    "true_label": labels[i],
                    "predicted_label": preds[i].item(),
                    "logits": logits[i].cpu().tolist(),
                    "probs": probs[i].cpu().tolist()
                })

    return all_results

# === MAIN ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    print("Running one-shot tests with original pretrained model...\n")
    for name, path in TEST_FILES.items():
        print(f"Testing: {name.upper()}")

        df = pd.read_csv(path)
        dataset = EmotionDataset(df, tokenizer)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        results = evaluate(model, loader, device)
        acc = accuracy_score([r["true_label"] for r in results],
                             [r["predicted_label"] for r in results])
        print(f"Accuracy ({name}): {acc:.4f}")

        out_path = os.path.join(OUTPUT_DIR, f"logits_{name}.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        print(f"Saved: {out_path}\n")

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
