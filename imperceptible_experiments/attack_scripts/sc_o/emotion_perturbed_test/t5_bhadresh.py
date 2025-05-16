import os
import json
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# === CONFIG ===
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
T5_DIR = "models/t5_ft_perturbed"
BATCH_SIZE = 32
MAX_LEN = 128

TEST_FILES = {
    "clean": "datasets/sc_o/emotion_perturbed/files/test_annotated/clean_full_test_annotated.csv",
    "deletions": "datasets/sc_o/emotion_perturbed/files/test_annotated/deletions_full_1to10_test_annotated.csv",
    "homoglyphs": "datasets/sc_o/emotion_perturbed/files/test_annotated/homoglyphs_full_1to10_test_annotated.csv",
    "invisible": "datasets/sc_o/emotion_perturbed/files/test_annotated/invisible_full_1to10_test_annotated.csv",
    "reorderings": "datasets/sc_o/emotion_perturbed/files/test_annotated/reorderings_full_1to10_test_annotated.csv",
}

OUTPUT_DIR = os.path.join("results/sc_o/emotion_perturbed", "t5_bhadresh")
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
            max_length=MAX_LEN, return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i]),
            "raw_text": self.texts[i]
        }

# === CLEANING FUNCTION ===
def denoise_inputs_with_t5(texts, t5_tokenizer, t5_model, device, max_len=128):
    cleaned = []
    for i in range(0, len(texts), 16):  # batching to save memory
        batch = texts[i:i+16]
        inputs = t5_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        outputs = t5_model.generate(**inputs, max_length=max_len)
        decoded = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned.extend(decoded)
    return cleaned

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

    # Load models
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    t5_tokenizer = AutoTokenizer.from_pretrained(T5_DIR)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_DIR).to(device)

    print("Running one-shot tests with T5 pre-tokeniser + DistilBERT classifier...\n")
    for name, path in TEST_FILES.items():
        print(f"Testing: {name.upper()}")

        df = pd.read_csv(path)

        if name != "clean":
            print("Cleaning perturbed inputs with T5...")
            df["input"] = denoise_inputs_with_t5(df["input"].tolist(), t5_tokenizer, t5_model, device)

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
