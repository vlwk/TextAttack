import os
import json
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from finetune_scripts.sentiment_classification.bhadresh_ft_enc.train import WordEncoder 

# === CONFIG ===
MODEL_DIR = "models/bhadresh_ft_enc/checkpoints"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
TEST_FILES = {
    "clean": "datasets/sentiment_classification/dair_ai_emotion/oneshot/clean_full_test_annotated.csv",
    "deletions": "datasets/sentiment_classification/dair_ai_emotion/oneshot/deletions_full_1to10_test_annotated.csv",
    "homoglyphs": "datasets/sentiment_classification/dair_ai_emotion/oneshot/homoglyphs_full_1to10_test_annotated.csv",
    "invisible": "datasets/sentiment_classification/dair_ai_emotion/oneshot/invisible_full_1to10_test_annotated.csv",
    "reorderings": "datasets/sentiment_classification/dair_ai_emotion/oneshot/reorderings_full_1to10_test_annotated.csv",
}
BATCH_SIZE = 32
OUTPUT_DIR = os.path.join("results/sentiment_classification/dair_ai_emotion", "oneshot", "bhadresh_ft_enc")
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
            logits, _ = model(input_ids, attention_mask, word_indices=[[[0]]] * len(input_ids))  # dummy word_indices
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

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    base_model = DistilBertModel.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    model = WordEncoder(base_model).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model loaded.")

    print("Running one-shot tests...\n")
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
