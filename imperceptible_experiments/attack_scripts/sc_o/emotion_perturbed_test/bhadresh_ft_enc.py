import os
import json
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import ast
from torch import nn

# === CONFIG ===
MODEL_DIR = "models/bhadresh_ft_enc/checkpoints"
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "model.pt")

TEST_FILES = {
    "clean": "datasets/sc_o/emotion_perturbed_test/files/test_annotated/clean_full_test_annotated.csv",
    "deletions": "datasets/sc_o/emotion_perturbed_test/files/test_annotated/deletions_full_1to10_test_annotated.csv",
    "homoglyphs": "datasets/sc_o/emotion_perturbed_test/files/test_annotated/homoglyphs_full_1to10_test_annotated.csv",
    "invisible": "datasets/sc_o/emotion_perturbed_test/files/test_annotated/invisible_full_1to10_test_annotated.csv",
    "reorderings": "datasets/sc_o/emotion_perturbed_test/files/test_annotated/reorderings_full_1to10_test_annotated.csv",
}
BATCH_SIZE = 32
OUTPUT_DIR = os.path.join("results/sc_o/emotion_perturbed_test", "bhadresh_ft_enc")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === DATASET ===
class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["input"].tolist()
        self.labels = df["label"].tolist()
        self.word_indices = [ast.literal_eval(x) for x in df["word_indices_perturbed"]]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        tokens = self.tokenizer(
            self.texts[i], truncation=True, padding="max_length", max_length=128, return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i]),
            "word_indices": self.word_indices[i],
            "raw_text": self.texts[i],
        }

def custom_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        elif key == "word_indices":
            collated[key] = [b[key] for b in batch]
        else:
            collated[key] = [b[key] for b in batch]
    return collated

# === MODEL ===
class WordEncoder(nn.Module):
    def __init__(self, base_model, hidden_size=768, num_labels=6):
        super().__init__()
        self.encoder = base_model
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4),
            num_layers=6
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, word_indices):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hidden[:, 0, :]
        batch_size, seq_len, hidden_size = hidden.shape

        word_embeds = []
        for i in range(batch_size):
            vecs = []
            for group in word_indices[i]:
                if all(idx < seq_len for idx in group):
                    vecs.append(hidden[i, group].mean(dim=0))
            w = torch.stack(vecs) if vecs else torch.zeros((1, hidden_size), device=hidden.device)
            w = torch.cat([cls[i].unsqueeze(0), w], dim=0)
            word_embeds.append(w)

        padded = nn.utils.rnn.pad_sequence(word_embeds, batch_first=True)
        transformed = self.transformer(padded)
        pooled = transformed[:, 0, :]
        logits = self.classifier(pooled)
        return logits

# === EVALUATION ===
def evaluate(model, loader, device):
    model.eval()
    all_results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_indices = batch["word_indices"]
            labels = batch["label"].tolist()
            raw_texts = batch["raw_text"]

            logits = model(input_ids, attention_mask, word_indices)
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
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model.eval()

    print("Running one-shot tests...\n")
    for name, path in TEST_FILES.items():
        print(f"Testing: {name.upper()}")
        df = pd.read_csv(path)
        dataset = EmotionDataset(df, tokenizer)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

        results = evaluate(model, loader, device)
        acc = accuracy_score([r["true_label"] for r in results], [r["predicted_label"] for r in results])
        print(f"Accuracy ({name}): {acc:.4f}")

        out_path = os.path.join(OUTPUT_DIR, f"logits_{name}.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        print(f"Saved: {out_path}\n")

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
