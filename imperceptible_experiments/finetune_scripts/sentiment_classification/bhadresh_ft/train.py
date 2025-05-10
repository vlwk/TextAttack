import os
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import ast

ALL_FILES = [
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/clean_full_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/homoglyphs_full_1to5_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/deletions_full_1to5_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/invisible_full_1to5_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/reorderings_full_1to5_train_annotated.csv",
]

MODEL_DIR = "models/bhadresh_ft/checkpoints"
VAL_RATIO = 0.1
NUM_EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5

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
            "label": torch.tensor(self.labels[i])
        }

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["label"].tolist())
    return accuracy_score(labels, preds)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer = DistilBertTokenizerFast.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    model = DistilBertForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion").cuda()

    df_all = pd.concat([pd.read_csv(path) for path in ALL_FILES], ignore_index=True)
    train_df, val_df = train_test_split(df_all, test_size=VAL_RATIO, random_state=42)

    train_ds = EmotionDataset(train_df, tokenizer)
    val_ds = EmotionDataset(val_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR)
    best_acc = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["label"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate(model, val_loader, device="cuda")
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"New best model saved at {MODEL_DIR}")

    print(f"\nTraining complete. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
