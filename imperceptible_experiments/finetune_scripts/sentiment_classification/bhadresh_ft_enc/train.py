import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import ast

# === CONFIG ===
ALL_FILES = [
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/clean_full_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/homoglyphs_full_1to5_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/deletions_full_1to5_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/invisible_full_1to5_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/reorderings_full_1to5_train_annotated.csv",
]
MODEL_DIR = "models/bhadresh_ft_enc/checkpoints"
VAL_RATIO = 0.1
NUM_EPOCHS = 1
BATCH_SIZE = 16
LR = 2e-5
LAMBDA_CONTRASTIVE = 0.5

# === DATASET ===
class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.input = df["input"].tolist()
        self.original = df["original_text"].tolist()
        self.label = df["label"].tolist()
        self.word_indices = [ast.literal_eval(x) for x in df["word_indices_perturbed"]]
        self.word_indices_clean = [ast.literal_eval(x) for x in df["word_indices_clean"]]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input)

    def __getitem__(self, i):
        tokens_p = self.tokenizer(self.input[i], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        tokens_c = self.tokenizer(self.original[i], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {
            "input_ids": tokens_p["input_ids"].squeeze(0),
            "attention_mask": tokens_p["attention_mask"].squeeze(0),
            "original_input_ids": tokens_c["input_ids"].squeeze(0),
            "original_attention_mask": tokens_c["attention_mask"].squeeze(0),
            "label": torch.tensor(self.label[i]),
            "word_indices": self.word_indices[i],
            "word_indices_clean": self.word_indices_clean[i],
            "raw_input": self.input[i]
        }

def custom_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        elif key in {"word_indices", "word_indices_clean"}:
            collated[key] = [b[key] for b in batch]  # preserve nested structure
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
            num_layers=2
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, word_indices, verbose=False, raw_inputs=None, tokenizer=None):
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
            if verbose and raw_inputs:
                print(f"\nRaw: {raw_inputs[i]}")
                print(f"Word groups: {word_indices[i]}")
                print(f"Emb shape: {w.shape}")

        padded = nn.utils.rnn.pad_sequence(word_embeds, batch_first=True)
        batch_size, max_len, _ = padded.shape
        word_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=hidden.device)
        for i, w in enumerate(word_embeds):
            word_mask[i, :w.size(0)] = False

        transformed = self.transformer(padded, src_key_padding_mask=word_mask)
        pooled = transformed[:, 0, :]
        logits = self.classifier(pooled)
        return logits, pooled

# === LOSS ===
def cosine_contrastive_loss(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1 - F.cosine_similarity(a, b, dim=-1).mean()

# === EVALUATION ===
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_indices = batch["word_indices"]
            logits, _ = model(input_ids, attention_mask, word_indices)
            preds += torch.argmax(logits, dim=-1).cpu().tolist()
            labels += batch["label"]
    return accuracy_score(labels, preds)

# === MAIN ===
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer = DistilBertTokenizerFast.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    base_model = DistilBertModel.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    model = WordEncoder(base_model).cuda()

    df = pd.concat([pd.read_csv(f) for f in ALL_FILES], ignore_index=True)
    train_df, val_df = train_test_split(df, test_size=VAL_RATIO, random_state=42)
    train_ds = EmotionDataset(train_df, tokenizer)
    val_ds = EmotionDataset(val_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    optimizer = AdamW(model.parameters(), lr=LR)
    best_acc = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}")
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            original_ids = batch["original_input_ids"].cuda()
            original_mask = batch["original_attention_mask"].cuda()
            word_indices = batch["word_indices"]
            word_indices_clean = batch["word_indices_clean"]
            labels = batch["label"].cuda()
            raw_inputs = batch["raw_input"]

            verbose = (step % 500 == 0)
            logits, emb_perturbed = model(input_ids, attention_mask, word_indices, verbose=verbose, raw_inputs=raw_inputs, tokenizer=tokenizer)
            logits_clean, emb_clean = model(original_ids, original_mask, word_indices_clean)

            loss_cls = F.cross_entropy(logits, labels)
            loss_clean_cls = F.cross_entropy(logits_clean, labels)
            loss_ctr = cosine_contrastive_loss(emb_perturbed, emb_clean)
            loss = loss_cls + loss_clean_cls + LAMBDA_CONTRASTIVE * loss_ctr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate(model, val_loader, device="cuda")
        print(f"Validation Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))
            tokenizer.save_pretrained(MODEL_DIR)
            print("Best model saved")

    print(f"\nTraining complete. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
