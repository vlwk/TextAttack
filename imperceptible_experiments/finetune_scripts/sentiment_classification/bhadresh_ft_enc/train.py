# Full fixed version of your training script using row-wise clean-perturbed contrastive alignment

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
import random

ALL_FILES = [
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/clean_full_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/homoglyphs_full_1to5_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/deletions_full_1to5_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/invisible_full_1to5_train_annotated.csv",
    "datasets/sentiment_classification/dair_ai_emotion/oneshot/reorderings_full_1to5_train_annotated.csv",
]

MODEL_DIR = "models/bhadresh_ft_enc/checkpoints"
VAL_RATIO = 0.1
NUM_EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
LAMBDA_CONTRASTIVE = 0.5

class EmotionDatasetWithPairs(Dataset):
    def __init__(self, df, tokenizer):
        self.perturbed_inputs = df["input"].tolist()
        self.clean_inputs = df["original_text"].tolist()
        self.labels = df["label"].tolist()
        self.word_indices = [ast.literal_eval(x) for x in df["word_indices_perturbed"]]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.perturbed_inputs)

    def __getitem__(self, i):
        perturbed_tokens = self.tokenizer(
            self.perturbed_inputs[i], truncation=True, padding="max_length",
            max_length=128, return_tensors="pt"
        )
        clean_tokens = self.tokenizer(
            self.clean_inputs[i], truncation=True, padding="max_length",
            max_length=128, return_tensors="pt"
        )
        return {
            "input_ids": perturbed_tokens["input_ids"].squeeze(0),
            "attention_mask": perturbed_tokens["attention_mask"].squeeze(0),
            "original_input_ids": clean_tokens["input_ids"].squeeze(0),
            "original_attention_mask": clean_tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i]),
            "word_indices": self.word_indices[i],
            "raw_input": self.perturbed_inputs[i]
        }

def custom_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        else:
            collated[key] = [b[key] for b in batch]
    return collated

class WordEncoder(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.encoder = base_model
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4), num_layers=1
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, word_indices, verbose=False, raw_inputs=None, tokenizer=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch_size, seq_len, hidden_size = hidden.shape
        cls_embeddings = hidden[:, 0, :]

        word_embeds = []
        for i in range(batch_size):
            word_vecs = []
            for group in word_indices[i]:
                if all(idx < seq_len for idx in group):
                    tokens = hidden[i, group, :]
                    word_vecs.append(tokens.mean(dim=0))
            word_embed = torch.stack(word_vecs) if word_vecs else torch.zeros((1, hidden_size), device=hidden.device)
            word_embed = torch.cat([cls_embeddings[i].unsqueeze(0), word_embed], dim=0)
            word_embeds.append(word_embed)

            if verbose and raw_inputs is not None and tokenizer is not None:
                print(f"Sentence {i}: {raw_inputs[i]}")
                print(f"Tokens: {tokenizer.tokenize(raw_inputs[i])}")
                print(f"Input IDs: {input_ids[i].tolist()}")
                print(f"Word groups: {word_indices[i]}")
                print(f"Word embeddings shape (with CLS): {word_embed.shape}")
                print(f"Word embedding sample:\n{word_embed[:5]}")

        padded = nn.utils.rnn.pad_sequence(word_embeds, batch_first=True)
        transformed = self.transformer(padded)
        if verbose:
            print("\nTransformer output sample:\n", transformed[0][:5])
        pooled = transformed[:, 0, :]
        logits = self.classifier(pooled)
        return logits, pooled

def cosine_contrastive_loss(emb1, emb2):
    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)
    return 1 - F.cosine_similarity(emb1, emb2, dim=-1).mean()

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_indices = batch["word_indices"]
            logits, _ = model(input_ids, attention_mask, word_indices)
            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.cpu().tolist())
            labels.extend(batch["label"].tolist())
    return accuracy_score(labels, preds)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer = DistilBertTokenizerFast.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    base_model = DistilBertModel.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    model = WordEncoder(base_model, hidden_size=768, num_labels=6).cuda()

    df_all = pd.concat([pd.read_csv(path) for path in ALL_FILES], ignore_index=True)
    train_df, val_df = train_test_split(df_all, test_size=VAL_RATIO, random_state=42)

    train_ds = EmotionDatasetWithPairs(train_df, tokenizer)
    val_ds = EmotionDatasetWithPairs(val_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    optimizer = AdamW(model.parameters(), lr=LR)
    best_acc = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            original_input_ids = batch["original_input_ids"].cuda()
            original_attention_mask = batch["original_attention_mask"].cuda()
            labels = batch["label"].cuda()
            word_indices = batch["word_indices"]
            raw_inputs = batch["raw_input"]

            verbose = (step % 1000 == 0)
            logits, emb_perturbed = model(input_ids, attention_mask, word_indices,
                                          verbose=verbose, raw_inputs=raw_inputs, tokenizer=tokenizer)
            _, emb_clean = model(original_input_ids, original_attention_mask,
                                 word_indices=[[[0]]] * len(original_input_ids))  # dummy CLS

            loss_cls = F.cross_entropy(logits, labels)
            loss_contrast = cosine_contrastive_loss(emb_perturbed, emb_clean)
            loss = loss_cls + LAMBDA_CONTRASTIVE * loss_contrast

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate(model, val_loader, device="cuda")
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"New best model saved at {MODEL_DIR}")

    print(f"\nTraining complete. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
