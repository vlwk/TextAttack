import os
import torch
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from Levenshtein import distance as lev

# ---------- Config ----------
os.environ["WANDB_DISABLED"] = "true"
MODEL_ID   = "t5-base"
OUT_DIR    = "models/t5_ft"
VAL_SPLIT  = 0.03
BATCH      = 16
EPOCHS     = 3
LR         = 5e-4
SEED       = 42
MAX_LEN    = 512

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- Shared ----------
def preprocess_function(tokenizer, max_len):
    def preprocess(batch):
        model_in = tokenizer(batch["input"], padding="longest",
                             truncation=True, max_length=max_len)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["original_text"], padding="longest",
                               truncation=True, max_length=max_len).input_ids
        model_in["labels"] = labels
        return model_in
    return preprocess

def get_trainer(dataset_csv, model_dir=None, save_dir=OUT_DIR):
    # Load dataset
    ds = load_dataset("csv", data_files=dataset_csv)["train"].shuffle(seed=SEED)
    split = ds.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID if model_dir is None else model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID if model_dir is None else model_dir)

    # Preprocess
    preprocess = preprocess_function(tokenizer, MAX_LEN)
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    # Data collator
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training args
    args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        do_eval=False,
        generation_max_length=MAX_LEN,
        fp16=False,
        label_smoothing_factor=0.1,
        save_total_limit=1,
        seed=SEED,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator
    )

    return trainer, tokenizer

# ---------- Step 1: Train on clean ----------
def step1_train_clean():
    trainer, tokenizer = get_trainer("datasets/t5/ag_news/files/train/clean_full_train.csv")
    trainer.train()
    finetune_dir = OUT_DIR + "_clean"
    trainer.save_model(finetune_dir)
    tokenizer.save_pretrained(finetune_dir)
    print("Step 1 done – clean training saved to", finetune_dir)

# ---------- Step 2: Continue on perturbed ----------
def step2_train_perturbed():
    files = [
        "datasets/t5/ag_news/files/train/homoglyphs_full_1to5_train.csv",
        "datasets/t5/ag_news/files/train/deletions_full_1to5_train.csv",
        "datasets/t5/ag_news/files/train/invisible_full_1to5_train.csv",
        "datasets/t5/ag_news/files/train/reorderings_full_1to5_train.csv"
    ]
    
    merged_csv = "datasets/t5/ag_news/files/train/perturbed_combined_train.csv"
    (
        pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .to_csv(merged_csv, index=False)
    )

    finetune_dir = OUT_DIR + "_perturbed"
    trainer, tokenizer = get_trainer(merged_csv, model_dir=OUT_DIR + "_clean", save_dir=finetune_dir)
    trainer.train()
    trainer.save_model(finetune_dir)
    tokenizer.save_pretrained(finetune_dir)
    print("Step 2 done – adversarial training saved to", finetune_dir)

if __name__ == "__main__":
    step1_train_clean()
    step2_train_perturbed()
