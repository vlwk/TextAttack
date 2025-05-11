import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from imperceptible_experiments.model_wrappers.sentiment_classification.gpt4 import GPT4SentimentClassificationWrapper

# === CONFIG ===
TEST_FILES = {
    # "clean": "datasets/sentiment_classification/dair_ai_emotion/oneshot/clean_full_test_annotated.csv",
    "deletions": "datasets/sentiment_classification/dair_ai_emotion/oneshot/deletions_full_1to10_test_annotated.csv",
    "homoglyphs": "datasets/sentiment_classification/dair_ai_emotion/oneshot/homoglyphs_full_1to10_test_annotated.csv",
    "invisible": "datasets/sentiment_classification/dair_ai_emotion/oneshot/invisible_full_1to10_test_annotated.csv",
    "reorderings": "datasets/sentiment_classification/dair_ai_emotion/oneshot/reorderings_full_1to10_test_annotated.csv",
}
BATCH_SIZE = 8
OUTPUT_DIR = "results/sentiment_classification/dair_ai_emotion/oneshot/gpt4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "This sentence is one of the following emotions: sadness (0), joy (1), love (2), "
    "anger (3), fear (4), surprise (5). Output the number of the emotion. Sentence:"
)

# === DATASET ===
class EmotionDataset(Dataset):
    def __init__(self, df):
        self.texts = df["input"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {
            "text": self.texts[i],
            "label": self.labels[i]
        }

# === EVALUATION ===
def evaluate(model_wrapper, loader):
    all_results = []
    for batch in tqdm(loader, desc="Evaluating"):
        texts = [item["text"] for item in batch]
        true_labels = [item["label"] for item in batch]
        preds = model_wrapper(texts)

        for inp, true, pred in zip(texts, true_labels, preds):
            try:
                pred_int = int(pred.strip())
            except:
                pred_int = -1  # fallback if GPT4 returns invalid output
            all_results.append({
                "input": inp,
                "true_label": true,
                "predicted_label": pred_int,
                "raw_prediction": pred
            })
    return all_results

# === MAIN ===
def main():
    model_wrapper = GPT4SentimentClassificationWrapper(system_prompt=SYSTEM_PROMPT, no_classes=6)

    for name, path in TEST_FILES.items():
        print(f"\nTesting: {name.upper()}")
        df = pd.read_csv(path)
        dataset = EmotionDataset(df)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: x)

        results = evaluate(model_wrapper, loader)
        acc = accuracy_score(
            [r["true_label"] for r in results if r["predicted_label"] != -1],
            [r["predicted_label"] for r in results if r["predicted_label"] != -1]
        )
        print(f"Accuracy ({name}): {acc:.4f}")

        out_path = os.path.join(OUTPUT_DIR, f"logits_{name}.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
