from textattack.models.wrappers import ModelWrapper
from transformers import DistilBertTokenizerFast
import torch
import torch.nn.functional as F
from typing import List
from imperceptible_experiments.finetune_scripts.sentiment_classification.bhadresh_ft_enc.train import WordEncoder 

class BhadreshFtEncWrapper(ModelWrapper):
    def __init__(self, model: WordEncoder, tokenizer: str = "bhadresh-savani/distilbert-base-uncased-emotion"):
        self.model = model.eval().cuda()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer)

    def _compute_word_indices(self, text: str):
        tokens = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
        offsets = tokens["offset_mapping"]
        input_ids = tokens["input_ids"]
        text_words = text.split()
        word_indices = []
        char_pointer = 0
        token_pointer = 0
        for word in text_words:
            word_start = text.find(word, char_pointer)
            word_end = word_start + len(word)
            group = []
            for i, (start, end) in enumerate(offsets):
                if start >= word_start and end <= word_end:
                    group.append(i)
            if group:
                word_indices.append(group)
            char_pointer = word_end
        return word_indices

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        results = []
        for text in input_texts:
            tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            word_indices = [self._compute_word_indices(text)]
            input_ids = tokens["input_ids"].cuda()
            attention_mask = tokens["attention_mask"].cuda()
            logits, _ = self.model(input_ids, attention_mask, word_indices)
            probs = F.softmax(logits, dim=-1)
            results.append(probs[0].tolist())
        return results
