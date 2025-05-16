from textattack.models.wrappers import ModelWrapper
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class T5BhadreshWrapper(ModelWrapper):
    def __init__(self, classifier_model, t5_model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            classifier_model: A Hugging Face pipeline or model callable that returns classification results
            t5_model_path: Path to your fine-tuned T5 pre-tokeniser (local dir or Hugging Face Hub)
        """
        self.model = classifier_model
        self.device = device
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_path)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path).to(self.device)
        self.t5_model.eval()

    def _clean_inputs_with_t5(self, input_texts: List[str]) -> List[str]:
        cleaned = []
        for i in range(0, len(input_texts), 8):
            batch = input_texts[i:i+8]
            inputs = self.t5_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.t5_model.generate(**inputs, max_length=128)
            decoded = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            cleaned.extend(decoded)
        return cleaned

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        """
        Args:
            input_texts: List[str]

        Return:
            ret: List[List[float]]
            A list of elements, one per input. Each element is a list of class probabilities.
        """
        cleaned_inputs = self._clean_inputs_with_t5(input_texts)
        ret = []
        for i in cleaned_inputs:
            pred = self.model(i)[0]
            scores = [j['score'] for j in pred]
            ret.append(scores)
        return ret
