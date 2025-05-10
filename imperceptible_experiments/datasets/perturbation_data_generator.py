import os
import csv
from typing import List, Tuple, Dict, Any, Callable
from textattack.shared import AttackedText
import random

class PerturbationDatasetGenerator:
    def __init__(
        self,
        base_texts: List[Tuple[str, Any]],
        index_to_transformation: Callable[[int], Callable]
    ):
        """
        base_texts: list of (text, metadata)
        index_to_transformation: maps int -> transformation instance
        """
        self.base_texts = base_texts
        self.index_to_transformation = index_to_transformation

    def extract_fields(self, metadata: Any) -> List[str]:
        return []

    def perturb_single_entry(self, text: str, perturb_seq: Tuple[int, ...]) -> str:
        pert_text = AttackedText(text)
        words = pert_text.words
        for p in perturb_seq:
            idx = random.randint(0, len(words) - 1)
            tform = self.index_to_transformation(p)(random_one=True)
            transformed = tform._get_transformations(pert_text, [idx])
            if transformed:
                pert_text = transformed[0]
        return pert_text.text

    def generate_control_dataset(self, path: str, num_examples: int = None):
        if num_examples is None:
            num_examples = len(self.base_texts)
        rows = []
        for text, metadata in self.base_texts[:num_examples]:
            row = [text, text] + self.extract_fields(metadata)
            rows.append(row)
        self._write_csv(path, rows)

    def generate_full_dataset(self, perturbation_allocation: Dict[Tuple[int], float], path: str, num_examples: int = None):
        if num_examples is None:
            num_examples = len(self.base_texts)
        rows = []
        total = num_examples
        current_index = 0

        for seq, pct in perturbation_allocation.items():
            n_samples = int(total * pct)
            if current_index + n_samples > total:
                n_samples = total - current_index
            sampled = self.base_texts[current_index : current_index + n_samples]
            current_index += n_samples

            for text, metadata in sampled:
                perturbed = self.perturb_single_entry(text, seq)
                row = [perturbed, text] + self.extract_fields(metadata)
                rows.append(row)

        self._write_csv(path, rows)


    def _write_csv(self, path: str, rows: List[List[Any]]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["input", "original_text"] + self.get_additional_headers()
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Saved {len(rows)} examples → {path}")

    def get_additional_headers(self) -> List[str]:
        return []
