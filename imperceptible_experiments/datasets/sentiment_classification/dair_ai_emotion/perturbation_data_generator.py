from typing import Any, List
from imperceptible_experiments.datasets.perturbation_data_generator import PerturbationDatasetGenerator  # replace with actual import path

class EmotionPerturbationDatasetGenerator(PerturbationDatasetGenerator):
    def extract_fields(self, metadata: Any) -> List[str]:
        return [str(metadata)]  # e.g. label (int or str)

    def get_additional_headers(self) -> List[str]:
        return ["label"]
