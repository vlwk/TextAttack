from textattack.models.wrappers import ModelWrapper
from typing import List

class BhadreshDistilbertWrapper(ModelWrapper):

    def __init__(self, model):
        self.model = model

    def __call__(self, input_texts: List[str]) -> List[List[float]]:

        """
        Args:
            input_texts: List[str]

        Return:
            ret: List[List[float]]
            a list of elements, one per element of input_texts. Each element is a list of probabilities, one for each label.
        """
        ret = []
        for i in input_texts:
            pred = self.model(i)[0]
            scores = []
            for j in pred:
                scores.append(j['score'])
            ret.append(scores)
        return ret
