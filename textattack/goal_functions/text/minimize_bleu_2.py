"""
Goal Function to minimize BLEU score
--------------------------------------
This goal function attempts to make the perturbed translation diverge as much as possible from the ground truth translation,
as measured by BLEU score (Papineni et al., 2002).

Reference:
https://www.aclweb.org/anthology/P02-1040.pdf

Also used in:
https://www.aclweb.org/anthology/2020.acl-main.263
"""

import functools
import sacrebleu

import textattack
from .text_to_text_goal_function import TextToTextGoalFunction


class MinimizeBleu2(TextToTextGoalFunction):
    EPS = 1e-10

    def __init__(self, *args, target_bleu=0.0, **kwargs):
        self.target_bleu = target_bleu
        super().__init__(*args, **kwargs)

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()
        get_bleu.cache_clear()

    def _is_goal_complete(self, model_output, _):
        # Disable early stopping by default
        return False

    def _get_score(self, model_output, _):
        model_output_at = textattack.shared.AttackedText(model_output)
        ground_truth_at = textattack.shared.AttackedText(self.ground_truth_output)
        bleu_score = get_bleu(model_output_at, ground_truth_at)
        return bleu_score

    def extra_repr_keys(self):
        return ["maximizable", "target_bleu"]


@functools.lru_cache(maxsize=2**12)
def get_bleu(a, b):
    ref = [" ".join(a.words)]
    hyp = " ".join(b.words)
    try:
        bleu = sacrebleu.corpus_bleu([hyp], [[r] for r in ref])
        return bleu.score
    except Exception:
        return 0.0
