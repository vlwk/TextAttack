"""
Goal Function for Attempts to minimize the BLEU score
-------------------------------------------------------


"""

import functools

import nltk

import textattack

from .text_to_text_goal_function import TextToTextGoalFunction


class MaximizeBleu(TextToTextGoalFunction):
    """Attempts to maximize the BLEU score between the current output
    translation and the reference translation.

        BLEU score was defined in (BLEU: a Method for Automatic Evaluation of Machine Translation).

        `ArxivURL`_

    .. _ArxivURL: https://www.aclweb.org/anthology/P02-1040.pdf
    """

    def __init__(self, *args, target_bleu=None, **kwargs):
        self.target_bleu = target_bleu
        super().__init__(*args, **kwargs)

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()
        get_bleu.cache_clear()

    def _is_goal_complete(self, model_output, _):
        if self.target_bleu:
            bleu_score = 1.0 - self._get_score(model_output, _)
            return bleu_score >= self.target_bleu
        else:
            return False

    def _get_score(self, model_output, _):

        """
        model_output (str): Stores the output of the text-to-text model.
        ground_truth_output: The expected output.

        When used with ImperceptibleDE, this method maximizes the Bleu score between model_output and ground_truth_output.
        """

        model_output_at = textattack.shared.AttackedText(model_output)
        ground_truth_at = textattack.shared.AttackedText(self.ground_truth_output)
        bleu_score = get_bleu(model_output_at, ground_truth_at)
        return 1.0 - bleu_score

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_bleu"]


@functools.lru_cache(maxsize=2**12)
def get_bleu(a, b):
    ref = a.words
    hyp = b.words
    bleu_score = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
    return bleu_score
