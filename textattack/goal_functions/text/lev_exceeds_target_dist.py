import functools
from Levenshtein import distance as levenshtein_distance
import textattack
from .text_to_text_goal_function import TextToTextGoalFunction

class LevenshteinExceedsTargetDistance(TextToTextGoalFunction):
    """Attempts to maximise the Levenshtein distance between the current output
    translation and the reference translation.

    Levenshtein distance is defined as the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change one string into another.
    """

    def __init__(self, *args, target_distance=1000, **kwargs):
        """
        Args:
            model_wrapper: The model wrapper used for generating outputs.
            tokenizer: The tokenizer for decoding outputs.
            target_distance: The target Levenshtein distance.
        """
        self.target_distance = target_distance
        super().__init__(*args, **kwargs)

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        distance = self._get_score(model_output, _)
        return distance < -self.target_distance

    def _get_score(self, model_output, _):
        # Calculate Levenshtein distance between the model output and ground truth
        # model_output : str (model's output on perturbed input)
        # ground_truth_output : str (correct translation from dataset on original input)
        distance = levenshtein_distance(model_output, self.ground_truth_output)

        return -distance

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_distance"]
