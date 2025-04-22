from Levenshtein import distance as levenshtein_distance

def get_lev_score(model_output, ground_truth_output):
        """
        model_output (str): Stores the output of the text-to-text model.
        ground_truth_output: The expected output.

        When used with ImperceptibleDE, this method maximizes the Levenshtein distance between model_output and ground_truth_output.
        """
        distance = levenshtein_distance(model_output, ground_truth_output)

        return -distance