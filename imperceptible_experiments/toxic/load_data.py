def load_toxic_data(path):
    import json
    """
    Loads a JSON dataset of toxic comments and returns (input, label) pairs.

    Parameters:
        path (str): Path to the JSON file.

    Returns:
        List of (comment, label) pairs. Label is always 0 (non-toxic ground truth).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = [(row["comment"], 0) for row in data]

    return pairs
