from datasets import load_dataset, load_from_disk

def load_emotion_data_pairs(target: int = 0, max_rows: int = None):
    """
    Loads an emotion classification dataset and returns (text, target) pairs.

    Args:
        target (int): Index of target class in the list
            emotion_classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        max_rows (int, optional): Max number of examples to return.

    Returns:
        List of (input_text, target) tuples
    """
    dataset = load_dataset("emotion", split='test')
    pairs = []

    for ex in dataset:
        text = ex['text']
        pairs.append((text, target))
        if max_rows and len(pairs) >= max_rows:
            break

    return pairs
