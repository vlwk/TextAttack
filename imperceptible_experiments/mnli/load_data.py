import json
from collections import OrderedDict

def load_mnli_data_pairs(path: str, max_rows: int = None):
    """
    Loads MNLI data from a JSONL file and returns a list of (OrderedDict input, label) pairs.

    Args:
        path (str): Path to the JSONL dataset.
        max_rows (int): Max number of samples to return.

    Returns:
        List of (OrderedDict({'premise': ..., 'hypothesis': ...}), label_int)
    """
    label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    pairs = []

    with open(path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if sample['gold_label'] not in label_map:
                continue

            premise = sample['sentence1']
            hypothesis = sample['sentence2']
            label = label_map[sample['gold_label']]

            inp = OrderedDict([('premise', premise), ('hypothesis', hypothesis)])
            pairs.append((inp, label))

            if max_rows and len(pairs) >= max_rows:
                break

    return pairs
