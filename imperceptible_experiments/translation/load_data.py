import os
from bs4 import BeautifulSoup

from textattack.datasets import Dataset



def load_en_fr_data():

    original_dir = os.getcwd()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cur_dir)

    # -----------------------
    # Load source/target data
    # -----------------------
    source_path = "data/newstest2014-fren-src.en.sgm"
    target_path = "data/newstest2014-fren-ref.fr.sgm"

    source_doc = BeautifulSoup(open(source_path, 'r'), 'html.parser')
    target_doc = BeautifulSoup(open(target_path, 'r'), 'html.parser')

    pairs = []

    for doc in source_doc.find_all('doc'):
        docid = str(doc['docid'])
        for seg in doc.find_all('seg'):
            segid = str(seg['id'])
            src = str(seg.string)
            tgt_node = target_doc.select_one(f'doc[docid="{docid}"] > seg[id="{segid}"]')
            if tgt_node:
                tgt = str(tgt_node.string)
                # Only use the raw input for perturbation
                pairs.append((src, tgt))

    os.chdir(original_dir)

    return pairs


def load_en_fr_dataset():
    """
    Loads English-French sentence pairs from SGM files and returns a TextAttack Dataset.

    Returns:
        textattack.datasets.Dataset: wrapped dataset of (English, French) pairs.
    """
    original_dir = os.getcwd()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cur_dir)

    source_path = "data/newstest2014-fren-src.en.sgm"
    target_path = "data/newstest2014-fren-ref.fr.sgm"

    with open(source_path, "r", encoding="utf-8") as f:
        source_doc = BeautifulSoup(f, "html.parser")

    with open(target_path, "r", encoding="utf-8") as f:
        target_doc = BeautifulSoup(f, "html.parser")

    pairs = []

    for doc in source_doc.find_all("doc"):
        docid = str(doc["docid"])
        for seg in doc.find_all("seg"):
            segid = str(seg["id"])
            src = str(seg.string).strip() if seg.string else ""
            tgt_node = target_doc.select_one(f'doc[docid="{docid}"] > seg[id="{segid}"]')
            if tgt_node and tgt_node.string:
                tgt = str(tgt_node.string).strip()
                pairs.append((src, tgt))

    os.chdir(original_dir)
    return Dataset(pairs) 
