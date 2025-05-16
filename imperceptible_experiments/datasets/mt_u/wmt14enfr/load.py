import os
import requests
import tarfile
from io import BytesIO
from bs4 import BeautifulSoup
import textattack

def download_en_fr_dataset():
    

    # Define constants
    url = "http://statmt.org/wmt14/test-full.tgz"
    target_dir = os.path.join("temp/translation", "data")
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading WMT14 test data from {url}...")

    # Download and extract in-memory
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with tarfile.open(fileobj=BytesIO(response.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("newstest2014-fren-src.en.sgm") or member.name.endswith("newstest2014-fren-ref.fr.sgm"):
                print(f"Extracting {member.name} to {target_dir}")
                member.name = os.path.basename(member.name) 
                tar.extract(member, path=target_dir)

    print("en_fr dataset downloaded.")

def load_en_fr_dataset():
    """
    Loads English-French sentence pairs from SGM files and returns a TextAttack Dataset.

    Returns:
        textattack.datasets.Dataset: wrapped dataset of (English, French) pairs.
    """
    

    source_path = os.path.join("temp/translation", "data/newstest2014-fren-src.en.sgm")
    target_path = os.path.join("temp/translation", "data/newstest2014-fren-ref.fr.sgm")

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
    return textattack.datasets.Dataset(pairs) 