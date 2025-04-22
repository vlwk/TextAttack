cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)

def load_en_fr_data():

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

    return pairs