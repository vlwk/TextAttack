from imperceptible_experiments.ner.load import load_ner_data_pairs



print("Loading and slicing data...")
pairs = load_ner_data_pairs()
# pairs = sorted(pairs, key=lambda x: len(x[0]))
pairs = pairs[0:50]
print(pairs[17])
print(pairs[18])
print(pairs[19])