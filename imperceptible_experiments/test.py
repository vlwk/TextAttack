from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

clean = "yellow"
perturbed = "y\u200dellow"

print(tokenizer.tokenize(clean))      # ['yellow']
print(tokenizer.tokenize(perturbed))  # ['y', '##ellow'] or similar
