from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
  
ret = []

article_en = "hello i am very smart"
tokenizer.src_lang = "en_XX"
encoded_en = tokenizer(article_en, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_en,
    forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
)
output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print(output)