
from textattack.models.wrappers import HuggingFaceModelWrapper
from typing import List

class MarianWrapper(HuggingFaceModelWrapper):
    """
    A wrapper for the model
        facebook/mbart-large-50-many-to-many-mmt
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        source: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
    """

    def __call__(self, text_input_list: List[str]) -> List[str]:
        """
        Args:
            input_texts: List[str]
        
        Return:
            ret: List[str]
                Result of translation. One per element in input_texts.
        """


        device = next(self.model.parameters()).device
        ret = []

        for article_en in text_input_list:
            # translate English to French
            self.tokenizer.src_lang = "en_XX"
            encoded_en = self.tokenizer(article_en, return_tensors="pt")
            
            encoded_en = {k: v.to(device) for k, v in encoded_en.items()}
            
            generated_tokens = self.model.generate(
                **encoded_en,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["fr_XX"]
            )
            output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True) # Output translation (str) wrapped in a List
            ret.append(output[0])
        return ret
