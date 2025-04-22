from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class GPT2PromptWrapper:
    def __init__(self, model_name="gpt2", device=None, max_new_tokens=30):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, inputs: list[str]) -> list[str]:
        outputs = []
        for prompt in inputs:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            # Extract only the generated text (exclude prompt tokens)
            generated = output_ids[0][input_ids.shape[1]:]
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            outputs.append(decoded)
        return outputs
