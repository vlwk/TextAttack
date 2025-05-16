from textattack.models.wrappers import ModelWrapper
from typing import List
from google import genai
import os

class Gemini2FlashWrapperMT(ModelWrapper):
    def __init__(self, 
                 system_prompt: str):
        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        client = genai.Client(api_key=self.api_key)
        self.model = client.models
        self.system_prompt = system_prompt

    def __call__(self, inputs: List[str]) -> List[str]:
        outputs = []
        for text in inputs:
            prompt = f"{self.system_prompt}\n{text}"
            try:
                response = self.model.generate_content(model="gemini-2.0-flash", contents=prompt)
                outputs.append(response.text.strip())
            except Exception as e:
                outputs.append(f"[ERROR] {str(e)}")
        return outputs

