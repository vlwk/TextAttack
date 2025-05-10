from textattack.models.wrappers import ModelWrapper
from typing import List
from google import genai
import os

class GeminiSentimentClassificationWrapper(ModelWrapper):
    def __init__(self, 
                 system_prompt: str, # "Translate the following sentence from English to French. Output only the French sentence."
                 no_classes: int):
        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        client = genai.Client(api_key=self.api_key)
        self.model = client.models
        self.system_prompt = system_prompt
        self.no_classes = no_classes

    def __call__(self, inputs: List[str]) -> List[str]:
        outputs = []
        for text in inputs:
            prompt = f"{self.system_prompt}\n{text}"
            result = self.model.generate_content(model="gemini-2.0-flash", contents=prompt).text.strip()
            arr = [0.0] * self.no_classes
            try:
                idx = int(result)
                if 0 <= idx < len(arr):
                    arr[idx] = 1.0
            except ValueError:
                print(f"Could not parse result: {result}")
            outputs.append(arr)
        return outputs