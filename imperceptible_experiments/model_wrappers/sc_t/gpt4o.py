import os
from openai import OpenAI
from textattack.models.wrappers import ModelWrapper
from typing import List

class GPT4oWrapperSC(ModelWrapper):

    # This is a model wrapper for a sentiment classification task.
    # Example:
    # system_prompt = "This sentence is one of the following emotions: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5). Output the number of the emotion. Sentence:"
    # no_classes = 6

    def __init__(self, system_prompt: str, no_classes: int, temperature: float = 0.3):
        from dotenv import load_dotenv
        load_dotenv()   
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.system_prompt = system_prompt 
        self.no_classes = no_classes
        self.client = OpenAI(api_key=self.api_key)
        self.temperature = temperature
        self.model = self  # acts as a dummy reference to satisfy GoalFunction

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        outputs = []
        for text in inputs:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=self.temperature,
                max_tokens=50
            )
            result = response.choices[0].message.content.strip()
            arr = [0.0] * self.no_classes 
            try:
                idx = int(result)
                if 0 <= idx < len(arr):
                    arr[idx] = 1.0
            except ValueError:
                print(f"Could not parse result: {result}")
            outputs.append(arr)
        return outputs


