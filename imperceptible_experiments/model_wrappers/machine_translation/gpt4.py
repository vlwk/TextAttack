import os
from openai import OpenAI
from textattack.models.wrappers import ModelWrapper
from typing import List

class GPT4MachineTranslationWrapper(ModelWrapper):
    def __init__(self, system_prompt: str, temperature: float = 0.3):
        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = system_prompt
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
            outputs.append(str(result))
        return outputs