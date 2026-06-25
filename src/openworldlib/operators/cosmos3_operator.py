from .base_operator import BaseOperator


class Cosmos3Operator(BaseOperator):
    def __init__(self):
        super().__init__(operation_types=["textual_instruction", "visual_instruction"])

    def process_interaction(self, prompt: str) -> str:
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("Cosmos3 prompt must be a non-empty string.")
        return prompt
