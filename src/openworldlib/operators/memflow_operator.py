from .base_operator import BaseOperator


class MemFlowOperator(BaseOperator):
    def __init__(self):
        super().__init__(operation_types=["textual_instruction"])

    def process_interaction(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("MemFlow requires a non-empty prompt.")
        return prompt
