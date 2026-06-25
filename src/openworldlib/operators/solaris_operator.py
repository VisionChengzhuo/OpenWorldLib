from .base_operator import BaseOperator


class SolarisOperator(BaseOperator):
    def __init__(self):
        super().__init__(operation_types=["action_instruction", "visual_instruction"])

    def process_interaction(self, eval_num_samples: int = 1) -> int:
        if int(eval_num_samples) < 1:
            raise ValueError("Solaris eval_num_samples must be >= 1.")
        return int(eval_num_samples)
