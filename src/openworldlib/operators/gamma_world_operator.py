from .base_operator import BaseOperator


class GammaWorldOperator(BaseOperator):
    def __init__(self):
        super().__init__(operation_types=["action_instruction", "visual_instruction", "textual_instruction"])

    def process_interaction(self, mode: str = "causal_few_step", n_players: int = 2, max_eval_samples: int | None = 1):
        valid_modes = {"bidirectional", "causal", "causal_few_step"}
        if mode not in valid_modes:
            raise ValueError(f"Gamma-World mode must be one of {sorted(valid_modes)}, got {mode!r}.")
        if int(n_players) < 1:
            raise ValueError("Gamma-World n_players must be >= 1.")
        if max_eval_samples is not None and int(max_eval_samples) < 1:
            raise ValueError("Gamma-World max_eval_samples must be >= 1 when provided.")
        return mode, int(n_players), None if max_eval_samples is None else int(max_eval_samples)
