from typing import Iterable, List

from .base_operator import BaseOperator


class CtrlWorldOperator(BaseOperator):
    def __init__(self, interaction_template=None):
        super().__init__(operation_types=["action_instruction"])
        self.interaction_template = interaction_template or ["l", "r", "f", "b", "u", "d", "o", "c"]
        self.interaction_template_init()

    def check_interaction(self, interaction: str) -> bool:
        for key in interaction:
            if key not in self.interaction_template:
                raise ValueError(f"{key} not in template {self.interaction_template}")
        return True

    def get_interaction(self, interaction: str):
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, interactions: Iterable[str] | str = "ddcu") -> str:
        if isinstance(interactions, str):
            keyboard = interactions
        else:
            keyboard = "".join(interactions)
        self.check_interaction(keyboard)
        return keyboard
