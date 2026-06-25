from .base_operator import BaseOperator


class FantasyWorldOperator(BaseOperator):
    def __init__(self):
        super().__init__(operation_types=["visual_instruction", "textual_instruction"])

    def process_interaction(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("FantasyWorld requires a non-empty prompt.")
        return prompt

    def process_perception(self, image_path: str, camera_json_path: str):
        return {
            "image_path": image_path,
            "camera_json_path": camera_json_path,
        }
