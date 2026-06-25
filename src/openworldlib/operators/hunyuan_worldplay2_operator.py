from .base_operator import BaseOperator


class HunyuanWorldPlay2Operator(BaseOperator):
    def __init__(self):
        super().__init__(operation_types=["visual_instruction"])

    def process_interaction(self, target_size: int = 952, video_max_frames: int = 32) -> tuple[int, int]:
        if int(target_size) <= 0:
            raise ValueError("HY-World-2.0 target_size must be positive.")
        if int(video_max_frames) < 1:
            raise ValueError("HY-World-2.0 video_max_frames must be >= 1.")
        return int(target_size), int(video_max_frames)
