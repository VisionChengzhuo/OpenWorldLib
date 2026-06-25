from __future__ import annotations

import sys

from ...operators.hunyuan_worldplay2_operator import HunyuanWorldPlay2Operator
from ...synthesis.visual_generation.hunyuan_world.hunyuan_worldplay2_synthesis import HunyuanWorldPlay2Synthesis


class HunyuanWorldPlay2Pipeline:
    def __init__(self, operators: HunyuanWorldPlay2Operator, synthesis_model: HunyuanWorldPlay2Synthesis):
        self.operators = operators
        self.synthesis_model = synthesis_model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "tencent/HY-World-2.0",
        subfolder: str = "HY-WorldMirror-2.0",
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "HunyuanWorldPlay2Pipeline":
        return cls(
            operators=HunyuanWorldPlay2Operator(),
            synthesis_model=HunyuanWorldPlay2Synthesis.from_pretrained(
                pretrained_model_path=model_path,
                subfolder=subfolder,
                python_bin=python_bin,
            ),
        )

    def __call__(self, input_path: str, output_dir: str, target_size: int = 952, video_max_frames: int = 32, **kwargs):
        target_size, video_max_frames = self.operators.process_interaction(
            target_size=target_size,
            video_max_frames=video_max_frames,
        )
        return self.synthesis_model.predict(
            input_path=input_path,
            output_dir=output_dir,
            target_size=target_size,
            video_max_frames=video_max_frames,
            **kwargs,
        )
