from __future__ import annotations

import sys
from typing import Optional

from ...operators.fantasy_world_operator import FantasyWorldOperator
from ...synthesis.visual_generation.fantasy_world.fantasy_world_synthesis import FantasyWorldSynthesis


class FantasyWorldPipeline:
    def __init__(self, operators: FantasyWorldOperator, synthesis_model: FantasyWorldSynthesis):
        self.operators = operators
        self.synthesis_model = synthesis_model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        wan_ckpt_path: str,
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "FantasyWorldPipeline":
        return cls(
            operators=FantasyWorldOperator(),
            synthesis_model=FantasyWorldSynthesis.from_pretrained(
                pretrained_model_path=model_path,
                wan_ckpt_path=wan_ckpt_path,
                python_bin=python_bin,
            ),
        )

    def __call__(
        self,
        image_path: str,
        camera_json_path: str,
        prompt: str,
        output_dir: str,
        sample_steps: int = 50,
        using_scale: bool = True,
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ):
        prompt = self.operators.process_interaction(prompt)
        perception = self.operators.process_perception(image_path=image_path, camera_json_path=camera_json_path)
        return self.synthesis_model.predict(
            image_path=perception["image_path"],
            camera_json_path=perception["camera_json_path"],
            prompt=prompt,
            output_dir=output_dir,
            sample_steps=sample_steps,
            using_scale=using_scale,
            cuda_visible_devices=cuda_visible_devices,
            timeout=timeout,
            **kwargs,
        )
