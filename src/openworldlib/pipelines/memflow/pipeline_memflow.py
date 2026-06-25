from __future__ import annotations

import sys
from typing import Optional

from ...operators.memflow_operator import MemFlowOperator
from ...synthesis.visual_generation.memflow.memflow_synthesis import MemFlowSynthesis


class MemFlowPipeline:
    def __init__(self, operators: MemFlowOperator, synthesis_model: MemFlowSynthesis):
        self.operators = operators
        self.synthesis_model = synthesis_model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        wan_model_path: str,
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "MemFlowPipeline":
        return cls(
            operators=MemFlowOperator(),
            synthesis_model=MemFlowSynthesis.from_pretrained(
                pretrained_model_path=model_path,
                wan_model_path=wan_model_path,
                python_bin=python_bin,
            ),
        )

    def __call__(
        self,
        prompt: str,
        output_dir: str,
        num_output_frames: int = 120,
        num_samples: int = 1,
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ):
        prompt = self.operators.process_interaction(prompt)
        return self.synthesis_model.predict(
            prompt=prompt,
            output_dir=output_dir,
            num_output_frames=num_output_frames,
            num_samples=num_samples,
            cuda_visible_devices=cuda_visible_devices,
            timeout=timeout,
            **kwargs,
        )
