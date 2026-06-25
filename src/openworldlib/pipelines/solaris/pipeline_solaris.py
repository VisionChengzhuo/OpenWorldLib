from __future__ import annotations

import sys
from typing import Optional

from ...operators.solaris_operator import SolarisOperator
from ...synthesis.visual_generation.solaris.solaris_synthesis import SolarisSynthesis


class SolarisPipeline:
    def __init__(self, operators: SolarisOperator, synthesis_model: SolarisSynthesis):
        self.operators = operators
        self.synthesis_model = synthesis_model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dataset_dir: str,
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "SolarisPipeline":
        return cls(
            operators=SolarisOperator(),
            synthesis_model=SolarisSynthesis.from_pretrained(
                pretrained_model_path=model_path,
                dataset_dir=dataset_dir,
                python_bin=python_bin,
            ),
        )

    def __call__(
        self,
        output_dir: str,
        eval_num_samples: int = 1,
        eval_datasets=None,
        num_frames_eval=None,
        experiment_name: str = "solaris",
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ):
        eval_num_samples = self.operators.process_interaction(eval_num_samples=eval_num_samples)
        return self.synthesis_model.predict(
            output_dir=output_dir,
            eval_num_samples=eval_num_samples,
            eval_datasets=eval_datasets,
            num_frames_eval=num_frames_eval,
            experiment_name=experiment_name,
            cuda_visible_devices=cuda_visible_devices,
            timeout=timeout,
            **kwargs,
        )
