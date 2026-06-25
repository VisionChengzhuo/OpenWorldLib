from __future__ import annotations

import sys
from typing import Iterable, Optional

from ...operators.ctrl_world_operator import CtrlWorldOperator
from ...synthesis.vla_generation.ctrl_world_synthesis import CtrlWorldSynthesis


class CtrlWorldPipeline:
    def __init__(self, operators: CtrlWorldOperator, synthesis_model: CtrlWorldSynthesis):
        self.operators = operators
        self.synthesis_model = synthesis_model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        svd_model_path: str,
        clip_model_path: str,
        dataset_root_path: Optional[str] = None,
        dataset_meta_info_path: Optional[str] = None,
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "CtrlWorldPipeline":
        return cls(
            operators=CtrlWorldOperator(),
            synthesis_model=CtrlWorldSynthesis.from_pretrained(
                pretrained_model_path=model_path,
                svd_model_path=svd_model_path,
                clip_model_path=clip_model_path,
                dataset_root_path=dataset_root_path,
                dataset_meta_info_path=dataset_meta_info_path,
                python_bin=python_bin,
            ),
        )

    def __call__(
        self,
        interactions: Iterable[str] | str = "ddcu",
        task_type: str = "keyboard",
        output_dir: str = "./output/ctrl_world",
        cuda_visible_devices: str = "0",
        timeout: Optional[int] = None,
        **kwargs,
    ):
        keyboard = self.operators.process_interaction(interactions)
        return self.synthesis_model.predict(
            keyboard=keyboard,
            task_type=task_type,
            output_dir=output_dir,
            cuda_visible_devices=cuda_visible_devices,
            timeout=timeout,
            **kwargs,
        )
