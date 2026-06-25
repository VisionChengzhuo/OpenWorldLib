from __future__ import annotations

import sys
from typing import Optional

from ...operators.gamma_world_operator import GammaWorldOperator
from ...synthesis.visual_generation.gamma_world.gamma_world_synthesis import GammaWorldSynthesis


class GammaWorldPipeline:
    def __init__(self, operators: GammaWorldOperator, synthesis_model: GammaWorldSynthesis):
        self.operators = operators
        self.synthesis_model = synthesis_model

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        vae: Optional[str] = None,
        text_encoder: Optional[str] = None,
        python_bin: str = sys.executable,
        **kwargs,
    ) -> "GammaWorldPipeline":
        return cls(
            operators=GammaWorldOperator(),
            synthesis_model=GammaWorldSynthesis.from_pretrained(
                pretrained_model_path=model_path,
                vae=vae,
                text_encoder=text_encoder,
                python_bin=python_bin,
            ),
        )

    def __call__(
        self,
        output_dir: str,
        mode: str = "causal_few_step",
        n_players: int = 2,
        max_eval_samples: Optional[int] = 1,
        **kwargs,
    ):
        mode, n_players, max_eval_samples = self.operators.process_interaction(
            mode=mode,
            n_players=n_players,
            max_eval_samples=max_eval_samples,
        )
        return self.synthesis_model.predict(
            output_dir=output_dir,
            mode=mode,
            n_players=n_players,
            max_eval_samples=max_eval_samples,
            **kwargs,
        )
