from __future__ import annotations

from ...operators.cosmos3_operator import Cosmos3Operator
from ...synthesis.visual_generation.cosmos.cosmos3_synthesis import Cosmos3Synthesis


class Cosmos3Pipeline:
    def __init__(self, operators: Cosmos3Operator, synthesis_model: Cosmos3Synthesis):
        self.operators = operators
        self.synthesis_model = synthesis_model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "nvidia/Cosmos3-Nano",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ) -> "Cosmos3Pipeline":
        return cls(
            operators=Cosmos3Operator(),
            synthesis_model=Cosmos3Synthesis.from_pretrained(
                pretrained_model_path=model_path,
                torch_dtype=torch_dtype,
            ),
        )

    def __call__(self, prompt: str, output_path: str, **kwargs):
        prompt = self.operators.process_interaction(prompt)
        return self.synthesis_model.predict(prompt=prompt, output_path=output_path, **kwargs)
