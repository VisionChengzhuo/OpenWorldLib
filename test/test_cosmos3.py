import os

from openworldlib.pipelines.cosmos.pipeline_cosmos3 import Cosmos3Pipeline


def main():
    pipe = Cosmos3Pipeline.from_pretrained(
        model_path=os.environ.get("COSMOS3_MODEL_PATH", "nvidia/Cosmos3-Nano"),
    )
    result = pipe(
        prompt=os.environ.get("COSMOS3_PROMPT", "A mobile robot navigates a warehouse aisle and stops at a shelf."),
        output_path=os.environ.get("COSMOS3_OUTPUT", "./output/cosmos3/cosmos3_t2v.mp4"),
        num_frames=int(os.environ.get("COSMOS3_NUM_FRAMES", "17")),
        height=int(os.environ.get("COSMOS3_HEIGHT", "352")),
        width=int(os.environ.get("COSMOS3_WIDTH", "640")),
        fps=int(os.environ.get("COSMOS3_FPS", "24")),
        num_inference_steps=int(os.environ.get("COSMOS3_STEPS", "1")),
    )
    print(result["video_path"])


if __name__ == "__main__":
    main()
