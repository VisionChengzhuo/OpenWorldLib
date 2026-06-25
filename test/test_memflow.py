import os

from openworldlib.pipelines.memflow.pipeline_memflow import MemFlowPipeline


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must point to a local checkpoint/model path for this integration test.")
    return value


def main():
    pipe = MemFlowPipeline.from_pretrained(
        model_path=require_env("MEMFLOW_CKPT_DIR"),
        wan_model_path=require_env("MEMFLOW_WAN_MODEL"),
        python_bin=os.environ.get("MEMFLOW_PYTHON", "python"),
    )
    result = pipe(
        prompt=os.environ.get("MEMFLOW_PROMPT", "A cinematic shot of a red sports car driving along a coastal highway at sunset."),
        output_dir=os.environ.get("MEMFLOW_OUTPUT", "./output/memflow"),
        num_output_frames=int(os.environ.get("MEMFLOW_NUM_OUTPUT_FRAMES", "12")),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        timeout=None,
    )
    print(result["video_path"])


if __name__ == "__main__":
    main()
