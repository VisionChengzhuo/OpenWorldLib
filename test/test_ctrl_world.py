import os

from openworldlib.pipelines.ctrl_world.pipeline_ctrl_world import CtrlWorldPipeline


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must point to a local checkpoint/model path for this integration test.")
    return value


def main():
    pipe = CtrlWorldPipeline.from_pretrained(
        model_path=require_env("CTRL_WORLD_CKPT"),
        svd_model_path=require_env("CTRL_WORLD_SVD_MODEL"),
        clip_model_path=require_env("CTRL_WORLD_CLIP_MODEL"),
        dataset_root_path=require_env("CTRL_WORLD_DATASET_ROOT"),
        dataset_meta_info_path=require_env("CTRL_WORLD_DATASET_META"),
        python_bin=os.environ.get("CTRL_WORLD_PYTHON", "python"),
    )
    result = pipe(
        interactions=os.environ.get("CTRL_WORLD_KEYBOARD", "ddcu"),
        output_dir=os.environ.get("CTRL_WORLD_OUTPUT", "./output/ctrl_world"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        timeout=None,
    )
    print(result["video_path"])


if __name__ == "__main__":
    main()
