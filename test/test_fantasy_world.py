import os
from pathlib import Path

from openworldlib.pipelines.fantasy_world.pipeline_fantasy_world import FantasyWorldPipeline


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must point to a local checkpoint/model path for this integration test.")
    return value


def main():
    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "data" / "test_case" / "test_image_seq_case1" / "image_0001.jpg"
    camera_json_path = repo_root / "data" / "test_case" / "fantasy_world" / "camera_data.json"
    pipe = FantasyWorldPipeline.from_pretrained(
        model_path=require_env("FANTASY_WORLD_CKPT"),
        wan_ckpt_path=require_env("FANTASY_WORLD_WAN_CKPT"),
        python_bin=os.environ.get("FANTASY_WORLD_PYTHON", "python"),
    )
    result = pipe(
        image_path=os.environ.get("FANTASY_WORLD_IMAGE", str(image_path)),
        camera_json_path=os.environ.get("FANTASY_WORLD_CAMERA", str(camera_json_path)),
        prompt=os.environ.get(
            "FANTASY_WORLD_PROMPT",
            "In the Open Loft Living Room, sunlight streams through large windows, highlighting the sleek fireplace and elegant wooden stairs.",
        ),
        output_dir=os.environ.get("FANTASY_WORLD_OUTPUT", "./output/fantasy_world"),
        sample_steps=int(os.environ.get("FANTASY_WORLD_SAMPLE_STEPS", "50")),
        frames=int(os.environ.get("FANTASY_WORLD_FRAMES", "17")),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        timeout=None,
    )
    print(result["video_path"])
    print(result["point_cloud_path"])


if __name__ == "__main__":
    main()
