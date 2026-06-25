import os
from pathlib import Path

from openworldlib.pipelines.hunyuan_world.pipeline_hunyuan_worldplay2 import HunyuanWorldPlay2Pipeline


def main():
    repo_root = Path(__file__).resolve().parents[1]
    pipe = HunyuanWorldPlay2Pipeline.from_pretrained(
        model_path=os.environ.get("HY_WORLD2_MODEL_PATH", "tencent/HY-World-2.0"),
        subfolder=os.environ.get("HY_WORLD2_SUBFOLDER", "HY-WorldMirror-2.0"),
        python_bin=os.environ.get("HY_WORLD2_PYTHON", "python"),
    )
    result = pipe(
        input_path=os.environ.get("HY_WORLD2_INPUT", str(repo_root / "data" / "test_case" / "hunyuan_worldplay2" / "teaser.png")),
        output_dir=os.environ.get("HY_WORLD2_OUTPUT", "./output/hunyuan_worldplay2"),
        target_size=int(os.environ.get("HY_WORLD2_TARGET_SIZE", "952")),
        video_max_frames=int(os.environ.get("HY_WORLD2_VIDEO_MAX_FRAMES", "32")),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        timeout=None,
    )
    print(result["gaussians_path"])
    print(result["points_path"])


if __name__ == "__main__":
    main()
