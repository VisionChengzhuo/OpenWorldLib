import os

from openworldlib.pipelines.solaris.pipeline_solaris import SolarisPipeline


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must point to a local model/data path for this integration test.")
    return value


def main():
    pipe = SolarisPipeline.from_pretrained(
        model_path=require_env("SOLARIS_MODEL_DIR"),
        dataset_dir=require_env("SOLARIS_DATASET_DIR"),
        python_bin=os.environ.get("SOLARIS_PYTHON", "python"),
    )
    result = pipe(
        output_dir=os.environ.get("SOLARIS_OUTPUT", "./output/solaris"),
        eval_num_samples=int(os.environ.get("SOLARIS_EVAL_NUM_SAMPLES", "1")),
        eval_datasets=[
            item.strip()
            for item in os.environ.get("SOLARIS_EVAL_DATASETS", "").split(",")
            if item.strip()
        ] or None,
        num_frames_eval=(
            int(os.environ["SOLARIS_NUM_FRAMES_EVAL"])
            if os.environ.get("SOLARIS_NUM_FRAMES_EVAL")
            else None
        ),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        eval_metrics=os.environ.get("SOLARIS_EVAL_METRICS", ""),
        timeout=None,
    )
    print("\n".join(result["video_paths"]))


if __name__ == "__main__":
    main()
