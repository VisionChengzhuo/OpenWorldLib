"""Sana-WM multi-turn streaming test.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_stream/test_sana_wm_stream.py
    python test_stream/test_sana_wm_stream.py --cpu
"""

import argparse
import os

from PIL import Image

from openworldlib.pipelines.sana_wm.pipeline_sana_wm import SanaWMPipeline


def main():
    parser = argparse.ArgumentParser(description="Sana-WM multi-turn streaming test")
    parser.add_argument("--model_path", type=str, default="Efficient-Large-Model/SANA-WM_bidirectional")
    parser.add_argument("--image", type=str, default="./data/test_case/test_image_case1/ref_image.png")
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_refiner", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"

    pipeline = SanaWMPipeline.from_pretrained(
        model_path=args.model_path,
        device=device,
        enable_refiner=not args.no_refiner,
    )

    if not os.path.isfile(args.image):
        print(f"[WARN] {args.image} not found; using dummy gradient image.")
        import numpy as np
        arr = np.zeros((704, 1280, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, 1280, dtype=np.uint8)[None, :]
        arr[:, :, 1] = np.linspace(0, 255, 704, dtype=np.uint8)[:, None]
        dummy = Image.fromarray(arr)
        dummy.save("/tmp/sana_wm_dummy_input.png")
        args.image = "/tmp/sana_wm_dummy_input.png"

    print("=" * 60)
    print("Turn 1: drive forward")
    video1 = pipeline.stream(
        images=args.image,
        prompt="Driving forward on a sunny road.",
        interactions=["w-40"],
        num_frames=args.num_frames,
        seed=42,
    )
    if video1 is not None:
        from diffusers.utils import export_to_video
        export_to_video(video1, "sana_wm_stream_turn1.mp4", fps=16)
        print("Saved turn 1 → sana_wm_stream_turn1.mp4")

    print("=" * 60)
    print("Turn 2: turn left (memory continues from last frame)")
    video2 = pipeline.stream(
        prompt="Turning left at the intersection.",
        interactions=["a-20"],
        num_frames=args.num_frames,
        seed=43,
    )
    if video2 is not None:
        from diffusers.utils import export_to_video
        export_to_video(video2, "sana_wm_stream_turn2.mp4", fps=16)
        print("Saved turn 2 → sana_wm_stream_turn2.mp4")

    print("Streaming test complete.")


if __name__ == "__main__":
    main()