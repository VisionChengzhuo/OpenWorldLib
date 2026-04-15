from diffusers.utils import export_to_video
from PIL import Image

from openworldlib.pipelines.infinite_world.pipeline_infinite_world import InfiniteWorldPipeline


image_path = "./data/test_case/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

pretrained_model_path = "/ytech_m2v8_hdd/zengbohan/hf_checkpoints/Infinite-World"

pipeline = InfiniteWorldPipeline.from_pretrained(
    model_path=pretrained_model_path,
    device="cuda",
)

output_video = pipeline(
    images=input_image,
    prompt="A serene campus walkway lined with modern glass buildings and soft daylight.",
    interactions=["forward", "forward+camera_r", "forward", "camera_l"],
    num_frames=80,
    size=(384, 1024),
)

export_to_video(output_video, "infinite_world_demo.mp4", fps=30)
