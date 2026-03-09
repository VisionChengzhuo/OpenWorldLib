from typing import Any, Dict, Optional, List

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from .base_operator import BaseOperator
from ..base_models.diffusion_model.video.wan_2p2.configs import SIZE_CONFIGS, SUPPORTED_SIZES

class Yume1p5Operator(BaseOperator):
    """Lightweight operator for YUME prompt/image preprocessing."""

    def __init__(self, operation_types=[]) -> None:
        super(Yume1p5Operator, self).__init__()
        self.interaction_template = ["forward", "left", "right", "backward", 
                                     "camera_l", "camera_r", "camera_up", "camera_down"]
        self.interaction_template_init()
    
    def check_interaction(self, interaction):
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template")
        return True
    
    def get_interaction(self, interactions):
        if not isinstance(interactions, list):
            interactions = [interactions]
        for interaction in interactions:
            self.check_interaction(interaction)
        self.current_interaction.append(interactions)

    def process_interaction(self, **kwargs) -> Dict[str, Any]:
        INTERACTION_2_CAPTION_DICT = {
                                        # movement
                                        "forward": "The camera pushes forward (W).", 
                                        "backward": "The camera pulls back (S).", 
                                        "left": "Camera turns left (←).",
                                        "right": "Camera turns right (→).",
                                        # rotation
                                        "camera_up": "Camera tilts up (↑).", 
                                        "camera_down": "Camera tilts down (↓).",
                                        "camera_l": "The camera pans to the left (←).",
                                        "camera_r": "The camera pans to the right (→).",
                                    }
        

        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return [INTERACTION_2_CAPTION_DICT[act] for act in now_interaction]

    def process_perception(
        self,
        size: Optional[str] = None,
        images: Optional[Image.Image] = None, # None or one PIL image
        videos: Optional[List[Image.Image]] = None # None or list of PIL images from one video
    ) -> Dict[str, Any]:
        
        assert size in SUPPORTED_SIZES['ti2v-5B'], f"Unsupported size: {size}. Supported sizes for ti2v-5B are: {SUPPORTED_SIZES['ti2v-5B']}"
        size = SIZE_CONFIGS[size]

        if images:
            images = np.array(images)
            if len(images.shape) == 2:
                images = np.stack((images,) * 3, axis=-1)
            elif images.shape[2] == 4:
                images = images[:, :, :3]
            
            images_tensor = torch.from_numpy(images).permute(2, 0, 1).float() / 255.0
            resized_images = F.interpolate(
                images_tensor.unsqueeze(0),
                size=size,
                mode='bilinear',
                align_corners=False
            )[0]
        
        if videos:
            video_transform = transforms.ToTensor()
            video_pixel_values = torch.stack([video_transform(frame) for frame in videos], dim=0)
            video_pixel_values = (torch.nn.functional.interpolate(video_pixel_values.sub_(0.5).div_(0.5), size=size, mode='bicubic')).clamp_(-1, 1)
        
        return {
            "ref_images": resized_images if images is not None else None, 
            "ref_videos": video_pixel_values if videos is not None else None
        }
