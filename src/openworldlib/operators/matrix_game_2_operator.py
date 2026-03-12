from .base_operator import BaseOperator

import torch
from torchvision.transforms import v2
import random


def encode_actions(action_list, mode):
    """
    将一个动作 list 编码成 keyboard_condition / mouse_condition (一次性的)
    """
    if mode == "universal":
        KEYBOARD_IDX = {"forward":0, "back":1, "left":2, "right":3}
        CAM_MAP = {"camera_l":[0,-0.1], "camera_r":[0,0.1]}
        keyboard_dim = 4
        mouse = True
        COMBINATION_MAP = {
            "forward_left": ["forward", "left"],
            "forward_right": ["forward", "right"],
            "back_left": ["back", "left"], 
            "back_right": ["back", "right"]
        }
    elif mode == "gta_drive":
        KEYBOARD_IDX = {"forward":0, "back":1}
        CAM_MAP = {"camera_l":[0,-0.1], "camera_r":[0,0.1]}
        keyboard_dim = 2
        mouse = True
    else: # templerun
        KEYBOARD_IDX = {
            "nomove":0,"jump":1,"slide":2,
            "turnleft":3,"turnright":4,
            "leftside":5,"rightside":6,
        }
        CAM_MAP = {}
        keyboard_dim = 7
        mouse = False

    keyboard = torch.zeros(keyboard_dim)
    if mouse:
        mouse_value = torch.zeros(2)

    for act in action_list:
        if act in COMBINATION_MAP:
            for sub_act in COMBINATION_MAP[act]:
                if sub_act in KEYBOARD_IDX:
                    keyboard[KEYBOARD_IDX[sub_act]] = 1
        if act in KEYBOARD_IDX:
            keyboard[KEYBOARD_IDX[act]] = 1
        if mouse and act in CAM_MAP:
            mouse_value[:] = torch.tensor(CAM_MAP[act])

    if mouse:
        return keyboard, mouse_value
    return keyboard, None


def resizecrop(image, th, tw):
    w, h = image.size
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    image = image.crop((left, top, right, bottom))
    return image


class MatrixGame2Operator(BaseOperator):

    def __init__(self, operation_types=[], mode="universal", interaction_template=[]):
        super().__init__(operation_types=operation_types)
        self.mode = mode
        if mode == 'universal':
            interaction_template = ["forward", "left", "right", "forward_left", "forward_right",
                                    "camera_l", "camera_r"]
        elif mode == 'gta_drive':
            interaction_template = ["forward", "back", "camera_l", "camera_r"]
        elif mode == 'templerun':
            interaction_template = ["jump","slide","leftside","rightside",
                                    "turnleft","turnright","nomove"]
        self.interaction_template = interaction_template
        self.interaction_template_init()

        self.current_interaction = []  # 保存用户按顺序输入的动作组

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def check_interaction(self, interaction):
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template")
        return True

    def get_interaction(self, interaction_list):
        # 用户传进来是 list
        for act in interaction_list:
            self.check_interaction(act)
        self.current_interaction.append(interaction_list)

    def _build_sequence(self, num_frames, frames_per_action=4):
        if len(self.current_interaction) == 0:
            raise RuntimeError("No interaction registered")

        cur_interaction = self.current_interaction[-1]

        total_actions = len(cur_interaction)
        available_frames = num_frames
        frames_per_action = max(frames_per_action, available_frames // total_actions)
        
        if frames_per_action < 1:
            frames_per_action = 1

        padded_actions = []
        for action in cur_interaction:
            padded_actions.extend([action] * frames_per_action)

        while len(padded_actions) < num_frames:
            padded_actions.append(padded_actions[-1])

        padded_actions = padded_actions[:num_frames]

        keyboard_list = []
        mouse_list = []
        mouse_enabled = (self.mode != "templerun")
        
        for action in padded_actions:
            kb, ms = encode_actions([action], self.mode)
            keyboard_list.append(kb)
            if mouse_enabled:
                mouse_list.append(ms)
        
        keyboard_tensor = torch.stack(keyboard_list)
        if mouse_enabled:
            mouse_tensor = torch.stack(mouse_list)
            return {
                "keyboard_condition": keyboard_tensor,
                "mouse_condition": mouse_tensor
            }
        
        return {"keyboard_condition": keyboard_tensor}

    # multi_turn 不用额外修改，外部调整num_frames以及输入interaction即可
    def process_action_universal(self, num_frames):
        return self._build_sequence(num_frames)

    def process_action_gta_drive(self, num_frames):
        return self._build_sequence(num_frames)

    def process_action_templerun(self, num_frames):
        return self._build_sequence(num_frames)
    
    def process_interaction(self, num_frames):
        if self.mode == "universal":
            return self.process_action_universal(num_frames)
        elif self.mode == "gta_drive":
            return self.process_action_gta_drive(num_frames)
        elif self.mode == "templerun":
            return self.process_action_templerun(num_frames)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def process_perception(self,
                           input_image,
                           num_output_frames,
                           resize_H=352,
                           resize_W=640,
                           device: str = "cuda",
                           weight_dtype = torch.bfloat16,):
        image = resizecrop(input_image, resize_H, resize_W)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=weight_dtype, device=device)

        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [resize_H//8, resize_W//8], "tile_stride": [resize_H//16+1, resize_W//16-2]}

        return {
            "image": image,
            "img_cond": img_cond,
            "tiler_kwargs": tiler_kwargs
        }
