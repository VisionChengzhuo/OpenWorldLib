import os
from typing import Dict, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from huggingface_hub import snapshot_download

from ....base_models.three_dimensions.point_clouds.vggt.vggt.models.vggt import VGGT
from ....base_models.three_dimensions.point_clouds.vggt.vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_square
from ....base_models.three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from ....base_models.three_dimensions.point_clouds.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
from ....base_models.three_dimensions.point_clouds.gaussian_splatting.scene.dataset_readers import (
    storePly,
    fetchPly,
)
from ....representations.point_clouds_generation.flash_world.flash_world.render import (
    gaussian_render,
)


class VGGTRepresentation:
    """VGGT Representation model for 3D scene reconstruction."""
    
    def __init__(self, model: Optional[VGGT] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
            
            if self.device == "cuda" and torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability()[0]
                self.dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = torch.float32
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> 'VGGTRepresentation':
        try:
            model = VGGT.from_pretrained(pretrained_model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load VGGT model: {e}")
        
        instance = cls(model=model, device=device)
        instance.preprocess_mode = kwargs.get('preprocess_mode', 'crop')
        instance.resolution = kwargs.get('resolution', 518)
        return instance
    
    def api_init(self, api_key: str, endpoint: str):
        """Initialize API access (placeholder for future API-based models)."""
        pass
    
    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get representation from input data using VGGT model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() first.")
        
        images_input = data['images']
        predict_cameras = data.get('predict_cameras', True)
        predict_depth = data.get('predict_depth', True)
        predict_points = data.get('predict_points', True)
        predict_tracks = data.get('predict_tracks', False)
        query_points = data.get('query_points', None)
        preprocess_mode = data.get('preprocess_mode', self.preprocess_mode)
        resolution = data.get('resolution', self.resolution)
        if isinstance(images_input, list):
            image_list = images_input
        elif isinstance(images_input, np.ndarray):
            if images_input.ndim == 3:
                image_list = [images_input]
            elif images_input.ndim == 4:
                image_list = [images_input[i] for i in range(images_input.shape[0])]
            else:
                image_list = [images_input]
        else:
            if isinstance(images_input, str):
                image_list = [images_input]
            else:
                image_list = images_input if isinstance(images_input, list) else [images_input]
        
        has_paths = any(isinstance(img, str) for img in image_list)
        
        if has_paths:
            if preprocess_mode == "square":
                images, _ = load_and_preprocess_images_square(image_list, target_size=resolution)
            else:
                images = load_and_preprocess_images(image_list, mode=preprocess_mode)
        else:
            image_tensors = []
            for img_array in image_list:
                if isinstance(img_array, np.ndarray):
                    if img_array.max() > 1.0:
                        img_array = img_array / 255.0
                    
                    if img_array.ndim == 3 and img_array.shape[2] == 3:
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
                    elif img_array.ndim == 2:
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
                        img_tensor = img_tensor.repeat(3, 1, 1)
                    else:
                        raise ValueError(f"Unsupported image array shape: {img_array.shape}")
                    image_tensors.append(img_tensor)
                else:
                    raise ValueError(f"Unsupported image type: {type(img_array)}")
            
            images = torch.stack(image_tensors)
            images = F.interpolate(
                images,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False
            )
        
        images = images.to(self.device)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        query_points_tensor = None
        if predict_tracks and query_points is not None:
            if isinstance(query_points, np.ndarray):
                query_points_tensor = torch.FloatTensor(query_points).to(self.device)
            elif isinstance(query_points, torch.Tensor):
                query_points_tensor = query_points.to(self.device)
        
        results = {}
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=(self.device == "cuda")):
                images_batch = images[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)
                
                if predict_cameras:
                    pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        pose_enc, images_batch.shape[-2:]
                    )
                    results['extrinsic'] = extrinsic.squeeze(0).cpu().numpy()
                    results['intrinsic'] = intrinsic.squeeze(0).cpu().numpy()
                
                if predict_depth:
                    depth_map, depth_conf = self.model.depth_head(
                        aggregated_tokens_list, images_batch, ps_idx
                    )
                    results['depth_map'] = depth_map.squeeze(0).cpu().numpy()
                    results['depth_conf'] = depth_conf.squeeze(0).cpu().numpy()
                
                if predict_points:
                    point_map, point_conf = self.model.point_head(
                        aggregated_tokens_list, images_batch, ps_idx
                    )
                    results['point_map'] = point_map.squeeze(0).cpu().numpy()
                    results['point_conf'] = point_conf.squeeze(0).cpu().numpy()
                
                if predict_tracks and query_points_tensor is not None:
                    if query_points_tensor.dim() == 2:
                        query_points_tensor = query_points_tensor.unsqueeze(0)
                    track, vis_score, conf_score = self.model.track_head(
                        aggregated_tokens_list, images_batch, ps_idx,
                        query_points=query_points_tensor
                    )
                    results['tracks'] = track.squeeze(0).cpu().numpy()
                    results['track_vis_score'] = vis_score.squeeze(0).cpu().numpy()
                    results['track_conf_score'] = conf_score.squeeze(0).cpu().numpy()
                
                if predict_depth and predict_cameras and predict_points:
                    point_map_from_depth = unproject_depth_map_to_point_map(
                        results['depth_map'],
                        results['extrinsic'],
                        results['intrinsic']
                    )
                    results['point_map_from_depth'] = point_map_from_depth
        
        return results

    @staticmethod
    def _estimate_gaussian_scale(points: np.ndarray, scene_center: np.ndarray) -> float:
        """
        Estimate a robust Gaussian scale for 3DGS points based on nearest-neighbor statistics.
        """
        if len(points) < 4:
            scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
            return max(scene_radius / 2000.0, 1e-4)

        sample_n = min(len(points), 2048)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=sample_n, replace=False)
        sample = torch.from_numpy(points[idx]).float()
        dist = torch.cdist(sample, sample, p=2)
        dist.fill_diagonal_(1e9)
        nn = dist.min(dim=1).values
        nn_med = float(nn.median().item())

        scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
        min_scale = max(scene_radius / 5000.0, 1e-4)
        max_scale = max(scene_radius / 300.0, min_scale)
        return float(np.clip(nn_med * 0.6, min_scale, max_scale))

    def render_with_3dgs(
        self,
        ply_path: str,
        camera_config: Dict[str, Any],
        image_width: int = 704,
        image_height: int = 480,
        device: Optional[str] = None,
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
    ) -> Image.Image:
        """
        Render a single frame from a VGGT reconstruction using 3D Gaussian Splatting.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pcd = fetchPly(ply_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)
        if points.size == 0:
            raise RuntimeError(f"No points loaded from PLY: {ply_path}")

        scene_center = points.mean(axis=0)
        scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
        scene_radius = max(scene_radius, 1e-6)

        # Normalize scene for rendering stability across large/small VGGT scales.
        points_norm = (points - scene_center[None, :]) / scene_radius
        center = np.asarray(camera_config.get("center", scene_center.tolist()), dtype=np.float32)
        center_norm = (center - scene_center) / scene_radius

        radius_raw = float(camera_config.get("radius", 1.0 * scene_radius))
        radius_norm = max(radius_raw / scene_radius, 1e-3)
        # +180° yaw so camera is on the opposite side (scene faces camera, not back).
        yaw_deg = float(camera_config.get("yaw", 0.0)) + 180.0
        pitch_deg = float(camera_config.get("pitch", 0.0))

        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        cam_x = center_norm[0] + radius_norm * np.cos(pitch) * np.sin(yaw)
        cam_y = center_norm[1] + radius_norm * np.sin(pitch)
        cam_z = center_norm[2] + radius_norm * np.cos(pitch) * np.cos(yaw)
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        def build_c2w(
            look_at: np.ndarray,
            eye: np.ndarray,
            reverse_forward: bool = False,
            basis_layout: str = "row",
        ) -> np.ndarray:
            forward = (eye - look_at) if reverse_forward else (look_at - eye)
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                right = np.cross(forward, up)
                right_norm = np.linalg.norm(right)
            right = right / (right_norm + 1e-8)
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 1e-8)

            c2w_local = np.eye(4, dtype=np.float32)
            if basis_layout == "row":
                c2w_local[0, :3] = right
                c2w_local[1, :3] = up
                c2w_local[2, :3] = forward
            else:
                c2w_local[:3, 0] = right
                c2w_local[:3, 1] = up
                c2w_local[:3, 2] = forward
            c2w_local[:3, 3] = eye
            return c2w_local

        fx = 0.5 * image_width / np.tan(np.deg2rad(60.0) / 2.0)
        fy = 0.5 * image_height / np.tan(np.deg2rad(45.0) / 2.0)
        cx = image_width / 2.0
        cy = image_height / 2.0

        xyz = torch.from_numpy(points_norm).to(device=device, dtype=torch.float32)
        scale_value = self._estimate_gaussian_scale(points_norm, center_norm)
        scale = torch.full((xyz.shape[0], 3), scale_value, device=device, dtype=torch.float32)
        rotation = torch.zeros((xyz.shape[0], 4), device=device, dtype=torch.float32)
        rotation[:, 0] = 1.0
        opacity = torch.full((xyz.shape[0], 1), 0.95, device=device, dtype=torch.float32)
        color_tensor = torch.from_numpy(np.clip(colors, 0.0, 1.0)).to(device=device, dtype=torch.float32)

        gaussian_params = torch.cat([xyz, opacity, scale, rotation, color_tensor], dim=-1).unsqueeze(0)
        intr = torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32, device=device).unsqueeze(0)

        # Dynamic planes are more robust for arbitrary VGGT world scales.
        near_dynamic = max(near_plane, radius_norm * 0.01)
        far_dynamic = max(far_plane, radius_norm * 20.0)

        if not hasattr(self, "_render_variant_cache"):
            self._render_variant_cache = {}

        def render_candidate(reverse_forward: bool, basis_layout: str):
            c2w_local = build_c2w(
                look_at=center_norm,
                eye=cam_pos,
                reverse_forward=reverse_forward,
                basis_layout=basis_layout,
            )
            test_c2ws_local = torch.from_numpy(c2w_local).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
            rgb_local, _ = gaussian_render(
                gaussian_params,
                test_c2ws_local,
                intr,
                image_width,
                image_height,
                near_plane=near_dynamic,
                far_plane=far_dynamic,
                use_checkpoint=False,
                sh_degree=0,
                bg_mode="black",
            )
            rgb_img_local = rgb_local[0, 0].clamp(-1.0, 1.0).add(1.0).div(2.0)
            gray = rgb_img_local.mean(dim=0)
            non_bg_ratio = float((gray > 0.03).float().mean().item())
            std_v = float(rgb_img_local.std().item())
            score = non_bg_ratio + 0.5 * std_v
            return rgb_img_local, score, non_bg_ratio

        cached_variant = self._render_variant_cache.get(
            ply_path,
            {"reverse_forward": False, "basis_layout": "row"},
        )
        rgb_img, best_score, best_non_bg_ratio = render_candidate(
            reverse_forward=bool(cached_variant["reverse_forward"]),
            basis_layout=str(cached_variant["basis_layout"]),
        )

        # If the cached/default pose is too empty, probe multiple camera conventions once.
        if best_score < 0.03 or best_non_bg_ratio < 0.001:
            candidates = [
                {"reverse_forward": False, "basis_layout": "row"},
                {"reverse_forward": True, "basis_layout": "row"},
                {"reverse_forward": False, "basis_layout": "col"},
                {"reverse_forward": True, "basis_layout": "col"},
            ]
            best_variant = cached_variant
            for cand in candidates:
                rgb_try, score_try, non_bg_try = render_candidate(
                    reverse_forward=bool(cand["reverse_forward"]),
                    basis_layout=str(cand["basis_layout"]),
                )
                if score_try > best_score:
                    rgb_img = rgb_try
                    best_score = score_try
                    best_non_bg_ratio = non_bg_try
                    best_variant = cand
            self._render_variant_cache[ply_path] = best_variant

        # If gsplat still fails (near-empty), fallback to deterministic point projection.
        if best_score < 0.03 or best_non_bg_ratio < 0.001:
            best_variant = self._render_variant_cache.get(
                ply_path,
                {"reverse_forward": False, "basis_layout": "row"},
            )
            c2w_best = build_c2w(
                look_at=center_norm,
                eye=cam_pos,
                reverse_forward=bool(best_variant["reverse_forward"]),
                basis_layout=str(best_variant["basis_layout"]),
            )

            img_fallback = np.zeros((image_height, image_width, 3), dtype=np.float32)
            depth_buf = np.full((image_height, image_width), np.inf, dtype=np.float32)

            max_points = 300000
            if points_norm.shape[0] > max_points:
                rng = np.random.default_rng(42)
                keep_idx = rng.choice(points_norm.shape[0], size=max_points, replace=False)
                proj_points = points_norm[keep_idx]
                proj_colors = colors[keep_idx]
            else:
                proj_points = points_norm
                proj_colors = colors

            w2c = np.linalg.inv(c2w_best).astype(np.float32)
            pts_h = np.concatenate(
                [proj_points, np.ones((proj_points.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            cam_pts = (w2c @ pts_h.T).T[:, :3]

            best_proj_count = -1
            best_proj_payload = None
            for depth_sign in [1.0, -1.0]:
                z = cam_pts[:, 2] * depth_sign
                valid_z = z > 1e-4
                cam_pts_s = cam_pts[valid_z]
                z_s = z[valid_z]
                c_s = proj_colors[valid_z]
                if cam_pts_s.shape[0] == 0:
                    continue

                u = (fx * (cam_pts_s[:, 0] / z_s) + cx).astype(np.int32)
                v = (fy * (cam_pts_s[:, 1] / z_s) + cy).astype(np.int32)
                in_view = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
                view_count = int(in_view.sum())
                if view_count > best_proj_count:
                    best_proj_count = view_count
                    best_proj_payload = (u[in_view], v[in_view], z_s[in_view], c_s[in_view])

            if best_proj_payload is not None and best_proj_count > 0:
                u, v, z, c_proj = best_proj_payload
                order = np.argsort(z)
                u = u[order]
                v = v[order]
                z = z[order]
                c_proj = c_proj[order]

                for uu, vv, zz, cc in zip(u, v, z, c_proj):
                    if zz < depth_buf[vv, uu]:
                        depth_buf[vv, uu] = zz
                        img_fallback[vv, uu] = np.clip(cc, 0.0, 1.0)

                valid_mask = np.isfinite(depth_buf).astype(np.uint8)
                if valid_mask.any():
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate((img_fallback * 255).astype(np.uint8), kernel, iterations=1)
                    filled = cv2.dilate(valid_mask, kernel, iterations=1)
                    img_fallback[filled > 0] = dilated[filled > 0] / 255.0

                rgb_img = torch.from_numpy(img_fallback).permute(2, 0, 1).to(torch.float32)

        rgb_np = (
            rgb_img.mul(255.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        rgb_np = np.flipud(rgb_np)
        return Image.fromarray(rgb_np)

