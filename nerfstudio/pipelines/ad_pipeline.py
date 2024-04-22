# Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple, Type

import torch
from PIL import Image
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_pil_image, to_tensor

from nerfstudio.data.datamanagers.ad_datamanager import ADDataManager, ADDataManagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.models.ad_model import ADModel, ADModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler


@dataclass
class ADPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ADPipeline)
    """target class to instantiate"""
    datamanager: ADDataManagerConfig = field(default_factory=ADDataManagerConfig)
    """specifies the datamanager config"""
    model: ADModelConfig = field(default_factory=ADModelConfig)
    """specifies the model config"""
    calc_fid_steps: Tuple[int, ...] = (20000,)  # NOTE: must also be an eval step for this to work
    """Whether to calculate FID for lane shifted images."""
    ray_patch_size: Tuple[int, int] = (32, 32)
    """Size of the ray patches to sample from the image during training (for camera rays only)."""

    def __post_init__(self) -> None:
        assert self.ray_patch_size[0] == self.ray_patch_size[1], "Non-square patches are not supported yet, sorry."
        self.datamanager.image_divisible_by = self.model.rgb_upsample_factor


class ADPipeline(VanillaPipeline):
    """Pipeline for training AD models."""

    def __init__(self, config: ADPipelineConfig, **kwargs):
        pixel_sampler = config.datamanager.pixel_sampler
        pixel_sampler.patch_size = config.ray_patch_size[0]
        pixel_sampler.patch_scale = config.model.rgb_upsample_factor
        super().__init__(config, **kwargs)

        # Fix type hints
        self.datamanager: ADDataManager = self.datamanager
        self.model: ADModel = self.model
        self.config: ADPipelineConfig = self.config

        # Disable ray drop classification if we do not add missing points
        if not self.datamanager.dataparser.config.add_missing_points:
            self.model.disable_ray_drop()

        self.fid = None

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        # Regular forward pass and loss calc
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle, patch_size=self.config.ray_patch_size)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if (actors := self.model.dynamic_actors).config.optimize_trajectories:
            pos_norm = (actors.actor_positions - actors.initial_positions).norm(dim=-1)
            metrics_dict["traj_opt_translation"] = pos_norm[pos_norm > 0].mean().nan_to_num()
            metrics_dict["traj_opt_rotation"] = (
                (actors.actor_rotations_6d - actors.initial_rotations_6d)[pos_norm > 0].norm(dim=-1).mean().nan_to_num()
            )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, patch_size=self.config.ray_patch_size)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        # Image eval
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()

        # Lidar eval
        lidar, batch = self.datamanager.next_eval_lidar(step)
        outputs, batch = self.model.get_outputs_for_lidar(lidar, batch=batch)
        lidar_metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
        assert not set(lidar_metrics_dict.keys()).intersection(metrics_dict.keys())
        metrics_dict.update(lidar_metrics_dict)

        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

         Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager))
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            lane_shift_fids = (
                {i: FrechetInceptionDistance().to(self.device) for i in (0, 2, 3)}
                if step in self.config.calc_fid_steps or step is None
                else {}
            )
            vertical_shift_fids = (
                {i: FrechetInceptionDistance().to(self.device) for i in (1,)}
                if step in self.config.calc_fid_steps or step is None
                else {}
            )
            actor_edits = {
                "rot": [(0.5, 0), (-0.5, 0)],
                "trans": [(0, 2.0), (0, -2.0)],
                # "both": [(0.5, 2.0), (-0.5, 2.0), (0.5, -2.0), (-0.5, -2.0)],
            }
            actor_fids = (
                {k: FrechetInceptionDistance().to(self.device) for k in actor_edits.keys()}
                if step in self.config.calc_fid_steps or step is None
                else {}
            )
            if actor_fids:
                actor_fids["true"] = FrechetInceptionDistance().to(self.device)

            num_images = len(self.datamanager.fixed_indices_eval_dataloader)
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                # Generate images from the original rays
                camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                # Compute metrics for the original image
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                        )
                # Add timing stuff
                assert "num_rays_per_sec" not in metrics_dict
                num_rays = math.prod(camera_ray_bundle.shape)
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / num_rays
                metrics_dict_list.append(metrics_dict)
                if lane_shift_fids:
                    self._update_lane_shift_fid(lane_shift_fids, camera_ray_bundle, batch["image"], outputs["rgb"])
                if vertical_shift_fids:
                    self._update_vertical_shift_fid(vertical_shift_fids, camera_ray_bundle, batch["image"])
                if actor_fids:
                    self._update_actor_fids(actor_fids, actor_edits, camera_ray_bundle, batch["image"])
                progress.advance(task)
            num_lidar = len(self.datamanager.fixed_indices_eval_lidar_dataloader)
            task = progress.add_task("[green]Evaluating all eval point clouds...", total=num_lidar)
            for lidar, batch in self.datamanager.fixed_indices_eval_lidar_dataloader:
                outputs, batch = self.model.get_outputs_for_lidar(lidar, batch=batch)
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

        # average the metrics list
        metrics_dict = {}
        keys = {key for metrics_dict in metrics_dict_list for key in metrics_dict.keys()}
        # remove the keys related to actor metrics as they need to be averaged differently
        actor_keys = {key for key in keys if key.startswith("actor_")}
        keys = keys - actor_keys

        for key in keys:
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list if key in metrics_dict])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list if key in metrics_dict])
                    )
                )
        # average the actor metrics. Note that due to the way we compute the actor metrics,
        # we need to weight them by how big portion of the image they cover.
        actor_metrics_dict = [md for md in metrics_dict_list if "actor_coverage" in md]
        if actor_metrics_dict:
            actor_coverages = torch.tensor([md["actor_coverage"] for md in actor_metrics_dict])
            for key in actor_keys:
                # we dont want to average the actor coverage in this way.
                if key == "actor_coverage":
                    continue
                # we should weight the actor metrics by the actor coverage
                metrics_dict[key] = float(
                    torch.sum(
                        torch.tensor(
                            [md[key] for md in actor_metrics_dict],
                        )
                        * actor_coverages
                    )
                    / actor_coverages.sum()
                )

        # Add FID metrics (if applicable)
        for shift, fid in lane_shift_fids.items():
            metrics_dict[f"lane_shift_{shift}_fid"] = fid.compute().item()

        for shift, fid in vertical_shift_fids.items():
            metrics_dict[f"vertical_shift_{shift}_fid"] = fid.compute().item()

        if actor_fids:
            for edit_type in actor_edits.keys():
                metrics_dict[f"actor_shift_{edit_type}_fid"] = actor_fids[edit_type].compute().item()

        self.train()
        return metrics_dict

    @staticmethod
    def _downsample_img(
        img: torch.Tensor,
        out_size: Tuple[int, int] = (
            299,
            299,
        ),
    ):
        """Converts tensor to PIL, downsamples with bicubic, and converts back to tensor."""
        img = to_pil_image(img)
        img = img.resize(out_size, Image.BICUBIC)
        img = to_tensor(img)
        return img

    def _update_lane_shift_fid(self, fids: Dict[int, FrechetInceptionDistance], ray_bundle, orig_img, gen_img):
        """Updates the FID metrics (for shifted views) for the given ray bundle and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
        )
        fids_list = list(fids.values())
        fids_list[0].update(img_original, real=True)
        for fid in fids_list[1:]:
            fid.real_features_sum = fids_list[0].real_features_sum
            fid.real_features_cov_sum = fids_list[0].real_features_cov_sum
            fid.real_features_num_samples = fids_list[0].real_features_num_samples

        # Compute FID for shifted views
        assert fids.keys() == {0, 2, 3}, "Shift amounts are hardcoded for now."
        imgs_generated = {0: gen_img}

        driving_direction = ray_bundle.metadata["velocities"][0, 0, :]
        driving_direction = driving_direction / driving_direction.norm()
        orth_right_direction = torch.cross(
            driving_direction, torch.tensor([0.0, 0.0, 1.0], device=driving_direction.device)
        )

        # TODO: Do we need to take z axis into account?
        shift_sign = self.datamanager.eval_lidar_dataset.metadata.get("lane_shift_sign", 1)
        original_ray_origins = ray_bundle.origins.clone()
        ray_bundle.origins[..., :2] += 2 * orth_right_direction[:2] * shift_sign
        imgs_generated[2] = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)["rgb"]
        ray_bundle.origins[..., :2] += 1 * orth_right_direction[:2] * shift_sign
        imgs_generated[3] = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)["rgb"]
        ray_bundle.origins = original_ray_origins
        for shift, img in imgs_generated.items():
            img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
            fids[shift].update(img, real=False)

    def _update_vertical_shift_fid(self, fids: Dict[int, FrechetInceptionDistance], ray_bundle, orig_img):
        """Updates the FID metrics (for shifted views) for the given ray bundle and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
        )
        fids_list = list(fids.values())
        fids_list[0].update(img_original, real=True)
        for fid in fids_list[1:]:
            fid.real_features_sum = fids_list[0].real_features_sum
            fid.real_features_cov_sum = fids_list[0].real_features_cov_sum
            fid.real_features_num_samples = fids_list[0].real_features_num_samples

        # Compute FID for shifted views
        assert fids.keys() == {1}, "Shift amounts are hardcoded for now."
        imgs_generated = {}

        original_ray_origins = ray_bundle.origins.clone()
        ray_bundle.origins[..., 2] += 1.0
        imgs_generated[1] = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)["rgb"]
        ray_bundle.origins = original_ray_origins

        for shift, img in imgs_generated.items():
            img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
            fids[shift].update(img, real=False)

    def _update_actor_fids(
        self,
        fids: Dict[str, FrechetInceptionDistance],
        actor_edits: Dict[str, List[Tuple]],
        ray_bundle,
        orig_img,
    ) -> None:
        """Updates the FID metrics (for shifted actor views) for the given ray bundle and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
        )
        fids["true"].update(img_original, real=True)
        for edit_type in actor_edits.keys():
            fids[edit_type].real_features_sum = fids["true"].real_features_sum
            fids[edit_type].real_features_cov_sum = fids["true"].real_features_cov_sum
            fids[edit_type].real_features_num_samples = fids["true"].real_features_num_samples

        # Compute FID for actor edits
        imgs_generated_per_edit = {}
        for edit_type in actor_edits.keys():
            imgs = []
            for rotation, lateral in actor_edits[edit_type]:
                self.model.dynamic_actors.actor_editing["rotation"] = rotation
                self.model.dynamic_actors.actor_editing["lateral"] = lateral
                imgs.append(self.model.get_outputs_for_camera_ray_bundle(ray_bundle)["rgb"])
            imgs_generated_per_edit[edit_type] = imgs

        for edit_type, imgs in imgs_generated_per_edit.items():
            for img in imgs:
                img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
                fids[edit_type].update(img, real=False)

        self.model.dynamic_actors.actor_editing["rotation"] = 0
        self.model.dynamic_actors.actor_editing["lateral"] = 0
