# Copyright 2025 the authors of NeuRAD and contributors.
# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Abstracts for the Pipeline class.
"""

from __future__ import annotations

import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_pil_image, to_tensor

from nerfstudio.cameras.lidars import transform_points
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.full_images_lidar_datamanager import FullImageLidarDatamanagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler


@dataclass
class SplatADPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SplatADPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=FullImageLidarDatamanagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""
    calc_fid_steps: Tuple[int, ...] = (99999999,)  # NOTE: must also be an eval step for this to work


class SplatADPipeline(VanillaPipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: SplatADPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        elif (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "point_clouds" in self.datamanager.train_dataparser_outputs.metadata
            and "lidars" in self.datamanager.train_dataparser_outputs.metadata
        ):
            points_in_world = []
            returning_masks = []
            for l2w, pc in zip(
                self.datamanager.train_dataparser_outputs.metadata["lidars"].lidar_to_worlds,
                self.datamanager.train_dataparser_outputs.metadata["point_clouds"],
            ):
                returning = (
                    pc[:, :3].norm(dim=-1)
                    < self.datamanager.train_dataparser_outputs.metadata["lidars"].valid_lidar_distance_threshold
                )
                returning_masks.append(returning)
                points_in_world.append(transform_points(pc[returning, :3], l2w))
            points_in_world = torch.cat([pc_[:, :3] for pc_ in points_in_world], dim=0)

            if (
                "point_clouds_times" in self.datamanager.train_dataparser_outputs.metadata
                and self.datamanager.train_dataparser_outputs.metadata["point_clouds_times"] is not None
            ):
                points_in_world_times = torch.cat(
                    [
                        t_[r_]
                        for t_, r_ in zip(
                            self.datamanager.train_dataparser_outputs.metadata["point_clouds_times"], returning_masks
                        )
                    ]
                )
            else:
                points_in_world_times = None

            if (
                "point_clouds_rgb" in self.datamanager.train_dataparser_outputs.metadata
                and self.datamanager.train_dataparser_outputs.metadata["point_clouds_rgb"] is not None
            ):
                points_in_world_rgb = torch.cat(
                    [
                        c_[r_]
                        for c_, r_ in zip(
                            self.datamanager.train_dataparser_outputs.metadata["point_clouds_rgb"], returning_masks
                        )
                    ]
                )
            else:
                points_in_world_rgb = torch.rand_like(points_in_world) * 255
            seed_pts = (points_in_world, points_in_world_rgb, points_in_world_times)

        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=self.datamanager.get_num_train_data(),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()

        lidar, batch = self.datamanager.next_eval_lidar(step)
        outputs = self.model.get_lidar_outputs(lidar)
        lidar_metrics_dict, lidar_images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        images_dict.update(lidar_images_dict)
        assert not set(lidar_metrics_dict.keys()).intersection(metrics_dict.keys())
        metrics_dict.update(lidar_metrics_dict)

        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
        dump_img_to_disk: bool = False,
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
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        num_lidar = len(self.datamanager.fixed_indices_eval_lidar_dataloader)
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
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

            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)

            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                torch.cuda.synchronize()
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                torch.cuda.synchronize()
                inference_time_camera = time() - inner_start
                height, width = camera.height, camera.width
                num_camera_rays = height * width
                # Compute metrics for the original image
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                pred_height, pred_width = outputs["rgb"].shape[:2]
                batch["image"] = batch["image"][:pred_height, :pred_width]
                if dump_img_to_disk:
                    assert output_path is not None
                    os.makedirs(output_path, exist_ok=True)
                    os.makedirs(output_path / "fid", exist_ok=True)
                    os.makedirs(output_path / "fid" / "gt_rgb", exist_ok=True)
                    os.makedirs(output_path / "fid" / "pred_rgb", exist_ok=True)
                    gt_img = batch["image"]
                    if gt_img.max() > 1:
                        gt_img = gt_img / 255.0
                    pred_height, pred_width = outputs["rgb"].shape[:2]
                    gt_img = gt_img[:pred_height, :pred_width]
                    Image.fromarray((gt_img * 255).byte().cpu().numpy()).save(
                        output_path / "fid" / "gt_rgb" / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
                    )
                    Image.fromarray((outputs["rgb"] * 255).byte().cpu().numpy()).save(
                        output_path / "fid" / "pred_rgb" / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
                    )
                # if output_path is not None:
                #     raise NotImplementedError("Saving images is not implemented yet")

                assert "num_camera_rays_per_sec" not in metrics_dict
                metrics_dict["num_camera_rays_per_sec"] = (num_camera_rays / inference_time_camera).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_camera_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)

                if lane_shift_fids:
                    if dump_img_to_disk:
                        assert output_path is not None
                        for shift in lane_shift_fids.keys():
                            if shift == 0:
                                continue
                            os.makedirs(output_path / "fid" / f"lane_shift_{shift}", exist_ok=True)
                    self._update_lane_shift_fid(
                        lane_shift_fids, camera, batch["image"], outputs["rgb"], dump_img_to_disk, output_path
                    )
                if vertical_shift_fids:
                    if dump_img_to_disk:
                        assert output_path is not None
                        for shift in vertical_shift_fids.keys():
                            os.makedirs(output_path / "fid" / f"vertical_shift_{shift}", exist_ok=True)
                    self._update_vertical_shift_fid(
                        vertical_shift_fids, camera, batch["image"], dump_img_to_disk, output_path
                    )
                if actor_fids:
                    if dump_img_to_disk:
                        assert output_path is not None
                        for edit_type, edit_amounts in actor_edits.items():
                            for edit_amount in edit_amounts:
                                edit_amount = edit_amount[0] if edit_type == "rot" else edit_amount[1]
                                mount_prefix = "neg" if edit_amount < 0 else "pos"
                                if abs(edit_amount - int(edit_amount)) < 1e-6:
                                    edit_amount = mount_prefix + str(int(edit_amount))
                                else:
                                    edit_amount = mount_prefix + str(edit_amount).replace(".", "0")
                                os.makedirs(
                                    output_path / "fid" / f"actor_shift_{edit_type}_{edit_amount}", exist_ok=True
                                )
                    self._update_actor_fids(
                        actor_fids, actor_edits, camera, batch["image"], dump_img_to_disk, output_path
                    )
                progress.advance(task)

            task = progress.add_task("[green]Evaluating all eval point clouds...", total=num_lidar)
            for lidar, batch in self.datamanager.fixed_indices_eval_lidar_dataloader:
                torch.cuda.synchronize()
                inner_start = time()
                outputs = self.model.get_lidar_outputs(lidar)
                torch.cuda.synchronize()
                inference_time_lidar = time() - inner_start
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                num_lidar_rays = (batch["raster_pts"][..., 2] > 0).sum()
                assert "num_lidar_rays_per_sec" not in metrics_dict
                metrics_dict["num_lidar_rays_per_sec"] = (num_lidar_rays / inference_time_lidar).item()
                metrics_dict_list.append(metrics_dict)
                if dump_img_to_disk:
                    assert output_path is not None
                    os.makedirs(output_path / "fid" / "lidar", exist_ok=True)
                    gt_points = batch["lidar"][batch["lidar_pts_did_return"]]  # N, 5 (xyz, intensity, time_offset)
                    # if filter_lidar_pred_and_gt is a function of the model, call it here
                    if hasattr(self.model, "filter_lidar_pred_and_gt"):
                        lidar_pred, lidar_gt = self.model.filter_lidar_pred_and_gt(
                            outputs, batch, output_point_cloud=True
                        )
                        pred_points = lidar_pred["point_cloud"]  # M, 3
                        pred_points_median = lidar_pred["median_point_cloud"]
                        pred_points_mask = (lidar_pred["ray_drop"].sigmoid() <= 0.5) * lidar_gt["valid"]
                        intensity = outputs["intensity"].flatten()[pred_points_mask]
                        time_offset = batch["raster_pts"][..., 3].flatten()[pred_points_mask]
                        pred_points = torch.cat(
                            [pred_points, intensity[..., None], time_offset[..., None]], dim=-1
                        )  # M, 5 (xyz, intensity, time_offset)
                        pred_points_median = torch.cat(
                            [pred_points_median, intensity[..., None], time_offset[..., None]], dim=-1
                        )  # M, 5 (xyz, intensity, time_offset)

                        # save the pred_points and gt_points to a file
                        np.savez(
                            output_path / "fid" / "lidar" / f"points_{str(lidar.metadata['cam_idx']).zfill(6)}.npz",
                            pred_points=pred_points.cpu().numpy(),
                            pred_points_median=pred_points_median.cpu().numpy(),
                            gt_points=gt_points.cpu().numpy(),
                        )
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

    def _update_lane_shift_fid(
        self, fids: Dict[int, FrechetInceptionDistance], camera, orig_img, gen_img, dump_img_to_disk, output_path
    ):
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

        driving_direction = camera.metadata["velocities"][0].clone()
        driving_direction = driving_direction / driving_direction.norm()
        orth_right_direction = torch.cross(
            driving_direction, torch.tensor([0.0, 0.0, 1.0], device=driving_direction.device)
        )

        # TODO: Do we need to take z axis into account?
        shift_sign = self.datamanager.eval_lidar_dataset.metadata.get("lane_shift_sign", 1)
        original_camera_to_worlds = camera.camera_to_worlds.clone()
        camera.camera_to_worlds[0, :2, 3] += 2 * orth_right_direction[:2] * shift_sign
        imgs_generated[2] = self.model.get_outputs_for_camera(camera)["rgb"]
        camera.camera_to_worlds[0, :2, 3] += 1 * orth_right_direction[:2] * shift_sign
        imgs_generated[3] = self.model.get_outputs_for_camera(camera)["rgb"]
        camera.camera_to_worlds = original_camera_to_worlds
        for shift, img in imgs_generated.items():
            if dump_img_to_disk:
                if shift == 0:
                    continue
                assert output_path is not None
                fid_output_path = output_path / "fid" / f"lane_shift_{shift}"
                filepath = fid_output_path / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
                Image.fromarray((img * 255).byte().cpu().numpy()).save(filepath)
            img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
            fids[shift].update(img, real=False)

    def _update_vertical_shift_fid(
        self, fids: Dict[int, FrechetInceptionDistance], camera, orig_img, dump_img_to_disk, output_path
    ):
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

        original_camera_to_worlds = camera.camera_to_worlds.clone()
        camera.camera_to_worlds[0, 2, 3] += 1.0
        imgs_generated[1] = self.model.get_outputs_for_camera(camera)["rgb"]
        camera.camera_to_worlds = original_camera_to_worlds

        if dump_img_to_disk:
            assert output_path is not None
            fid_output_path = output_path / "fid" / "vertical_shift_1"
            filepath = fid_output_path / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
            Image.fromarray((imgs_generated[1] * 255).byte().cpu().numpy()).save(filepath)

        for shift, img in imgs_generated.items():
            img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
            fids[shift].update(img, real=False)

    def _update_actor_fids(
        self,
        fids: Dict[str, FrechetInceptionDistance],
        actor_edits: Dict[str, List[Tuple]],
        camera,
        orig_img,
        dump_img_to_disk,
        output_path,
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
                prediction = self.model.get_outputs_for_camera(camera)["rgb"]
                imgs.append(prediction)
                if dump_img_to_disk:
                    assert output_path is not None
                    edit_amount = rotation if edit_type == "rot" else lateral
                    mount_prefix = "neg" if edit_amount < 0 else "pos"
                    if abs(edit_amount - int(edit_amount)) < 1e-6:
                        edit_amount = mount_prefix + str(int(edit_amount))
                    else:
                        edit_amount = mount_prefix + str(edit_amount).replace(".", "0")
                    fid_output_path = output_path / "fid" / f"actor_shift_{edit_type}_{edit_amount}"
                    filepath = fid_output_path / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
                    Image.fromarray((prediction * 255).byte().cpu().numpy()).save(filepath)
            imgs_generated_per_edit[edit_type] = imgs

        for edit_type, imgs in imgs_generated_per_edit.items():
            for img in imgs:
                img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
                fids[edit_type].update(img, real=False)

        self.model.dynamic_actors.actor_editing["rotation"] = 0
        self.model.dynamic_actors.actor_editing["lateral"] = 0
