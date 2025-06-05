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

"""Base data parser for Autonomous Driving datasets."""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import Lidars
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils import poses as pose_utils
from nerfstudio.utils.poses import interpolate_trajectories, multiply as pose_multiply, to4x4

SensorData = TypeVar("SensorData", Cameras, Lidars)


class SplitTypes(Enum):
    """Split types for train/eval."""

    LINSPACE = "linspace"
    NUSCENES_SAMPLES = "nuscenes_samples"


OPENCV_TO_NERFSTUDIO = np.array(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]
)
DUMMY_DISTANCE_VALUE = 2e3  # meters, used for missing points


@dataclass
class ADDataParserConfig(DataParserConfig):
    """AD (autonomous driving) dataset config."""

    _target: Type = field(default_factory=lambda: ADDataParser)
    """target class to instantiate"""
    sequence: str = "001"
    """Name of the sequence to load."""
    train_split_fraction: float = 0.5
    """The percent of images to use for training. The remaining images are for eval."""
    train_eval_split_type: SplitTypes = SplitTypes.LINSPACE
    """The type of split to use for train/eval."""
    max_eval_frames: Optional[int] = None
    """The maximum number of frames to use for eval. If None, use all eval frames."""
    dataset_start_fraction: float = 0.0
    """At what fraction of the dataset to start."""
    dataset_end_fraction: float = 1.0
    """At what fraction of the dataset to end."""
    scene_box_height = (-10, 30)
    """The upper and lower height bounds for the scene box (m)."""

    cameras: Tuple[str, ...] = tuple()
    """Which cameras to use."""
    lidars: Tuple[str, ...] = tuple()
    """Which lidars to use."""
    min_lidar_dist: Tuple[float, float, float] = (1.0, 2.0, 2.0)
    """Remove all points within this distance from lidar sensor."""
    radars: Tuple[str, ...] = tuple()
    """Which radars to use."""

    load_cuboids: bool = True
    """Whether to load cuboid annotations."""
    include_deformable_actors: bool = False
    """Whether to include deformable actors in the loaded trajectories (like pedestrians)."""
    annotation_interval: float = 0.0
    """The time interval at which the sequence is annotated (s)."""
    trajectory_extrapolation_length: float = 1.0
    """The amount of time to extrapolate the trajectory (s)."""
    rolling_shutter_time: float = 0.0
    """The rolling shutter time for the cameras (seconds)."""
    time_to_center_pixel: float = 0.0
    """The time offset for the center pixel, relative to the image timestamp (seconds)."""
    allow_per_point_times: bool = True
    """Whether to allow per-point times (for sub-frame time correction)."""

    add_missing_points: bool = False
    """Whether to add missing points (rays that did not return) to the point clouds."""
    lidar_elevation_mapping: Optional[Dict[str, Dict[int, float]]] = None
    """Elevation mapping for each lidar."""
    skip_elevation_channels: Optional[Dict[str, Tuple[int, ...]]] = None
    """Channels to skip when adding missing points."""
    lidar_azimuth_resolution: Optional[Dict[str, float]] = None
    """Azimuth resolution for each lidar."""

    def __post_init__(self):
        if type(self) == ADDataParserConfig:
            return  # only make the following checks for child classes
        # Hack to allow settings empty tuple on command line
        if len(self.cameras) == 1 and self.cameras[0].lower() == "none":
            self.cameras = ()
        if len(self.lidars) == 1 and self.lidars[0].lower() == "none":
            self.lidars = ()
        if len(self.radars) == 1 and self.radars[0].lower() == "none":
            self.radars = ()
        assert self.cameras or self.lidars or self.radars, "Must specify at least one sensor to load"
        assert self.annotation_interval > 1e-6, "Child classes must specify the annotation interval"
        assert self.dataset_start_fraction >= 0.0, "Dataset start fraction must be >= 0.0"
        assert self.dataset_end_fraction <= 1.0, "Dataset end fraction must be <= 1.0"
        assert self.dataset_start_fraction < self.dataset_end_fraction, "Dataset start must be < dataset end"


@dataclass
class ADDataParser(DataParser):
    """PandaSet DatasetParser"""

    config: ADDataParserConfig
    includes_time: bool = True

    @property
    def actor_transform(self) -> Tensor:
        """The transform to convert from our actor frame (x-right, y-forward, z-up) to the original actor frame."""
        return torch.eye(4)[:3, :]

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        """Returns camera info and image filenames."""
        raise NotImplementedError()

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        """Returns lidar info and filenames."""
        raise NotImplementedError()

    def _read_lidars(self, lidars: Lidars, filenames: List[Path]) -> List[Tensor]:
        """Reads the point clouds from the given filenames. Should be in x,y,z,r,t order. t is optional."""
        raise NotImplementedError()

    def _get_actor_trajectories(self) -> List[Dict]:
        """Returns a list of actor trajectories.

        Each trajectory is a dictionary with the following keys:
            - poses: the poses of the actor (float32)
            - timestamps: the timestamps of the actor (float64)
            - dims: the dimensions of the actor, wlh order (float32)
            - label: the label of the actor (str)
            - stationary: whether the actor is stationary (bool)
            - symmetric: whether the actor is expected to be symmetric (bool)
            - deformable: whether the actor is expected to be deformable (e.g. pedestrian)
        """
        raise NotImplementedError()

    def _get_radars(self):
        """Returns a list of radars."""
        raise NotImplementedError()

    def _get_lane_shift_sign(self, sequence: str) -> Literal[-1, 1]:
        """Get the sign of the most realistic lane shift for the given sequence.

        This is to avoid shifting into unreasonable locations, that do not correspond
        to a realistic lane shift (like shifting into a parked car, wall, etc.)
        """
        return 1

    def _generate_dataparser_outputs(self, split="train"):
        # Load data (this is implemented in the dataset-specific subclass)
        cameras, img_filenames = self._get_cameras() if self.config.cameras else (_empty_cameras(), [])
        lidars, pc_filenames = self._get_lidars() if self.config.lidars else (_empty_lidars(), [])
        radars = self._get_radars() if self.config.radars else None
        assert radars is None, "Radars not supported yet"
        trajectories = self._get_actor_trajectories() if self.config.load_cuboids else []
        # use dataset fraction to filter data
        cameras, img_filenames, lidars, pc_filenames, trajectories = self._filter_based_on_time(
            cameras, img_filenames, lidars, pc_filenames, trajectories
        )
        # read all the point clouds
        point_clouds = self._read_lidars(lidars, pc_filenames)

        # Process the data
        self._remove_ego_points(point_clouds)
        time_offset = self._adjust_times(cameras, lidars, point_clouds, trajectories)
        dataparser_transform = self._adjust_poses(cameras, lidars, trajectories)
        scene_box = self._compute_scene_box(cameras, lidars)
        self._add_sensor_velocities(cameras, lidars)

        # Adjust sensor idxs to avoid overlap (if necessary)
        cam_sensor_idxs = cameras.metadata["sensor_idxs"].unique()
        if set(lidars.metadata["sensor_idxs"].unique().tolist()).intersection(set(cam_sensor_idxs.tolist())):
            assert cam_sensor_idxs.max() == cam_sensor_idxs.numel() - 1, "Sensor idxs must be contiguous"
            lidars.metadata["sensor_idxs"] += cam_sensor_idxs.numel()
        sensor_idx_to_name = {idx: name for idx, name in enumerate(self.config.cameras + self.config.lidars)}

        # Apply splits
        split = 0 if split == "train" else 1
        cam_idxs = self._get_train_eval_indices(cameras)[split]
        cameras, img_filenames = cameras[cam_idxs], [img_filenames[i] for i in cam_idxs]
        lidar_idxs = self._get_train_eval_indices(lidars)[split]
        lidars, point_clouds = lidars[lidar_idxs], [point_clouds[i] for i in lidar_idxs]

        # sensor_times = torch.cat([cameras.times, lidars.times], dim=0).squeeze(-1).unique()
        sensor_times = lidars.times.squeeze(-1).unique()
        trajectories = self._interpolate_trajectories(
            trajectories, sensor_times, self.config.trajectory_extrapolation_length
        )
        if len(trajectories) > 0:
            trajectories = self._add_trajectories_velocities(trajectories)

        # remove empty trajectories
        trajectories = [traj for traj in trajectories if len(traj["timestamps"]) > 1]

        point_clouds_times = [t.repeat(len(point_clouds[i])) for i, t in enumerate(lidars.times.squeeze(-1))]

        # Massage into output format
        return DataparserOutputs(
            image_filenames=img_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=None,  # TODO: handle masks
            dataparser_scale=1.0,  # no scaling
            dataparser_transform=dataparser_transform,
            actor_transform=self.actor_transform,
            time_offset=time_offset,
            metadata={
                "lidars": lidars,
                "point_clouds": [pc.float() for pc in point_clouds],  # Ensure they are float32
                "point_clouds_times": point_clouds_times,
                "trajectories": trajectories,
                "lane_shift_sign": self._get_lane_shift_sign(self.config.sequence),
                "sensor_idx_to_name": sensor_idx_to_name,
                "duration": torch.cat([cameras.times, lidars.times]).max()
                - torch.cat([cameras.times, lidars.times]).min(),
            },
        )

    def _compute_scene_box(self, cameras: Cameras, lidars: Lidars) -> SceneBox:
        if len(lidars):
            # Pad 60 meters left/right and 80 meters forward/backward
            og_poses = lidars.lidar_to_worlds
            padded_poses = _get_padded_poses(og_poses, 60.0, 80.0)
        else:
            # Need to figure out padding here, since cameras can point in all directions
            raise NotImplementedError("Scene box computation using cameras not implemented yet")
            og_poses = cameras.camera_to_worlds
            padded_poses = _get_padded_poses(og_poses, 80.0, 80.0)
        poses = torch.concatenate([padded_poses, og_poses[:, :3, 3]])
        aabb_scale = poses.abs().max(dim=0)[0]
        aabb = torch.stack([-aabb_scale, aabb_scale], dim=0)
        aabb[1, 2] = self.config.scene_box_height[1]
        aabb[0, 2] = self.config.scene_box_height[0]
        return SceneBox(aabb=aabb)

    def _remove_ego_points(self, point_clouds: List[Tensor]):
        dist_thresh = torch.tensor(self.config.min_lidar_dist)
        for i, pc in enumerate(point_clouds):
            mask = (pc[:, :3].abs() >= dist_thresh).any(-1)
            point_clouds[i] = pc[mask]

    def _filter_based_on_time(
        self,
        cameras: Cameras,
        img_filenames: List[Path],
        lidars: Lidars,
        pc_filenames: List[Path],
        trajectories: List[Dict],
    ) -> Tuple[Cameras, List[Path], Lidars, List[Path], List[Dict]]:
        # Remove the data that is outside the dataset start/end fraction
        if self.config.dataset_start_fraction == 0.0 and self.config.dataset_end_fraction == 1.0:
            return cameras, img_filenames, lidars, pc_filenames, trajectories

        times = torch.cat([cameras.times, lidars.times], dim=0)
        end_time = times.max().item()
        start_time = times.min().item()
        duration = end_time - start_time
        end_time = start_time + duration * self.config.dataset_end_fraction
        start_time += duration * self.config.dataset_start_fraction

        cameras, img_filenames = _filter_sensordata_on_time(cameras, img_filenames, start_time, end_time)
        lidars, pc_filenames = _filter_sensordata_on_time(lidars, pc_filenames, start_time, end_time)
        assert len(cameras) or len(lidars), "No cameras or lidars in the dataset"
        # filter the trajectories that are not in the sequence at all
        trajectories = [
            traj
            for traj in trajectories
            if (traj["timestamps"] >= start_time).any() and (traj["timestamps"] <= end_time).any()
        ]

        return cameras, img_filenames, lidars, pc_filenames, trajectories

    def _adjust_times(
        self, cameras: Cameras, lidars: Lidars, point_clouds: List[Tensor], trajectories: List[Dict]
    ) -> float:
        times = torch.cat([cameras.times, lidars.times], dim=0)
        min_time = times.min().item()
        # Set the times for the cameras and lidars to be relative to the min time
        cameras.times = (cameras.times - min_time).float()
        lidars.times = (lidars.times - min_time).float()
        for traj in trajectories:
            traj["timestamps"] = (traj["timestamps"] - min_time).float()
        for pc in point_clouds:
            # Sometimes we have only x,y,z,r sometimes x,y,z,r,t
            if pc.shape[-1] > 4 and not self.config.allow_per_point_times:
                pc[..., 4] = 0.0
        return min_time

    def _adjust_poses(self, cameras: Cameras, lidars: Lidars, trajectories: List[Dict]):
        """Determines a new, centered, world coordinate system, and adjusts all poses."""
        w2m = _get_world_to_mean_transform(cameras, lidars)
        cameras.camera_to_worlds = pose_multiply(w2m, cameras.camera_to_worlds)
        lidars.lidar_to_worlds = pose_multiply(w2m, lidars.lidar_to_worlds)
        for traj in trajectories:
            traj["poses"][:, :3] = pose_multiply(w2m, traj["poses"][:, :3])
        return w2m

    def _get_train_eval_indices(self, sensors: Union[Cameras, Lidars]) -> Tuple[Tensor, Tensor]:
        if self.config.train_eval_split_type == SplitTypes.LINSPACE:
            return self._get_linspaced_indices(sensors.metadata["sensor_idxs"].squeeze(-1))
        raise NotImplementedError(
            f"Split type {self.config.train_eval_split_type} not implemented for this dataparser, override _get_train_eval_indices."
        )

    def _get_linspaced_indices(self, sensor_idxs: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns indices of a linspaced subset of length `length` with ratio `ratio` with contiguous chunks."""
        if sensor_idxs.numel() == 0:
            return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
        elif self.config.train_split_fraction == 1.0:
            # if we are using the full dataset, just return all indices
            # we will use this for both trian and val, as we do not mean to use the validation performance for anything
            # only for visual inspection
            train_indices = torch.arange(sensor_idxs.numel(), dtype=torch.int64)
            eval_indices = train_indices.clone()
        else:
            train_indices = []
            for sensor_idx in sensor_idxs.unique():
                sensor_sample_idxs = (sensor_idxs == sensor_idx).nonzero().squeeze(-1)
                # split according to train_split_fraction
                num_for_train_split = math.ceil(len(sensor_sample_idxs) * self.config.train_split_fraction)
                sensor_train_split = np.linspace(0, len(sensor_sample_idxs) - 1, num_for_train_split, dtype=np.int64)
                train_indices += sensor_sample_idxs[sensor_train_split]
            eval_indices = np.setdiff1d(np.arange(len(sensor_idxs)), train_indices)
            train_indices, eval_indices = torch.tensor(train_indices), torch.from_numpy(eval_indices)

        if self.config.max_eval_frames is not None:
            torch.manual_seed(123)
            eval_indices = eval_indices[torch.randperm(len(eval_indices))[: self.config.max_eval_frames]]
        return train_indices, eval_indices

    def _add_sensor_velocities(self, cameras: Cameras, lidars: Lidars):
        """Adds the sensor velocities to the metadata."""
        assert cameras.metadata is not None and lidars.metadata is not None, "Must have metadata"
        cameras.metadata["velocities"] = torch.zeros_like(cameras.camera_to_worlds[:, :3, 3])
        cameras.metadata["linear_velocities_local"] = torch.zeros_like(cameras.camera_to_worlds[:, :3, 3])
        cameras.metadata["angular_velocities_local"] = torch.zeros_like(cameras.camera_to_worlds[:, :3, 3])
        for sensor_idx in cameras.metadata["sensor_idxs"].unique():
            mask = (cameras.metadata["sensor_idxs"] == sensor_idx).squeeze(-1)
            cam2worlds, times = cameras.camera_to_worlds[mask], cameras.times[mask]
            translation_velo = (cam2worlds[1:, :3, 3] - cam2worlds[:-1, :3, 3]) / (times[1:] - times[:-1])
            next_cam = cam2worlds[1:]
            prev_cam = cam2worlds[:-1]
            next_cam_2_prev_cam = pose_utils.to4x4(pose_utils.inverse(prev_cam)) @ pose_utils.to4x4(next_cam)
            translation_velo_cam_ref = next_cam_2_prev_cam[:, :3, 3] / (times[1:] - times[:-1])
            angular_velo = pose_utils.rotation_difference(cam2worlds[:-1, :3, :3], cam2worlds[1:, :3, :3]) / (
                times[1:] - times[:-1]
            )
            cameras.metadata["velocities"][mask] = torch.cat((translation_velo, translation_velo[-1:]), 0)
            cameras.metadata["linear_velocities_local"][mask] = torch.cat(
                (translation_velo_cam_ref, translation_velo_cam_ref[-1:]), 0
            )

            cameras.metadata["angular_velocities_local"][mask] = torch.cat((angular_velo, angular_velo[-1:]), 0)
        cameras.metadata["rolling_shutter_time"] = (
            torch.tensor(self.config.rolling_shutter_time).unsqueeze(0).repeat(len(cameras), 1)
        )
        cameras.metadata["time_to_center_pixel"] = (
            torch.tensor(self.config.time_to_center_pixel).unsqueeze(0).repeat(len(cameras), 1)
        )

        lidars.metadata["velocities"] = torch.zeros_like(lidars.lidar_to_worlds[:, :3, 3])
        lidars.metadata["linear_velocities_local"] = torch.zeros_like(lidars.lidar_to_worlds[:, :3, 3])
        lidars.metadata["angular_velocities_local"] = torch.zeros_like(lidars.lidar_to_worlds[:, :3, 3])
        for sensor_idx in lidars.metadata["sensor_idxs"].unique():
            mask = (lidars.metadata["sensor_idxs"] == sensor_idx).squeeze(-1)
            lidar2worlds, times = lidars.lidar_to_worlds[mask], lidars.times[mask]
            translation_velo = (lidar2worlds[1:, :3, 3] - lidar2worlds[:-1, :3, 3]) / (times[1:] - times[:-1])
            next_lidar = lidar2worlds[1:]
            prev_lidar = lidar2worlds[:-1]
            next_lidar_in_prev_lidar = pose_utils.to4x4(pose_utils.inverse(prev_lidar)) @ pose_utils.to4x4(next_lidar)
            translation_velo_lidar_ref = next_lidar_in_prev_lidar[:, :3, 3] / (times[1:] - times[:-1])
            angular_velo = pose_utils.rotation_difference(lidar2worlds[:-1, :3, :3], lidar2worlds[1:, :3, :3]) / (
                times[1:] - times[:-1]
            )
            lidars.metadata["velocities"][mask] = torch.cat((translation_velo, translation_velo[-1:]), 0)
            lidars.metadata["linear_velocities_local"][mask] = torch.cat(
                (translation_velo_lidar_ref, translation_velo_lidar_ref[-1:]), 0
            )
            lidars.metadata["angular_velocities_local"][mask] = torch.cat((angular_velo, angular_velo[-1:]), 0)

    def _interpolate_trajectories(self, trajectories: List[Dict], timestamps: Tensor, extrapolation_length: float):
        # sort query times, just to be sure
        timestamps = timestamps.sort().values
        for i, traj in enumerate(trajectories):
            assert torch.all(traj["timestamps"][1:] >= traj["timestamps"][:-1]), "Trajectory timestamps must be sorted"
            # Only query for times between min and max of trajectory
            query_times = timestamps[timestamps >= traj["timestamps"][0] - extrapolation_length]
            query_times = query_times[query_times <= traj["timestamps"][-1] + extrapolation_length]
            # query_times = timestamps
            # interpolate_poses expects num_poses x num_trajectories x 4 x 4, so we unsqueeze to get a single trajectory
            new_poses, _, _ = interpolate_trajectories(
                traj["poses"].unsqueeze(1), traj["timestamps"], query_times, clamp_frac=False
            )
            traj["poses"] = to4x4(new_poses)
            traj["timestamps"] = query_times
        return trajectories

    def _add_trajectories_velocities(self, trajectories: List[Dict]):
        cloned_trajectories = deepcopy(trajectories)
        unique_timestamps = torch.cat([traj["timestamps"] for traj in cloned_trajectories]).unique()
        interpolated_trajectories = self._interpolate_trajectories(cloned_trajectories, unique_timestamps, 1e6)
        for i, traj in enumerate(trajectories):
            interpolated_traj = interpolated_trajectories[i]
            linear_velocities_global = (
                interpolated_traj["poses"][1:, :3, 3] - interpolated_traj["poses"][:-1, :3, 3]
            ) / (interpolated_traj["timestamps"][1:] - interpolated_traj["timestamps"][:-1]).reshape(-1, 1)
            linear_velocities_global = torch.cat((linear_velocities_global, linear_velocities_global[-1:]), 0)
            angular_velocities_local = pose_utils.rotation_difference(
                interpolated_traj["poses"][:-1, :3, :3], interpolated_traj["poses"][1:, :3, :3]
            ) / (interpolated_traj["timestamps"][1:] - interpolated_traj["timestamps"][:-1]).reshape(-1, 1)
            angular_velocities_local = torch.cat((angular_velocities_local, angular_velocities_local[-1:]), 0)
            # insert at the correct timestamps
            traj["linear_velocities_global"] = torch.zeros_like(traj["poses"][..., :3])
            traj["linear_velocities_global"] = linear_velocities_global
            traj["angular_velocities_local"] = torch.zeros_like(traj["poses"])
            traj["angular_velocities_local"] = angular_velocities_local
        return trajectories

    @staticmethod
    def _remove_ego_motion_compensation(
        point_cloud: torch.Tensor, l2ws: torch.Tensor, times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Removes ego motion compensation from point cloud.

        Args:
            point_cloud: Point cloud to remove ego motion compensation from (in world frame). Shape: [num_points, 5+N] x,y,z,intensity,timestamp,(channel_id)
            l2ws: Poses of the lidar. Shape: [num_poses, 4, 4]
            times: Timestamps of the lidar poses. Shape: [num_poses]

        Returns:
            Point cloud without ego motion compensation in sensor frame. Shape: [num_points, 5+N] x,y,z,intensity,timestamp,(channel_id)
            Lidar pose for each point in the point cloud. Shape: [num_points, 4, 4]
        """

        interpolated_l2ws, _, _ = interpolate_trajectories(
            l2ws.unsqueeze(1), times - times.min(), point_cloud[:, 4] - times.min(), clamp_frac=False
        )
        interpolated_l2ws = interpolated_l2ws[:, :3, :4]
        interpolated_w2ls = pose_utils.inverse(interpolated_l2ws)
        homogen_points = torch.cat([point_cloud[:, :3], torch.ones_like(point_cloud[:, -1:])], dim=-1)
        points = torch.matmul(interpolated_w2ls, homogen_points.unsqueeze(-1))[:, :, 0]
        return torch.cat([points, point_cloud[:, 3:]], dim=-1), interpolated_l2ws

    def _get_missing_points(
        self,
        point_cloud: torch.Tensor,
        l2ws: torch.Tensor,
        lidar_name: str,
        dist_cutoff: float = 1.0,
        ignore_regions: list = [],
        outlier_thresh: float = 0.2,
    ) -> torch.Tensor:
        """Finds missing points in the point cloud according to sensor spec (self.config.lidar_elevation_mapping)

        Args:
            point_cloud: Point cloud to find missing points in (in sensor frame). Shape: [num_points, 4+x] x,y,z,channel_id(timestamp, intensity, etc.)
            l2ws: Poses of the lidar. Shape: [num_points, 4, 4]
            lidar_name: Name of the lidar
            dist_cutoff: Distance cutoff for points to consider. Points closer than this will be ignored.
            ignore_regions: List of regions to ignore. Each region is a list of [min_azimuth, max_azimuth, min_elevation, max_elevation]
            outlier_thresh: Threshold for outlier elevation values. If the median elevation of the missing points is more than this value away from the median elevation of the points in the channel, we ignore the missing points.

        Returns:
            Missing points in the point cloud, in world_frame. Shape: [num_points, 4+x] x,y,z,channel_id(timestamp, intensity, etc.)
        """
        dist = torch.norm(point_cloud[:, :3], dim=-1)
        dist_mask = dist > dist_cutoff
        dist = dist[dist_mask]
        point_cloud = point_cloud[dist_mask]
        l2ws = l2ws[dist_mask]
        elevation = torch.arcsin(point_cloud[:, 2] / dist)
        elevation = torch.rad2deg(elevation)
        azimuth = torch.atan2(point_cloud[:, 1], point_cloud[:, 0])
        azimuth = torch.rad2deg(azimuth)

        # find missing points
        missing_points = []
        missing_points_sensor_frame = []
        assert self.config.lidar_elevation_mapping is not None, "Must specify lidar elevation mapping"
        assert self.config.lidar_azimuth_resolution is not None, "Must specify lidar azimuth resolution"
        assert self.config.skip_elevation_channels is not None, "Must specify skip elevation channels"
        for channel_id, expected_elevation in self.config.lidar_elevation_mapping[lidar_name].items():
            if channel_id in self.config.skip_elevation_channels[lidar_name]:
                continue
            channel_mask = (point_cloud[:, 3] - channel_id).abs() < 0.1  # handle floats
            curr_azimuth = azimuth[channel_mask]
            curr_l2ws = l2ws[channel_mask]
            curr_elev = elevation[channel_mask]
            if not channel_mask.any():
                continue
            curr_azimuth, sort_idx = curr_azimuth.sort()
            curr_l2ws = curr_l2ws[sort_idx]
            curr_elev = curr_elev[sort_idx]

            # find missing azimuths, we should have 360 / lidar_azimuth_resolution azimuths
            num_expected_azimuths = int(360 / self.config.lidar_azimuth_resolution[lidar_name]) + 1
            expected_idx = torch.arange(num_expected_azimuths, device=curr_azimuth.device)
            # find offset
            offset = curr_azimuth[0] % self.config.lidar_azimuth_resolution[lidar_name]
            current_idx = (
                ((curr_azimuth - offset + 180) / self.config.lidar_azimuth_resolution[lidar_name]).round().int()
            )
            missing_idx = expected_idx[torch.isin(expected_idx, current_idx, invert=True)]
            # interpolate missing azimuths
            missing_azimuth = (
                torch.from_numpy(
                    np.interp(
                        missing_idx,
                        torch.cat([torch.tensor(-1).view(1), current_idx, torch.tensor(num_expected_azimuths).view(1)]),
                        torch.cat(
                            [
                                (-180 + offset - self.config.lidar_azimuth_resolution[lidar_name]).view(1),
                                curr_azimuth,
                                (180 + offset + self.config.lidar_azimuth_resolution[lidar_name]).view(1),
                            ]
                        ),
                    )
                )
                .float()
                .view(-1, 1)
            )
            missing_elevation = (
                torch.from_numpy(
                    np.interp(
                        missing_idx,
                        torch.cat([torch.tensor(-1).view(1), current_idx, torch.tensor(num_expected_azimuths).view(1)]),
                        torch.cat(
                            [
                                torch.tensor(expected_elevation).view(1),
                                curr_elev.view(-1),
                                torch.tensor(expected_elevation).view(1),
                            ]
                        ),
                    )
                )
                .float()
                .view(-1, 1)
            )
            elevation_outlier_mask = (missing_elevation - curr_elev.median()).abs() > outlier_thresh
            missing_elevation[elevation_outlier_mask] = curr_elev.median()

            ignore_mask = torch.zeros_like(missing_azimuth.squeeze(-1)).bool()
            for ignore_region in ignore_regions:
                ignore_mask = (
                    (missing_azimuth > ignore_region[0])
                    & (missing_azimuth < ignore_region[1])
                    & (missing_elevation > ignore_region[2])
                    & (missing_elevation < ignore_region[3])
                ).squeeze(-1) | ignore_mask
            missing_azimuth = missing_azimuth[~ignore_mask]
            missing_elevation = missing_elevation[~ignore_mask]

            # missing_elevation = torch.ones_like(missing_azimuth) * current_elevation
            missing_distance = torch.ones_like(missing_azimuth) * DUMMY_DISTANCE_VALUE
            x = (
                torch.cos(torch.deg2rad(missing_azimuth))
                * torch.cos(torch.deg2rad(missing_elevation))
                * missing_distance
            )
            y = (
                torch.sin(torch.deg2rad(missing_azimuth))
                * torch.cos(torch.deg2rad(missing_elevation))
                * missing_distance
            )
            z = torch.sin(torch.deg2rad(missing_elevation)) * missing_distance
            points = torch.cat([x, y, z], dim=-1)
            missing_points_sensor_frame.append(points)
            # transform points from sensor space to world space
            points = torch.cat([points, torch.ones_like(points[:, -1:])], dim=-1)
            # find closest pose idx
            closest_pose_idx = torch.searchsorted(curr_azimuth, missing_azimuth.squeeze())
            closest_pose_idx = closest_pose_idx.clamp(max=len(curr_azimuth) - 1)
            if closest_pose_idx.shape == ():
                closest_pose_idx = closest_pose_idx.unsqueeze(0)
            closest_l2w = curr_l2ws[closest_pose_idx]
            points = torch.matmul(closest_l2w.float(), points.unsqueeze(-1))[:, :, 0]
            closest_pc_values = point_cloud[channel_mask][sort_idx][closest_pose_idx, 3:]
            points = torch.cat([points, closest_pc_values], dim=-1)
            missing_points.append(points)

        missing_points = torch.cat(missing_points, dim=0)
        missing_points_sensor_frame = torch.cat(missing_points_sensor_frame, dim=0)
        return missing_points


def _get_mean_pose_from_trajectory(trajectory):
    """Computes the mean pose from a trajectory of positions."""
    trajectory = np.array(trajectory)
    mean_position = np.mean(trajectory, axis=0)
    # Compute the mean direction vector
    directions = np.diff(trajectory, axis=0)  # we dont normalize as we want to weight each direction with the distance
    mean_direction = np.mean(directions, axis=0)
    mean_direction /= np.linalg.norm(mean_direction)
    # Set the up and right vectors of the mean pose
    up = np.array([0, 0, 1])
    right = np.cross(mean_direction, up)
    # Compute the up vector of the mean pose
    up = np.cross(right, mean_direction)  # need to recompute up because it might not be orthogonal
    # Normalize the right, and up vectors
    right /= np.linalg.norm(right)
    up /= np.linalg.norm(up)
    # Construct the transformation matrix
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = mean_direction
    pose[:3, 2] = up
    pose[:3, 3] = mean_position
    return pose


def _get_padded_poses(l2ws, padx: float, pady: float):
    """Pads the poses with four locations: padx forward/backward, pady left/right."""
    pad = torch.tensor([[-padx, 0.0, 0.0], [padx, 0.0, 0.0], [0.0, -pady, 0.0], [0.0, pady, 0.0]])
    pad = pad.unsqueeze(0).repeat(l2ws.shape[0], 1, 1)  # [N, 4, 3]
    pad = pad.reshape(-1, 3, 1)  # [N*4, 3, 1]
    l2ws = torch.repeat_interleave(l2ws, 4, dim=0)  # [N*4, 4, 4]
    pad_in_world = torch.matmul(l2ws[:, :3, :3], pad) + l2ws[:, :3, 3].unsqueeze(-1)  # [N*4, 3, 1]
    pad_in_world = pad_in_world.reshape(-1, 3)  # [N*4, 3]
    return pad_in_world


def _get_world_to_mean_transform(cameras: Cameras, lidars: Lidars):
    poses = lidars.lidar_to_worlds if len(lidars) else cameras.camera_to_worlds
    meta = lidars.metadata if len(lidars) else cameras.metadata
    assert meta and "sensor_idxs" in meta, "Must have sensor ids in metadata"
    sensor_idxs = meta["sensor_idxs"].squeeze(-1)
    selected_idx = sensor_idxs[0]
    select_poses = poses[sensor_idxs == selected_idx]
    select_trajectory = select_poses[:, :3, 3]
    if torch.std(select_trajectory, dim=0).max() < 1e-1:
        # If the trajectory is stationary, return the first pose
        m2w = to4x4(select_poses[0:1])[0]
    else:
        # Otherwise
        m2w = torch.from_numpy(_get_mean_pose_from_trajectory(select_trajectory).astype(np.float32))
    return torch.linalg.inv(m2w)[:3]


def _empty_cameras():
    return Cameras(
        camera_to_worlds=torch.zeros((0, 3, 4)),
        cx=torch.zeros((0)),
        cy=torch.zeros((0)),
        fx=torch.zeros((0)),
        fy=torch.zeros((0)),
        times=torch.zeros((0)),
        metadata={"sensor_idxs": torch.zeros((0))},
    )


def _empty_lidars():
    return Lidars(
        lidar_to_worlds=torch.zeros((0, 3, 4)),
        times=torch.zeros((0)),
        metadata={"sensor_idxs": torch.zeros((0))},
    )


def _filter_sensordata_on_time(
    data: SensorData, filepaths: List[Path], start_time: float, end_time: float
) -> Tuple[SensorData, List[Path]]:
    """Filter the data onbased on start and end time."""
    mask = (data.times >= start_time) & (data.times <= end_time)
    data = data[mask.squeeze(-1)]
    filepaths = [filepaths[i] for i in range(len(filepaths)) if mask[i]]
    return data, filepaths
