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

"""Data parser for the ZOD dataset"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
from pyquaternion import Quaternion
from typing_extensions import Literal
from zod import Anonymization, Camera as ZodCamera, Lidar as ZodLidar, ZodSequences
from zod.constants import EGO as ZOD_EGO
from zod.data_classes.box import Box3D
from zod.data_classes.sensor import LidarData

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType, transform_points
from nerfstudio.data.dataparsers.ad_dataparser import OPENCV_TO_NERFSTUDIO, ADDataParser, ADDataParserConfig
from nerfstudio.data.dataparsers.nuscenes_dataparser import WLH_TO_LWH
from nerfstudio.data.utils.lidar_elevation_mappings import VELODYNE_128_ELEVATION_MAPPING
from nerfstudio.utils import poses as pose_utils

TOP = "top"
LEFT = "left"
RIGHT = "right"


ZOD_ELEVATION_MAPPING = {TOP: VELODYNE_128_ELEVATION_MAPPING}
ZOD_LIDAR_IGNORE_REGIONS = {TOP: [[-34.0, -5.0, -30.0, -1.0], [-158.0, -134.0, -30.0, -1.0]]}
ZOD_SKIP_ELEVATION_CHANNELS = {
    TOP: (
        36,
        69,
        54,
        87,
        0,
        97,
        18,
        115,
        44,
        77,
        62,
        95,
    )
}  # these are channel indices that correspond to a low elevation angle, as per the VLS128 manual.

HORIZONTAL_BEAM_DIVERGENCE = 3.0e-3  # radians, or meters at a distance of 1m
VERTICAL_BEAM_DIVERGENCE = 1.5e-3  # radians, or meters at a distance of 1m
HOOD_HEIGHT = 750  # px
MAX_INTENSITY_VALUE = 255

ALLOWED_CATEGORIES = {
    "Vehicle",
    "LargeVehicle",
    "Motorcyclist",
    "Bicyclist",
    "Trailer",
}
SYMMETRIC_CATEGORIES = ALLOWED_CATEGORIES

DEFORMABLE_CATEGORIES = {
    "Pedestrian",
}

AVAILABLE_CAMERAS = ("front",)
AVAILABLE_LIDARS = (
    TOP,
    LEFT,
    RIGHT,
)
ZOD_LIDAR_TO_INDEX = {
    TOP: 0,
    LEFT: 1,
    RIGHT: 2,
}

ZOD_INDEX_TO_LIDAR = {v: k for k, v in ZOD_LIDAR_TO_INDEX.items()}


ZOD_AZIMUTH_RESOLUTION = {
    TOP: 0.2,
    LEFT: 0.2,
    RIGHT: 0.2,
}

ZOD_CHANNEL_MAPPING = {
    TOP: (1, 128),
    LEFT: (129, 144),
    RIGHT: (145, 160),
}

ZOD_LIDAR_NAME_TO_LIDAR_TYPE = {
    TOP: LidarType.VELODYNE128,
    LEFT: LidarType.VELODYNE16,
    RIGHT: LidarType.VELODYNE16,
}

LANE_SHIFT_SIGN: Dict[str, Literal[-1, 1]] = defaultdict(lambda: -1)
LANE_SHIFT_SIGN.update(
    {
        "000784": -1,
        "000005": 1,
        "000030": -1,
        "000221": -1,
        "000231": 1,
        "000387": -1,
        "001186": -1,
        "000657": -1,
        "000581": -1,
        "000619": 1,
        "000546": -1,
        "000244": 1,
        "000811": -1,
    }
)


@dataclass
class ZodDataParserConfig(ADDataParserConfig):
    """ZOD dataset config.
    ZOD (https://zod.zenseact.com) is an autonomous driving dataset containing:
    - 100k frames.
    - ~1500 20 second sequences.
    - ~30 multi-minute drives.

    Each clip was recorded with a suite of sensors including a wide angle camera, and a LiDAR.
    It also includes 3D cuboid annotations around objects.
    We optionally use these cuboids to mask dynamic objects by specifying the mask_dir flag.
    To create these masks, install zod and run `zod generate object-masks`.
    """

    _target: Type = field(default_factory=lambda: Zod)
    """target class to instantiate"""
    sequence: str = "000581"
    """Name of the scene."""
    data: Path = Path("data/zod")
    """Path to ZOD dataset."""
    subset: Literal["sequences"] = "sequences"  # only sequences supported currently
    """Dataset subset."""
    version: Literal["mini", "full"] = "full"
    """Dataset version."""
    cameras: Tuple[Literal["front", "none", "all"], ...] = ("front",)
    """Which cameras to use."""
    lidars: Tuple[Literal["top", "left", "right", "all", "none"], ...] = (
        "top",
    )  # we have not implemented the correct transforms for left and right lidars
    """Which lidars to use."""
    min_lidar_dist: Tuple[float, float, float] = (1.5, 3.0, 1.5)
    """Remove all points within this distance from lidar sensor."""
    anonymization: Anonymization = Anonymization.BLUR
    """Which cameras to use."""
    annotation_interval: float = 0.1  # auto annotation interval
    """Interval between annotations in seconds."""
    auto_annotation_dir: Union[
        Literal["default"], Path
    ] = "default"  # if None, use default, which is <config.data>/auto_annotations/<config.subset>/<config.sequence>.json
    """Directory to load auto annotations from."""
    add_missing_points: bool = True
    """Whether to add missing points to the point cloud."""
    lidar_elevation_mapping: Dict[str, Dict[int, float]] = field(default_factory=lambda: ZOD_ELEVATION_MAPPING)
    """Elevation mapping for each lidar."""
    lidar_azimuth_resolution: Dict[str, float] = field(default_factory=lambda: ZOD_AZIMUTH_RESOLUTION)
    """Azimuth resolution for each lidar."""
    skip_elevation_channels: Dict[str, Tuple] = field(default_factory=lambda: ZOD_SKIP_ELEVATION_CHANNELS)
    """Channels to skip when adding missing points."""


@dataclass
class Zod(ADDataParser):
    """Zenseact Open Dataset DatasetParser"""

    config: ZodDataParserConfig

    @property
    def actor_transform(self) -> torch.Tensor:
        """ZOD uses x-forward, so we need to rotate to x-right."""
        return torch.from_numpy(WLH_TO_LWH)

    def _get_lane_shift_sign(self, sequence: str) -> Literal[-1, 1]:
        return LANE_SHIFT_SIGN.get(sequence, 1)

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        """Returns camera info and image filenames."""

        if "all" in self.config.cameras:
            self.config.cameras = AVAILABLE_CAMERAS

        filenames, times, intrinsics, poses, idxs, distortions = [], [], [], [], [], []
        heights, widths = [], []

        for camera_idx, cam_name in enumerate(self.config.cameras):
            zod_cam = getattr(ZodCamera, cam_name.upper())
            cam_frames = self.zod_seq.info.get_camera_frames(anonymization=self.config.anonymization, camera=zod_cam)
            for frame in cam_frames:
                timestamp = frame.time.timestamp()

                filenames.append(frame.filepath)
                times.append(timestamp)

                intrinsics.append(self.zod_seq.calibration.cameras[zod_cam].intrinsics)
                ego_pose = self.zod_seq.oxts.get_poses(timestamp)
                cam_pose = self.zod_seq.calibration.cameras[zod_cam].extrinsics.transform.copy()
                # Convert from OpenCV to NerfStudio coordinate system
                cam_pose[:3, :3] = cam_pose[:3, :3] @ OPENCV_TO_NERFSTUDIO
                pose = ego_pose @ cam_pose
                poses.append(pose)
                idxs.append(camera_idx)
                width, height = self.zod_seq.calibration.cameras[zod_cam].image_dimensions
                distortion = self.zod_seq.calibration.cameras[zod_cam].distortion
                # we only have the radial distortion parameters, so we pad with zeros
                distortion = np.concatenate([distortion, np.zeros(2)])
                distortions.append(distortion)
                heights.append(height - HOOD_HEIGHT)
                widths.append(width)

        # To tensors
        intrinsics = torch.from_numpy(np.array(intrinsics)).float()
        distortions = torch.from_numpy(np.array(distortions)).float()
        poses = torch.tensor(np.array(poses), dtype=torch.float32)
        times = torch.tensor(times, dtype=torch.float64)  # need higher precision
        idxs = torch.tensor(idxs).int().unsqueeze(-1)
        heights = torch.tensor(heights).int()
        widths = torch.tensor(widths).int()

        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=heights,
            width=widths,
            distortion_params=distortions,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.FISHEYE,
            times=times,
            metadata={"sensor_idxs": idxs},
        )
        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        """Returns lidar info and loaded point clouds."""
        if "all" in self.config.lidars:
            self.config.lidars = AVAILABLE_LIDARS

        if self.config.add_missing_points and self.config.lidars != (TOP,):
            raise NotImplementedError(
                "Adding missing points is only implemented for the top lidar. "
                "If you want to add missing points for other lidars, please implement it and submit a PR."
            )

        times, poses, idxs, lidar_filenames = [], [], [], []
        lidar_frames = self.zod_seq.info.get_lidar_frames(lidar=ZodLidar.VELODYNE)
        for frame in lidar_frames:
            timestamp = frame.time.timestamp()
            ego_pose = self.zod_seq.oxts.get_poses(timestamp)
            lidar_pose = self.zod_seq.calibration.lidars[ZodLidar.VELODYNE].extrinsics.transform.copy()
            pose = ego_pose @ lidar_pose
            # TODO: zod does not provide the correct extrinsics for the left and right lidars
            # we should move them to their correct origins.
            for lidar_name in self.config.lidars:
                poses.append(pose)
                idxs.append(ZOD_LIDAR_TO_INDEX[lidar_name])
                lidar_filenames.append(Path(frame.filepath))
                times.append(timestamp)

        # To tensors
        poses = torch.tensor(np.array(poses), dtype=torch.float64)  # will be converted to float32 later
        times = torch.tensor(times, dtype=torch.float64)
        idxs = torch.tensor(idxs).int().unsqueeze(-1)

        lidars = Lidars(
            lidar_to_worlds=poses[:, :3, :4],
            lidar_type=LidarType.VELODYNE128,
            times=times,
            metadata={"sensor_idxs": idxs},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
        )

        return lidars, lidar_filenames

    def _read_lidars(self, lidars: Lidars, filepaths: List[Path]) -> List[torch.Tensor]:
        point_clouds = []

        @lru_cache(maxsize=1)
        def read_pc(fp):
            pc = LidarData.from_npy(str(fp))
            return pc

        for i, fp in enumerate(filepaths):
            lidar_idx = lidars.metadata["sensor_idxs"][i].item()
            lidar_type = ZOD_INDEX_TO_LIDAR[lidar_idx]
            min_channel, max_channel = ZOD_CHANNEL_MAPPING[lidar_type]

            pc = read_pc(fp)
            valid_channels = np.logical_and(min_channel <= pc.diode_idx, pc.diode_idx <= max_channel)
            xyz = pc.points[valid_channels]  # N x 3
            intensity = pc.intensity[valid_channels] / MAX_INTENSITY_VALUE  # N,
            t = pc.timestamps[valid_channels] - pc.core_timestamp  # N, relative timestamps
            diode_idx = pc.diode_idx[valid_channels]  # N,
            diode_idx -= min_channel  # make diode_idx start at 0
            pc = np.hstack((xyz, intensity[:, None], t[:, None], diode_idx[:, None]))
            point_clouds.append(torch.from_numpy(pc).float())

        times = lidars.times
        poses = lidars.lidar_to_worlds

        if self.config.add_missing_points:
            assert self.config.lidars == (TOP,), "Only top lidar supported for missing points"
            # add missing points
            missing_points = []
            ego_times = torch.from_numpy(self.zod_seq.oxts.timestamps)
            times_close_to_lidar = (ego_times > (times.min() - 0.1)) & (ego_times < (times.max() + 0.1))
            ego_times = ego_times[times_close_to_lidar]
            ego_poses = torch.from_numpy(self.zod_seq.oxts.poses)[times_close_to_lidar]
            lidar2ego = torch.from_numpy(self.zod_seq.calibration.lidars[ZodLidar.VELODYNE].extrinsics.transform.copy())
            oxts_lidar_poses = ego_poses @ lidar2ego.unsqueeze(0)
            for point_cloud, l2w, time in zip(point_clouds, poses, times):
                pc = point_cloud.clone().to(torch.float64)
                # absolute time
                pc[:, 4] = pc[:, 4] + time
                # project to world frame
                pc[..., :3] = transform_points(pc[..., :3], l2w.unsqueeze(0).to(pc))
                # remove ego motion compensation and move to sensor frame
                pc, interpolated_poses = self._remove_ego_motion_compensation(pc, oxts_lidar_poses, ego_times)
                # reset time
                pc[:, 4] = point_cloud[:, 4].clone()
                # transform to common lidar frame again
                interpolated_poses = torch.matmul(
                    pose_utils.inverse(l2w.unsqueeze(0)).float(), pose_utils.to4x4(interpolated_poses).float()
                )
                # move channel from index 5 to 3
                pc = pc[..., [0, 1, 2, 5, 3, 4]]
                # add missing points
                mp = self._get_missing_points(
                    pc, interpolated_poses, TOP, dist_cutoff=0.0, ignore_regions=ZOD_LIDAR_IGNORE_REGIONS[TOP]
                ).float()
                # move channel from index 3 to 5
                mp = mp[..., [0, 1, 2, 4, 5, 3]]
                missing_points.append(mp)
            # add missing points to point clouds
            point_clouds = [torch.cat([pc, missing], dim=0) for pc, missing in zip(point_clouds, missing_points)]

        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        return point_clouds

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
        anno_dir = self.config.data / "auto_annotations" / self.config.subset / (self.config.sequence + ".json")
        anno_dir = anno_dir if self.config.auto_annotation_dir == "default" else self.config.auto_annotation_dir

        with open(str(anno_dir), "r") as f:
            annos = json.load(f)

        allowed_cats = ALLOWED_CATEGORIES
        if self.config.include_deformable_actors:
            allowed_cats.union(DEFORMABLE_CATEGORIES)

        trajs = []
        # loop over trajectories
        for boxes in annos.values():
            # they should be alive at least two frames
            if len(boxes) < 2:
                continue
            label = boxes[0]["name"]

            if label not in allowed_cats:
                continue

            poses = []
            timestamps = []
            wlh = []
            for box in boxes:
                bbox = Box3D(
                    center=np.array(box["center"]),
                    size=np.array(box["size"]),
                    orientation=Quaternion(box["orientation"]),
                    frame=ZodLidar.VELODYNE,
                )
                bbox.convert_to(ZOD_EGO, self.zod_seq.calibration)
                pose = np.eye(4)
                pose[:3, :3] = bbox.orientation.rotation_matrix
                pose[:3, 3] = bbox.center
                # TODO, do we need to do lwh to wlh here?
                pose = pose @ WLH_TO_LWH
                ego_pose = self.zod_seq.oxts.get_poses(box["timestamp"])
                pose = ego_pose @ pose
                poses.append(pose)
                timestamps.append(box["timestamp"])
                l, w, h = bbox.size  # noqa: E741
                wlh.append(np.array([w, l, h]))

            poses = np.array(poses)
            # dynamic if we move more that 1m in any direction
            dynamic = np.any(np.std(poses[:, :3, 3], axis=0) > 1.0)
            # we skip all stationary objects
            if not dynamic:
                continue

            symmetric = label in SYMMETRIC_CATEGORIES
            deformable = label in DEFORMABLE_CATEGORIES

            trajs.append(
                {
                    "poses": torch.tensor(poses, dtype=torch.float32),
                    "timestamps": torch.tensor(timestamps, dtype=torch.float64),
                    "dims": torch.tensor(np.median(wlh, axis=0), dtype=torch.float32),
                    "label": label,
                    "stationary": not dynamic,
                    "symmetric": symmetric,
                    "deformable": deformable,
                }
            )

        return trajs

    def _generate_dataparser_outputs(self, split="train"):
        self.zod = ZodSequences(dataset_root=str(self.config.data), version=self.config.version, mp=False)
        assert self.config.sequence in self.zod.get_all_ids(), f"Sequence {self.config.sequence} not found in dataset"

        self.zod_seq = self.zod[self.config.sequence]

        out = super()._generate_dataparser_outputs(split=split)
        del self.zod
        del self.zod_seq
        return out
