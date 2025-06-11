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
"""Data parser for NuScenes dataset"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Set, Tuple, Type, Union

import numpy as np
import pyquaternion
import torch
from nuscenes.nuscenes import NuScenes as NuScenesDatabase

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType, transform_points
from nerfstudio.data.dataparsers.ad_dataparser import (
    DUMMY_DISTANCE_VALUE,
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
    SplitTypes,
)
from nerfstudio.data.utils.lidar_elevation_mappings import VELODYNE_HDL32E_ELEVATION_MAPPING
from nerfstudio.utils import poses as pose_utils

MAX_RELECTANCE_VALUE = 255.0
LIDAR_FREQUENCY = 20.0  # Hz
LIDAR_CHANNELS = 32  # number of vertical channels
ALLOWED_RIGID_CLASSES = (
    "vehicle.car",
    "vehicle.bicycle",
    "vehicle.motorcycle",
    "vehicle.bus",
    "vehicle.bus",
    "vehicle.truck",
    "vehicle.trailer",
    "movable_object.pushable_pullable",
)
ALLOWED_DEFORMABLE_CLASSES = ("human.pedestrian",)

TRACKING_TO_GT_CLASSNAME_MAPPING = {
    "pedestrian": "human.pedestrian",
    "bicycle": "vehicle.bicycle",
    "motorcycle": "vehicle.motorcycle",
    "car": "vehicle.car",
    "bus": "vehicle.bus",
    "truck": "vehicle.truck",
    "trailer": "vehicle.truck",
}
# Nuscenes defines actor coordinate system as x-forward, y-left, z-up
# But we want to use x-right, y-forward, z-up
# So we need to rotate the actor coordinate system by 90 degrees around z-axis
WLH_TO_LWH = np.array(
    [
        [0, 1.0, 0, 0],
        [-1.0, 0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0],
    ]
)
HORIZONTAL_BEAM_DIVERGENCE = 0.00333333333  # radians, given as 4 inches at 100 feet
VERTICAL_BEAM_DIVERGENCE = 0.00166666666  # radians, given as 2 inches at 100 feet

NUSCENES_ELEVATION_MAPPING = {
    "LIDAR_TOP": VELODYNE_HDL32E_ELEVATION_MAPPING,
}
NUSCENES_AZIMUTH_RESOLUTION = {
    "LIDAR_TOP": 1 / 3.0,
}
NUSCENES_SKIP_ELEVATION_CHANNELS = {
    "LIDAR_TOP": (
        0,
        1,
    )
}
AVAILABLE_CAMERAS = (
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "BACK",
    "BACK_LEFT",
    "BACK_RIGHT",
)
CAMERA_TO_BOTTOM_RIGHT_CROP = {
    "FRONT": (0, 0),
    "FRONT_LEFT": (0, 0),
    "FRONT_RIGHT": (0, 0),
    "BACK": (80, 0),
    "BACK_LEFT": (0, 0),
    "BACK_RIGHT": (0, 0),
}
SEQ_CAMERA_TO_BOTTOM_RIGHT_CROP = defaultdict(lambda: CAMERA_TO_BOTTOM_RIGHT_CROP)
SEQ_CAMERA_TO_BOTTOM_RIGHT_CROP["scene-0164"] = {
    "FRONT": (0, 0),
    "FRONT_LEFT": (0, 0),
    "FRONT_RIGHT": (0, 0),
    "BACK": (80 + 66, 0),
    "BACK_LEFT": (0, 0),
    "BACK_RIGHT": (0, 0),
}

DEFAULT_IMAGE_HEIGHT = 900
DEFAULT_IMAGE_WIDTH = 1600


@dataclass
class NuScenesDataParserConfig(ADDataParserConfig):
    """NuScenes dataset config.
    NuScenes (https://www.nuscenes.org/nuscenes) is an autonomous driving dataset containing 1000 20s clips.
    Each clip was recorded with a suite of sensors including 6 surround cameras.
    It also includes 3D cuboid annotations around objects.
    We optionally use these cuboids to mask dynamic objects by specifying the mask_dir flag.
    To create these masks use nerfstudio/scripts/datasets/process_nuscenes_masks.py.
    """

    _target: Type = field(default_factory=lambda: NuScenes)
    """target class to instantiate"""
    data: Path = Path("data/nuscenes")
    """Directory specifying location of data."""
    sequence: str = "0103"
    """Name of the scene."""
    version: Literal["v1.0-mini", "v1.0-trainval"] = "v1.0-trainval"
    """Dataset version."""
    cameras: Tuple[
        Literal[
            "FRONT",
            "FRONT_LEFT",
            "FRONT_RIGHT",
            "BACK",
            "BACK_LEFT",
            "BACK_RIGHT",
            "none",
            "all",
        ],
        ...,
    ] = ("all",)
    """Which cameras to use."""
    lidars: Tuple[Literal["LIDAR_TOP", "none"], ...] = ("LIDAR_TOP",)
    """Which lidars to use. Currently only supports LIDAR_TOP"""
    verbose: bool = False
    """Load dataset with verbose messaging"""
    annotation_interval: float = 0.5
    """Length of time interval used to sample annotations from the dataset"""
    include_deformable_actors: bool = True
    """Include deformable actors in the dataset (NuScenes has many pedestrians so we default this to true)."""
    scene_box_height = (-10, 20)
    """The upper and lower height bounds for the scene box (m)."""
    train_eval_split_type: SplitTypes = SplitTypes.LINSPACE
    """Type of split to use for train/eval split."""
    lidar_elevation_mapping: Dict[str, Dict] = field(default_factory=lambda: NUSCENES_ELEVATION_MAPPING)
    """Elevation mapping for each lidar."""
    skip_elevation_channels: Dict[str, Tuple] = field(default_factory=lambda: NUSCENES_SKIP_ELEVATION_CHANNELS)
    """Channels to skip when adding missing points."""
    lidar_azimuth_resolution: Dict[str, float] = field(default_factory=lambda: NUSCENES_AZIMUTH_RESOLUTION)
    """Azimuth resolution for each lidar."""
    add_missing_points: bool = True
    """Add missing points to lidar point clouds."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if "scene" not in self.sequence:
            self.sequence = "scene-" + self.sequence


@dataclass
class NuScenes(ADDataParser):
    """NuScenes DatasetParser"""

    config: NuScenesDataParserConfig

    @property
    def actor_transform(self) -> torch.Tensor:
        """Nuscenes uses x-forward, so we need to rotate to x-right."""
        return torch.from_numpy(WLH_TO_LWH)

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        if "all" in self.config.cameras:
            self.config.cameras = AVAILABLE_CAMERAS
        filenames, times, intrinsics, poses, cam2egos, idxs = [], [], [], [], [], []
        heights, widths = [], []
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        is_key_frame = []
        for cam_idx, camera in enumerate(["CAM_" + camera for camera in self.config.cameras]):
            for sample_data in self._find_all_sample_data(first_sample["data"][camera]):
                calibrated_sensor_data = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
                ego_pose_data = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
                ego_pose = _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
                cam_pose = _rotation_translation_to_pose(
                    calibrated_sensor_data["rotation"], calibrated_sensor_data["translation"]
                )
                cam_pose[:3, :3] = cam_pose[:3, :3] @ OPENCV_TO_NERFSTUDIO
                pose = ego_pose @ cam_pose
                cam2egos.append(cam_pose)
                poses.append(pose)
                filenames.append(self.config.data / sample_data["filename"])
                intrinsics.append(calibrated_sensor_data["camera_intrinsic"])
                times.append(sample_data["timestamp"] / 1e6)
                idxs.append(cam_idx)
                heights.append(
                    DEFAULT_IMAGE_HEIGHT - SEQ_CAMERA_TO_BOTTOM_RIGHT_CROP[self.config.sequence][camera[4:]][0]
                )  # :4 to remove CAM_
                widths.append(
                    DEFAULT_IMAGE_WIDTH - SEQ_CAMERA_TO_BOTTOM_RIGHT_CROP[self.config.sequence][camera[4:]][1]
                )  # :4 to remove CAM_
                is_key_frame.append(sample_data["is_key_frame"])

        # To tensors
        intrinsics = torch.tensor(np.array(intrinsics), dtype=torch.float32)
        poses = torch.tensor(np.array(poses), dtype=torch.float32)
        cam2egos = torch.tensor(np.array(cam2egos), dtype=torch.float32)
        times = torch.tensor(times, dtype=torch.float64)
        idxs = torch.tensor(idxs).int().unsqueeze(-1)
        heights = torch.tensor(heights).int()
        widths = torch.tensor(widths).int()
        is_key_frame = torch.tensor(is_key_frame).reshape(-1, 1).bool()
        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=heights,
            width=widths,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={"sensor_idxs": idxs, "is_key_frame": is_key_frame},
        )
        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        lidar_filenames, times, poses, lid2egos, idxs = [], [], [], [], []
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        is_key_frame = []
        for lidar_idx, lidar in enumerate(self.config.lidars):
            for lidar_data in self._find_all_sample_data(first_sample["data"][lidar]):
                calibrated_sensor_data = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
                ego_pose_data = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
                ego_pose = _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
                lidar_pose = _rotation_translation_to_pose(
                    calibrated_sensor_data["rotation"], calibrated_sensor_data["translation"]
                )
                pose = ego_pose @ lidar_pose

                lidar_filenames.append(self.config.data / lidar_data["filename"])

                poses.append(pose)
                lid2egos.append(lidar_pose)
                times.append(lidar_data["timestamp"] / 1e6)
                idxs.append(lidar_idx)
                is_key_frame.append(lidar_data["is_key_frame"])

        poses = torch.tensor(np.array(poses), dtype=torch.float64)  # will be changed to float32 later
        lid2egos = torch.tensor(np.array(lid2egos), dtype=torch.float32)
        times = torch.tensor(times, dtype=torch.float64)  # need higher precision
        idxs = torch.tensor(idxs).int().unsqueeze(-1)

        is_key_frame = torch.tensor(is_key_frame).reshape(-1, 1).bool()
        lidars = Lidars(
            lidar_to_worlds=poses[:, :3, :4],
            lidar_type=LidarType.VELODYNE_HDL32E,
            assume_ego_compensated=True,
            times=times,
            metadata={"sensor_idxs": idxs, "is_key_frame": is_key_frame},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
            valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
        )
        return lidars, lidar_filenames

    def _read_lidars(self, lidars: Lidars, filepaths: List[Path]) -> List[torch.Tensor]:
        point_clouds = []
        for filepath in filepaths:
            pc = np.fromfile(str(filepath), dtype=np.float32).reshape([-1, 5])
            pc[..., 3] = pc[..., 3] / MAX_RELECTANCE_VALUE  # normalize reflectance
            # nuscenes lidar time is the time of the end of the sweep. Add estimated times per point
            offsets = np.repeat(np.linspace(-1 / LIDAR_FREQUENCY, 0, pc.shape[0] // LIDAR_CHANNELS), LIDAR_CHANNELS)
            # warning: here we overwrite the beam index channel with time offsets, since we assume x,y,z,r,t format
            channel_id = pc[..., 4].astype(np.int32)
            pc[..., 4] = offsets
            # concatenate pc with channel id
            pc = np.concatenate([pc, channel_id[:, None]], axis=-1).astype(np.float32)
            point_clouds.append(torch.from_numpy(pc))

        if self.config.add_missing_points:
            # remove ego motion compensation
            poses = lidars.lidar_to_worlds
            times = lidars.times.squeeze(-1)
            missing_points = []
            for point_cloud, l2w, time in zip(point_clouds, poses, times):
                # make sure we have double for avoid overflow when adding time
                pc = point_cloud.clone().double()
                # absolute time
                pc[:, 4] = pc[:, 4] + time
                # project to world frame
                pc[..., :3] = transform_points(pc[..., :3], l2w.unsqueeze(0).to(pc))
                # remove ego motion compensation
                pc, interpolated_poses = self._remove_ego_motion_compensation(pc, poses, times)
                # reset time
                pc[:, 4] = point_cloud[:, 4].clone()
                # transform to common lidar frame again
                interpolated_poses = torch.matmul(
                    pose_utils.inverse(l2w.unsqueeze(0)).float(), pose_utils.to4x4(interpolated_poses).float()
                )
                # move channel from index 5 to 3
                pc = pc[..., [0, 1, 2, 5, 3, 4]]
                # get missing points
                miss_pc = self._get_missing_points(pc, interpolated_poses, "LIDAR_TOP", dist_cutoff=0.05)
                # move channel from index 3 to 5
                miss_pc = miss_pc[..., [0, 1, 2, 4, 5, 3]]
                # add missing points
                missing_points.append(miss_pc.float())

            # add missing points to point clouds
            point_clouds = [torch.cat([pc, missing], dim=0) for pc, missing in zip(point_clouds, missing_points)]

        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        return point_clouds

    def _get_actor_trajectories(self) -> List[Dict]:
        trajs = defaultdict(list)
        curr_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        while True:
            for box_token in curr_sample["anns"]:
                box = self.nusc.get_box(box_token)
                pose = np.eye(4)
                pose[:3, :3] = box.orientation.rotation_matrix
                pose[:3, 3] = np.array(box.center)
                pose = pose @ WLH_TO_LWH
                instance_token = self.nusc.get("sample_annotation", box.token)["instance_token"]
                trajs[instance_token].append(
                    {
                        "pose": pose,
                        "wlh": np.array(box.wlh),
                        "label": box.name,
                        "time": curr_sample["timestamp"] / 1e6,
                    }
                )
            if curr_sample["next"]:
                curr_sample = self.nusc.get("sample", curr_sample["next"])
            else:
                break
        return self._traj_dict_to_list(trajs)

    def _generate_dataparser_outputs(self, split="train"):
        self.nusc = NuScenesDatabase(
            version=self.config.version,
            dataroot=str(self.config.data.absolute()),
            verbose=self.config.verbose,
        )
        self.scene = self.nusc.get("scene", self.nusc.field2token("scene", "name", str(self.config.sequence))[0])
        out = super()._generate_dataparser_outputs(split)
        del self.nusc
        del self.scene
        return out

    def _find_all_sample_data(self, sample_data_token: str):
        """Finds all sample data from a given sample token."""
        curr_token = sample_data_token
        sd = self.nusc.get("sample_data", curr_token)
        # Rewind to first sample data
        while sd["prev"]:
            curr_token = sd["prev"]
            sd = self.nusc.get("sample_data", curr_token)
        # Forward to last sample data
        all_sample_data = [sd]
        while sd["next"]:
            curr_token = sd["next"]
            sd = self.nusc.get("sample_data", curr_token)
            all_sample_data.append(sd)
        return all_sample_data

    def _traj_dict_to_list(self, traj: dict) -> list:
        """Convert a dictionary of lists with trajectories to a list of dictionaries with trajectories"""
        allowed_classes: Set[str] = set(ALLOWED_RIGID_CLASSES)
        if self.config.include_deformable_actors:
            allowed_classes.update(ALLOWED_DEFORMABLE_CLASSES)
        traj_out = []
        for instance_token, traj_list in traj.items():
            poses = torch.from_numpy(np.stack([t["pose"] for t in traj_list]).astype(np.float32))
            times = torch.from_numpy(np.array([t["time"] for t in traj_list]))
            dims = torch.from_numpy(np.array([t["wlh"] for t in traj_list]).astype(np.float32))
            dims = dims.max(0).values  # take max dimensions (important for deformable objects)
            dynamic = (poses[:, :2, 3].std(dim=0) > 0.50).any()
            stationary = not dynamic  # TODO: maybe make this stricter
            if stationary or not _is_label_allowed(traj_list[0]["label"], allowed_classes):
                continue
            traj_dict = {
                "uuid": instance_token,
                "label": traj_list[0]["label"],
                "poses": poses,
                "timestamps": times,
                "dims": dims,
                "stationary": stationary,
                "symmetric": "human" not in traj_list[0]["label"],
                "deformable": "human" in traj_list[0]["label"],
            }
            traj_out.append(traj_dict)
        return traj_out

    def _get_nuscenes_sample_indices(self, is_key_frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        train_mask = ~is_key_frame
        eval_mask = is_key_frame
        return train_mask.nonzero(as_tuple=True)[0], eval_mask.nonzero(as_tuple=True)[0]

    def _get_train_eval_indices(self, sensors: Union[Cameras, Lidars]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.train_eval_split_type == SplitTypes.LINSPACE:
            return self._get_linspaced_indices(sensors.metadata["sensor_idxs"].squeeze(-1))
        elif self.config.train_eval_split_type == SplitTypes.NUSCENES_SAMPLES:
            is_key_frame = sensors.metadata["is_key_frame"]
            del sensors.metadata["is_key_frame"]
            return self._get_nuscenes_sample_indices(is_key_frame)
        else:
            raise ValueError(f"Unknown split type {self.config.train_eval_split_type}")


def _rotation_translation_to_pose(r_quat, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""

    pose = np.eye(4)

    # NB: Nuscenes recommends pyquaternion, which uses scalar-first format (w x y z)
    # https://github.com/nutonomy/nuscenes-devkit/issues/545#issuecomment-766509242
    # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L299
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    pose[:3, :3] = pyquaternion.Quaternion(r_quat).rotation_matrix

    pose[:3, 3] = t_vec
    return pose


def _is_label_allowed(label: str, allowed_classes: Set[str]) -> bool:
    """Check if label is allowed, on all possible hierarchies."""
    split_label = label.split(".")
    for i in range(len(split_label)):
        if ".".join(split_label[: i + 1]) in allowed_classes:
            return True
    return False
