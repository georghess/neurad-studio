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
"""Data parser for PandaSet dataset"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import pandas as pd
import pyquaternion
import torch
import yaml
from pandaset import DataSet

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import (
    DUMMY_DISTANCE_VALUE,
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.lidar_elevation_mappings import PANDAR64_ELEVATION_MAPPING
from nerfstudio.utils import poses as pose_utils

PANDASET_ELEVATION_MAPPING = {"Pandar64": PANDAR64_ELEVATION_MAPPING}

LIDAR_NAME_TO_INDEX = {
    "Pandar64": 0,
    "PandarGT": 1,
}

PANDASET_SEQ_LEN = 80
EXTRINSICS_FILE_PATH = os.path.join(os.path.dirname(__file__), "pandaset_extrinsics.yaml")
MAX_RELECTANCE_VALUE = 255.0
BACK_CAMERA_BOTTOM_CROP = 260
ALLOWED_RIGID_CLASSES = (
    "Car",
    "Pickup Truck",
    "Medium-sized Truck",
    "Semi-truck",
    "Towed Object",
    "Motorcycle",
    "Other Vehicle - Construction Vehicle",
    "Other Vehicle - Uncommon",
    "Other Vehicle - Pedicab",
    "Emergency Vehicle",
    "Bus",
    "Personal Mobility Device",
    "Motorized Scooter",
    "Bicycle",
    "Train",
    "Trolley",
    "Tram / Subway",
)
ALLOWED_DEFORMABLE_CLASSES = (
    "Pedestrian",
    "Pedestrian with Object",
)

LANE_SHIFT_SIGN: Dict[str, Literal[-1, 1]] = defaultdict(lambda: -1)
LANE_SHIFT_SIGN.update(
    {
        "001": -1,
        "011": 1,
        "016": 1,
        "028": -1,
        "053": 1,
        "063": -1,
        "084": -1,
        "106": -1,
        "123": -1,
        "158": -1,
    }
)

PANDASET_AZIMUTH_RESOLUTION = {"Pandar64": 0.2}
PANDASET_SKIP_ELEVATION_CHANNELS = {
    "Pandar64": (
        62,
        63,
    )
}

HORIZONTAL_BEAM_DIVERGENCE = 3e-3  # radians
VERTICAL_BEAM_DIVERGENCE = 1.5e-3  # radians

AVAILABLE_CAMERAS = ("front", "front_left", "front_right", "back", "left", "right")


@dataclass
class PandaSetDataParserConfig(ADDataParserConfig):
    """PandaSet dataset config.
    PandaSet (https://pandaset.org/) is an autonomous driving dataset containing 100+ 8s clips.
    Each clip was recorded with a suite of sensors including 6 surround cameras and two lidars.
    It also includes 3D cuboid annotations around objects.
    """

    _target: Type = field(default_factory=lambda: PandaSet)
    """target class to instantiate"""
    data: Path = Path("data/pandaset")
    """Directory specifying location of data."""
    sequence: str = "001"
    """Name of the scene."""
    cameras: Tuple[Literal["front", "front_left", "front_right", "back", "left", "right", "none", "all"], ...] = (
        "front",
        "front_left",
        "front_right",
        "back",
        "left",
        "right",
    )
    """Which cameras to use."""
    lidars: Tuple[Literal["Pandar64", "PandarGT", "none"], ...] = ("Pandar64",)
    """Which lidars to use."""
    annotation_interval: float = 0.1
    """Interval between annotations in seconds."""
    correct_cuboid_time: bool = True
    """Whether to correct the cuboid time to match the actual time of observation, not the end of the lidar sweep."""
    min_lidar_dist: Tuple[float, float, float] = (1.0, 2.0, 2.0)
    """Pandaset lidar is x-right, y-down, z-forward."""
    lidar_elevation_mapping: Dict[str, Dict] = field(default_factory=lambda: PANDASET_ELEVATION_MAPPING)
    """Elevation mapping for each lidar."""
    skip_elevation_channels: Dict[str, Tuple] = field(default_factory=lambda: PANDASET_SKIP_ELEVATION_CHANNELS)
    """Channels to skip when adding missing points."""
    lidar_azimuth_resolution: Dict[str, float] = field(default_factory=lambda: PANDASET_AZIMUTH_RESOLUTION)
    """Azimuth resolution for each lidar."""
    rolling_shutter_time: float = 0.03
    """The rolling shutter time for the cameras (seconds)."""
    time_to_center_pixel: float = -0.01
    """In pandaset the image time seems to line up with the final row."""


@dataclass
class PandaSet(ADDataParser):
    """PandaSet DatasetParser"""

    config: PandaSetDataParserConfig

    def _get_lane_shift_sign(self, sequence: str) -> Literal[-1, 1]:
        return LANE_SHIFT_SIGN.get(sequence, 1)

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        """Returns camera info and image filenames."""
        if "all" in self.config.cameras:
            self.config.cameras = AVAILABLE_CAMERAS
        cameras = [cam + "_camera" for cam in self.config.cameras]
        # get image filenames and camera data
        image_filenames = []
        times = []
        intrinsics = []
        poses = []
        idxs = []
        heights = []
        for i in range(PANDASET_SEQ_LEN):
            for camera in cameras:
                curr_cam = self.sequence.camera[camera]
                file_path = curr_cam._data_structure[i]
                pose = _pandaset_pose_to_matrix(curr_cam.poses[i])
                pose[:3, :3] = pose[:3, :3] @ OPENCV_TO_NERFSTUDIO
                intrinsic_ = curr_cam.intrinsics
                intrinsic = np.array(
                    [
                        [intrinsic_.fx, 0, intrinsic_.cx],
                        [0, intrinsic_.fy, intrinsic_.cy],
                        [0, 0, 1],
                    ]
                )
                image_filenames.append(file_path)
                intrinsics.append(intrinsic)
                poses.append(pose)
                times.append(curr_cam.timestamps[i])
                idxs.append(cameras.index(camera))
                heights.append(1080 - (BACK_CAMERA_BOTTOM_CROP if camera == "back_camera" else 0))

        intrinsics = torch.tensor(np.array(intrinsics), dtype=torch.float32)
        poses = torch.tensor(np.array(poses), dtype=torch.float32)
        times = torch.tensor(times, dtype=torch.float64)  # need higher precision
        idxs = torch.tensor(idxs).int().unsqueeze(-1)
        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=torch.tensor(heights),
            width=1920,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={"sensor_idxs": idxs},
        )
        return cameras, image_filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        """Returns lidar info and loaded point clouds."""
        poses = []
        times = []
        idxs = []
        lidar_filenames = []
        for i in range(PANDASET_SEQ_LEN):
            # the pose information in self.sequence.lidar.poses is not correct, so we compute it from the camera pose and extrinsics
            # the lidar scans are synced such that the middle of a scan is at the same time as the front camera image
            front_cam = self.sequence.camera["front_camera"]
            front_cam2w = _pandaset_pose_to_matrix(front_cam.poses[i])
            front_cam_extrinsics = self.extrinsics["front_camera"]
            front_cam_extrinsics["position"] = front_cam_extrinsics["extrinsic"]["transform"]["translation"]
            front_cam_extrinsics["heading"] = front_cam_extrinsics["extrinsic"]["transform"]["rotation"]
            l2front_cam = _pandaset_pose_to_matrix(front_cam_extrinsics)
            l2w = torch.from_numpy(front_cam2w @ l2front_cam)

            # Load point cloud
            filename = self.sequence.lidar._data_structure[i]
            # since we are using the front camera pose, we need to use the front camera timestamp
            time = front_cam.timestamps[i]

            for lidar in self.config.lidars:
                lidar_idx = LIDAR_NAME_TO_INDEX[lidar]
                lidar_filenames.append(Path(filename))
                poses.append(l2w)
                idxs.append(lidar_idx)
                times.append(time)

        poses = torch.stack(poses)
        times = torch.tensor(times, dtype=torch.float64)  # need higher precision
        idxs = torch.tensor(idxs).int().unsqueeze(-1)

        lidars = Lidars(
            lidar_to_worlds=poses[:, :3, :4],
            lidar_type=LidarType.PANDAR64,
            times=times,
            metadata={"sensor_idxs": idxs},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
            valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
        )
        return lidars, lidar_filenames

    def _read_lidars(self, lidars: Lidars, filepaths: List[Path]) -> List[torch.Tensor]:
        point_clouds = []
        point_clouds_in_world = []

        @lru_cache(maxsize=1)
        def read_point_cloud(filepath):
            return pd.read_pickle(filepath).values

        for i, filename in enumerate(filepaths):
            lidar = lidars[i]
            lidar_idx = lidars.metadata["sensor_idxs"][i]
            l2w = pose_utils.to4x4(lidar.lidar_to_worlds)

            # Load point cloud
            point_cloud = torch.from_numpy(read_point_cloud(filename))
            point_cloud[:, 3] /= MAX_RELECTANCE_VALUE
            if lidar_idx == LIDAR_NAME_TO_INDEX["Pandar64"]:
                point_clouds_in_world.append(point_cloud[point_cloud[:, -1] == LIDAR_NAME_TO_INDEX["Pandar64"], :-1])
            points = point_cloud[:, :3]
            # transform points from world space to sensor space
            points = torch.hstack((points, torch.ones((points.shape[0], 1))))
            points = (torch.matmul(torch.linalg.inv(l2w), points.T).T)[:, :3]
            point_cloud[:, :3] = points

            # and adjust the point cloud timestamps accordingly
            point_cloud[:, 4] -= lidar.times

            pc = point_cloud[point_cloud[:, -1] == lidar_idx, :-1]
            point_clouds.append(pc.float())

        assert len(point_clouds) == len(
            lidars
        ), f"Number of point clouds ({len(point_clouds)}) does not match number of lidars ({len(lidars)})"

        if self.config.add_missing_points:
            poses = lidars.lidar_to_worlds
            times = lidars.times.squeeze(-1)

            pc_without_ego_motion_comp = [
                self._remove_ego_motion_compensation(pc, poses, times) for pc in point_clouds_in_world
            ]
            pc_without_ego_motion_comp = [
                (self._add_channel_info(pc, dim=3, lidar_name="Pandar64"), pose)
                for pc, pose in pc_without_ego_motion_comp
            ]  # TODO: clean up to handle multiple lidars

            # project back to joint lidar pose
            pc_without_ego_motion_comp = [
                (pc, torch.matmul(pose_utils.inverse(l2w.unsqueeze(0)).float(), pose_utils.to4x4(pc_poses).float()))
                for (pc, pc_poses), l2w in zip(pc_without_ego_motion_comp, poses)
            ]
            missing_points = [
                self._get_missing_points(pc, poses, "Pandar64") for pc, poses in pc_without_ego_motion_comp
            ]
            # drop channel info again
            missing_points = [torch.cat([pc[:, :3], pc[:, 4:]], dim=-1) for pc in missing_points]
            # subtracts lidar time
            for pc, time in zip(missing_points, times):
                pc[:, 4] -= time
            # add missing points to point clouds
            point_clouds = [
                torch.cat([pc, missing], dim=0).float() for pc, missing in zip(point_clouds, missing_points)
            ]

        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()

        return point_clouds

    def _get_actor_trajectories(self) -> List[Dict]:
        """Returns a list of actor trajectories."""
        allowed_classes = ALLOWED_RIGID_CLASSES
        if self.config.include_deformable_actors:
            allowed_classes += ALLOWED_DEFORMABLE_CLASSES
        cuboids = []
        for i in range(PANDASET_SEQ_LEN):
            curr_cuboids = self.sequence.cuboids[i]
            # Remove invalid cuboids
            is_allowed_class = np.array([label in allowed_classes for label in curr_cuboids["label"]])
            valid_mask = (~curr_cuboids["stationary"]) & is_allowed_class
            curr_cuboids = curr_cuboids[valid_mask]
            if not len(curr_cuboids):
                continue

            uuid = np.array(curr_cuboids["uuid"])
            label = np.array(curr_cuboids["label"])

            yaw = curr_cuboids["yaw"].astype(np.float32)
            rot = _yaw_to_rotation_matrix(yaw)

            stationary = np.array(curr_cuboids["stationary"], dtype=np.bool8)  # True for static objects
            pos_x = curr_cuboids["position.x"].astype(np.float32)  # x position of cuboid in world coords
            pos_y = curr_cuboids["position.y"].astype(np.float32)  # y position of cuboid in world coords
            pos_z = curr_cuboids["position.z"].astype(np.float32)  # z position of cuboid in world coords
            pos = np.vstack([pos_x, pos_y, pos_z]).T

            cuboid_poses = np.eye(4)[None].repeat(len(uuid), axis=0)
            cuboid_poses[:, :3, :3] = rot
            cuboid_poses[:, :3, 3] = pos

            width = curr_cuboids["dimensions.x"].astype(np.float32)  # width of cuboid in world coords
            length = curr_cuboids["dimensions.y"].astype(np.float32)  # length of cuboid in world coords
            height = curr_cuboids["dimensions.z"].astype(np.float32)  # height of cuboid in world coords
            dims = np.vstack([width, length, height]).T

            # if dynamic and visible in both 360 and front facing, two cuboids are annoated. 0 for 360, 1 for front facing, -1 otherwise
            sensor_id = np.array(curr_cuboids["cuboids.sensor_id"], dtype=np.int32)
            sibling_id = np.array(
                curr_cuboids["cuboids.sibling_id"]
            )  # uuid of sibling cuboid, i.e., if sensor_id != -1

            if self.config.correct_cuboid_time:
                # correct the cuboid time to match the actual time of observation, not the end of the lidar sweep
                lidpose = _pandaset_pose_to_matrix(self.sequence.lidar.poses[i])
                posinlid = pos @ lidpose[:3, :3].T + lidpose[:3, 3]
                angle = np.arctan2(posinlid[:, 0], posinlid[:, 1]) - np.pi / 2
                angle = (angle + np.pi) % (2 * np.pi) - np.pi
                timediff = angle / (2 * np.pi) * np.diff(self.sequence.lidar.timestamps).mean()
                cuboid_times = self.sequence.camera["front_camera"].timestamps[i] + timediff
            else:
                # assume the cuboid time matches the sequence time
                cuboid_times = np.repeat(self.sequence.camera["front_camera"].timestamps[i], len(uuid))

            for cuboid_index in range(len(uuid)):
                cuboids.append(
                    {
                        "uuid": uuid[cuboid_index],
                        "label": label[cuboid_index],
                        "poses": cuboid_poses[cuboid_index],
                        "stationary": stationary[cuboid_index],
                        "dims": dims[cuboid_index],
                        "sensor_ids": sensor_id[cuboid_index],
                        "sibling_id": sibling_id[cuboid_index] if sensor_id[cuboid_index] != -1 else None,
                        "timestamps": np.array(cuboid_times[cuboid_index]),
                    }
                )
        return _cuboids_to_trajectories(cuboids)

    def _generate_dataparser_outputs(self, split="train") -> DataparserOutputs:
        pandaset = DataSet(str(self.config.data.absolute()))

        if self.config.sequence not in pandaset.sequences():
            raise ValueError(f"Sequence {self.config.sequence} not found in {self.config.data}")

        self.sequence = pandaset[self.config.sequence]
        self.sequence.load()
        self.extrinsics = yaml.load(open(EXTRINSICS_FILE_PATH, "r"), Loader=yaml.FullLoader)
        return super()._generate_dataparser_outputs(split)

    def _add_channel_info(self, point_cloud: torch.Tensor, dim: int = -1, lidar_name: str = "") -> torch.Tensor:
        """Infer channel id from point cloud, and add it to the point cloud.

        Args:
            point_cloud: Point cloud to add channel id to (in sensor frame). Shape: [num_points, 3+x] x,y,z (timestamp, intensity, etc.)

        Returns:
            Point cloud with channel id. Shape: [num_points, 3+x+1] x,y,z (timestamp, intensity, etc.), channel_id
            channel_id is added to dim
        """
        # these are limits where channels are equally spaced
        ELEV_HIGH_IDX = 5
        ELEV_LOW_IDX = -11
        ELEV_LOW_IDX_ABS = len(self.config.lidar_elevation_mapping[lidar_name]) + ELEV_LOW_IDX

        dist = torch.norm(point_cloud[:, :3], dim=-1)
        elevation = torch.arcsin(point_cloud[:, 2] / dist)
        elevation = torch.rad2deg(elevation)

        middle_elev_mask = (elevation < (self.config.lidar_elevation_mapping[lidar_name][ELEV_HIGH_IDX] + 0.2)) & (
            elevation > (self.config.lidar_elevation_mapping[lidar_name][ELEV_LOW_IDX_ABS] - 0.2)
        )
        middle_elev = elevation[middle_elev_mask]

        histc, bin_edges = torch.histogram(middle_elev, bins=2000)

        # channels should be equally spaced
        expected_channel_edges = (bin_edges[-1] - bin_edges[0]) / 49 * torch.arange(50) + bin_edges[0]

        res = (
            self.config.lidar_elevation_mapping[lidar_name][ELEV_HIGH_IDX]
            - self.config.lidar_elevation_mapping[lidar_name][ELEV_HIGH_IDX + 1]
        )

        # find consecutive empty bins in histogram
        empty_bins = []
        empty_bin = []
        empty_bins_edges = []
        for i in range(len(histc)):
            if histc[i] == 0:
                empty_bin.append(i)
            else:
                if len(empty_bin) > 0:
                    empty_bins.append(empty_bin)
                    empty_bins_edges.append((bin_edges[empty_bin[0]], bin_edges[empty_bin[-1] + 1]))
                    empty_bin = []

        # find channel edges, use first expected for init
        found_channel_edges = [expected_channel_edges[0].tolist()]
        empty_bins_edges = torch.tensor(empty_bins_edges)
        for i, edge in enumerate(expected_channel_edges[1:-1]):
            found_edge = False
            for empty_bin in empty_bins_edges:
                # if edge is in empty bin, keep the edge as is
                if edge > empty_bin[0] and edge < empty_bin[1]:
                    found_channel_edges.append(edge.tolist())
                    found_edge = True
                    break
            if found_edge:
                continue
            distances = torch.abs(edge - empty_bins_edges)
            min_dist_idx = distances.argmin()
            if distances.flatten()[min_dist_idx] < 0.03:
                found_channel_edges.append(empty_bins_edges.flatten()[min_dist_idx].tolist())
                continue

        found_channel_edges.append(expected_channel_edges[-1].tolist())
        found_channel_edges = torch.tensor(found_channel_edges)

        if len(found_channel_edges) < len(expected_channel_edges):
            # we have missing channels, interpolate edges
            while (num_missing_edges := len(expected_channel_edges) - len(found_channel_edges)) > 0:
                distances = found_channel_edges.diff().abs()
                max_dist_idx = distances.argmax()
                num_edges_to_insert = max((distances[max_dist_idx] / res).round().int() - 1, 1)
                num_edges_to_insert = min(num_missing_edges, num_edges_to_insert)
                new_edges = torch.linspace(
                    found_channel_edges[max_dist_idx], found_channel_edges[max_dist_idx + 1], num_edges_to_insert + 2
                )[1:-1]
                found_channel_edges = torch.cat(
                    [found_channel_edges[: max_dist_idx + 1], new_edges, found_channel_edges[max_dist_idx + 1 :]]
                )  # insert new edges

        # add remaining edges
        for i in range(len(self.config.lidar_elevation_mapping[lidar_name])):
            if i >= ELEV_HIGH_IDX and i <= len(self.config.lidar_elevation_mapping[lidar_name]) + ELEV_LOW_IDX:
                continue
            current_elevation = self.config.lidar_elevation_mapping[lidar_name][i]
            if i == 0:
                new_edge = 100
            elif i == len(self.config.lidar_elevation_mapping[lidar_name]) - 1:
                new_edge = -100
            elif i < ELEV_HIGH_IDX:
                dist_to_prev_elevation = abs(current_elevation - self.config.lidar_elevation_mapping[lidar_name][i - 1])
                new_edge = current_elevation + dist_to_prev_elevation * 0.22
            elif i > len(self.config.lidar_elevation_mapping[lidar_name]) + ELEV_LOW_IDX:
                dist_to_next_elevation = (
                    abs(current_elevation - self.config.lidar_elevation_mapping[lidar_name][i + 1])
                    if i < len(self.config.lidar_elevation_mapping[lidar_name]) - 1
                    else 1000.0
                )
                new_edge = current_elevation - dist_to_next_elevation * 0.22

            found_channel_edges = torch.cat([found_channel_edges, torch.tensor([new_edge]).float()])

        found_channel_edges, _ = torch.sort(found_channel_edges, descending=True)
        channel_id = torch.full((point_cloud.shape[0], 1), -1, device=point_cloud.device)

        # assign channel id
        for i in range(len(self.config.lidar_elevation_mapping[lidar_name])):
            elevation_mask = (elevation >= found_channel_edges[i + 1]) & (elevation < found_channel_edges[i])
            channel_id[elevation_mask] = i

        point_cloud = torch.cat([point_cloud[:, :dim], channel_id, point_cloud[:, dim:]], dim=-1)
        return point_cloud


def _pandaset_pose_to_matrix(pose):
    translation = np.array([pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]])
    quaternion = np.array([pose["heading"]["w"], pose["heading"]["x"], pose["heading"]["y"], pose["heading"]["z"]])
    pose = np.eye(4)
    pose[:3, :3] = pyquaternion.Quaternion(quaternion).rotation_matrix
    pose[:3, 3] = translation
    return pose


def _yaw_to_rotation_matrix(yaw: np.ndarray):
    """Converts array of yaw angles to rotation matrices."""
    rotation_matrices = np.zeros((yaw.shape[0], 3, 3))
    rotation_matrices[:, 0, 0] = np.cos(yaw)
    rotation_matrices[:, 0, 1] = -np.sin(yaw)
    rotation_matrices[:, 1, 0] = np.sin(yaw)
    rotation_matrices[:, 1, 1] = np.cos(yaw)
    rotation_matrices[:, 2, 2] = 1
    return rotation_matrices


def _cuboids_to_trajectories(cuboids):
    """Connects cuboids into trajectories."""
    trajs = []
    trajs_dict = {}
    for cuboid in cuboids:
        if cuboid["sensor_ids"] == 1:  # TODO: allow for cuboids from front-facing lidar
            continue  # skip cuboids from front-facing lidar

        if cuboid["uuid"] not in trajs_dict:
            trajs_dict[cuboid["uuid"]] = []

        trajs_dict[cuboid["uuid"]] += [cuboid]

    for uuid, traj in trajs_dict.items():
        trajs_dict[uuid] = sorted(traj, key=lambda x: x["timestamps"])
        trajs.append(
            {
                # "uuid": uuid,
                "poses": torch.from_numpy(np.stack([t["poses"] for t in traj])).float(),
                "timestamps": torch.from_numpy(np.stack([t["timestamps"] for t in traj])),
                "dims": torch.from_numpy(np.array([t["dims"] for t in traj]).astype(np.float32).max(axis=0)),
                "label": traj[0]["label"],
                "stationary": traj[0]["stationary"],
                "symmetric": "Pedestrian" not in traj[0]["label"],
                "deformable": "Pedestrian" in traj[0]["label"],
            }
        )
    return trajs
