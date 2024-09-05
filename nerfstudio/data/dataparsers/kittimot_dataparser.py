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

"""Data parser for the KITTI MOT dataset"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Type

import numpy as np
import numpy.typing as npt
import torch
from pyquaternion import Quaternion
from torch import Tensor
from typing_extensions import Literal

from nerfstudio.cameras.camera_utils import rotmat_to_unitquat, unitquat_slerp, unitquat_to_rotmat
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import OPENCV_TO_NERFSTUDIO, ADDataParser, ADDataParserConfig
from nerfstudio.utils.poses import to4x4

HORIZONTAL_BEAM_DIVERGENCE = 3.0e-3  # radians, or meters at a distance of 1m
VERTICAL_BEAM_DIVERGENCE = 1.5e-3  # radians, or meters at a distance of 1m
MAX_INTENSITY_VALUE = 1.0
KITTI_IMAGE_WIDTH = 1242
KITTI_IMAGE_HEIGHT = 375

ALLOWED_CATEGORIES = {
    "Car",
    "Van",
    "Truck",
    "Cyclist",
    "Tram",
}  # skip Misc and DontCare
SYMMETRIC_CATEGORIES = ALLOWED_CATEGORIES

DEFORMABLE_CATEGORIES = {
    "Pedestrian",
    "Person_sitting",
}

AVAILABLE_CAMERAS = ("image_02", "image_03")

DATA_FREQUENCY = 10.0  # 10 Hz
LIDAR_ROTATION_TIME = 1.0 / DATA_FREQUENCY  # 10 Hz

MARS_SEQ_TO_START_FRAME = {str(i).zfill(4): 0 for i in range(1, 21)}
MARS_SEQ_TO_END_FRAME = {str(i).zfill(4): -1 for i in range(1, 21)}
# MARS only evaluate on the 0006 sequence. Note that the latest number on their github uses
# frames from 65 -> 120, but the results in their paper uses from frame 5 -> 260, which
# is what we will use to compare with.
MARS_SEQ_TO_START_FRAME["0006"] = 5  # taken from the paper
MARS_SEQ_TO_END_FRAME["0006"] = 260  # taken from the paper

RIGHT_FRONT_UP2RIGHT_DOWN_FRONT = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])


@dataclass
class KittiMotDataParserConfig(ADDataParserConfig):
    """KITTI MOT DatasetParser config."""

    _target: Type = field(default_factory=lambda: KittiMot)
    """target class to instantiate"""
    sequence: str = "0006"
    """Name of the scene."""
    data: Path = Path("data/kittimot")
    """Path to KITTI-MOT dataset."""
    split: Literal["training"] = "training"  # we do not have labels for testing set...
    """Which split to use."""
    cameras: Tuple[Literal["image_02", "image_03", "none", "all"], ...] = ("image_02", "image_03")
    """Which cameras to use."""
    lidars: Tuple[Literal["velodyne", "none"], ...] = ("velodyne",)
    """Which lidars to use."""
    annotation_interval: float = 0.1
    """Interval between annotations in seconds."""
    allow_per_point_times: bool = False
    """Whether to allow per-point timestamps."""
    compute_sensor_velocities: bool = False
    """Whether to compute sensor velocities."""
    min_lidar_dist: Tuple[float, ...] = (2.0, 1.6, 2.0)
    """Minimum distance of lidar points."""
    use_mars_nvs_50_split: bool = False
    """Whether to use the MARS NVS 50 proposed split."""
    use_sensor_timestamps: bool = True
    """Whether to use sensor timestamps."""


@dataclass
class KittiMot(ADDataParser):
    """Zenseact Open Dataset DatasetParser"""

    config: KittiMotDataParserConfig

    @property
    def actor_transform(self) -> Tensor:
        """The transform needed to convert the actor poses to our desired format (x-right, y-forward, z-up)."""
        return torch.from_numpy(RIGHT_FRONT_UP2RIGHT_DOWN_FRONT)

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        """Returns camera info and image filenames."""

        if "all" in self.config.cameras:
            self.config.cameras = AVAILABLE_CAMERAS

        filenames, times, poses, idxs = [], [], [], []
        # we will convert everything to the common camera, hence we use the camera parameters of the common camera
        fx = self.calibs["P0"][0, 0].copy()
        fy = self.calibs["P0"][1, 1].copy()
        cx = self.calibs["P0"][0, 2].copy()
        cy = self.calibs["P0"][1, 2].copy()

        for camera_idx, cam_name in enumerate(self.config.cameras):
            camera_folder = self.config.data / self.config.split / cam_name / self.config.sequence
            camera_files = sorted(camera_folder.glob("*.png"))

            for camera_file in camera_files:
                image_id = int(camera_file.stem)
                filenames.append(camera_file)
                if self.config.use_sensor_timestamps:
                    times.append(self.timestamp_per_sensor[cam_name][image_id])
                else:
                    times.append(image_id / DATA_FREQUENCY)
                ego_pose = self._get_ego_pose(image_id, sensor=cam_name).double()
                idxs.append(camera_idx)
                # get the ego pose
                velo2cam = torch.from_numpy(self.calibs["Tr_velo_to_cam"].copy()).double()  # 3x4
                cam2velo = to4x4(velo2cam).inverse()
                imu2velo = torch.from_numpy(self.calibs["Tr_imu_to_velo"].copy()).double()  # 3x4
                velo2imu = to4x4(imu2velo).inverse()
                # cam to imu
                cam2imu = velo2imu @ cam2velo
                # consider both the rectification similar to here
                # https://github.com/autonomousvision/kitti360Scripts/blob/7ecc14eab6fa30e5d2ac71ad37fed2bb4b0b8073/kitti360scripts/helpers/project.py#L35-L41
                rect_inv = torch.eye(4, dtype=torch.double)
                rect_inv[:3, :3] = torch.from_numpy(self.calibs["R_rect"].copy()).inverse().double()

                # get the calib for the current camera
                camera_int = int(cam_name.split("_")[1])
                calib = self.calibs[f"P{camera_int}"].copy()
                # consider that the cameras are rectified wrt to the common camera

                cam2commoncam = (
                    to4x4(torch.from_numpy(calib).double()).inverse()
                    @ to4x4(torch.from_numpy(self.calibs["P0"].copy())).double()
                )
                cam_pose = ego_pose @ cam2imu @ rect_inv @ cam2commoncam
                # consider the nerfstudio to opencv conversion
                cam_pose[:3, :3] = cam_pose[:3, :3] @ torch.from_numpy(OPENCV_TO_NERFSTUDIO).double()
                poses.append(cam_pose.float()[:3, :4])

        # To tensors
        poses = torch.stack(poses)
        times = torch.tensor(times, dtype=torch.float64)  # need higher precision
        idxs = torch.tensor(idxs).int().unsqueeze(-1)

        cameras = Cameras(
            fx=torch.tensor(fx, dtype=torch.float32),
            fy=torch.tensor(fy, dtype=torch.float32),
            cx=torch.tensor(cx, dtype=torch.float32),
            cy=torch.tensor(cy, dtype=torch.float32),
            height=KITTI_IMAGE_HEIGHT,
            width=KITTI_IMAGE_WIDTH,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={"sensor_idxs": idxs},
        )
        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        """Returns lidar info and loaded point clouds."""
        times, poses, idxs, lidar_filenames = [], [], [], []

        for lidar_idx, lidar_name in enumerate(self.config.lidars):
            lidar_folder = self.config.data / self.config.split / lidar_name / self.config.sequence

            lidar_files = sorted(lidar_folder.glob("*.bin"))

            # get the calibration of the lidar
            imu2velo = torch.from_numpy(self.calibs["Tr_imu_to_velo"].copy())  # 3x4
            velo2imu = to4x4(imu2velo).inverse()

            for lidar_file in lidar_files:
                frame_idx = int(lidar_file.stem)
                if self.config.use_sensor_timestamps:
                    times.append(self.timestamp_per_sensor[lidar_name][frame_idx])
                else:
                    times.append(frame_idx / DATA_FREQUENCY)

                ego_pose = self._get_ego_pose(frame_idx, sensor=lidar_name).float()
                pose = ego_pose @ velo2imu.clone()
                poses.append(pose)
                idxs.append(lidar_idx)
                lidar_filenames.append(lidar_file)

        # To tensors
        poses = torch.stack(poses)  # will change to float later, see _read_lidars
        times = torch.tensor(times, dtype=torch.float64)
        idxs = torch.tensor(idxs).int().unsqueeze(-1)

        lidars = Lidars(
            lidar_to_worlds=poses[:, :3, :4],
            lidar_type=LidarType.VELODYNE64E,
            times=times,
            assume_ego_compensated=False,
            metadata={"sensor_idxs": idxs},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
        )

        return lidars, lidar_filenames

    def _read_lidars(self, lidars: Lidars, filepaths: List[Path]) -> List[torch.Tensor]:
        point_clouds = []
        for filepath in filepaths:
            pc = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
            xyz = pc[:, :3]  # N x 3
            intensity = pc[:, 3] / MAX_INTENSITY_VALUE  # N,
            t = get_mock_timestamps(xyz)  # N, relative timestamps
            pc = np.hstack((xyz, intensity[:, None], t[:, None]))
            point_clouds.append(torch.from_numpy(pc).float())

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

        if self.config.include_deformable_actors:
            allowed_cats = ALLOWED_CATEGORIES.union(DEFORMABLE_CATEGORIES)
        else:
            allowed_cats = ALLOWED_CATEGORIES

        anno_file = self.config.data / self.config.split / "label_02" / f"{self.config.sequence}.txt"

        # read annotations
        with open(anno_file, "r") as f:
            lines = f.readlines()

        traj_list = defaultdict(list)

        # loop over all annotations to create per agent trajectories
        for line in lines:
            line = line.strip()
            frame, track_id, label, _, _, _, _, _, _, _, height, width, length, x, y, z, rotation_y = line.split(" ")

            # remove all actors that are not in the allowed categories
            if label not in allowed_cats:
                continue

            frame = int(frame)
            track_id = int(track_id)

            height = float(height)
            width = float(width)
            length = float(length)

            # defined in the camera coordinate system of the ego-vehicle
            x = float(x)
            y = float(y) - height / 2.0  # center of the object is at the bottom
            z = float(z)

            #
            rotation_y = float(rotation_y)

            traj_list[track_id].append(
                {
                    "frame": frame,
                    "label": label,
                    "height": height,
                    "width": width,
                    "length": length,
                    "x": x,
                    "y": y,
                    "z": z,
                    "rotation_y": rotation_y,
                }
            )

        trajs = []
        for track_id, track in traj_list.items():
            poses, timestamps, wlh = [], [], []
            if len(track) < 2:
                continue  # skip if there is only one frame

            label = track[0]["label"]
            deformable = label in DEFORMABLE_CATEGORIES
            symmetric = label in SYMMETRIC_CATEGORIES

            # cam to velo
            velo2cam = self.calibs["Tr_velo_to_cam"].copy()  # 3x4
            velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1])))
            cam2velo = np.linalg.inv(velo2cam)
            # imu to velo
            imu2velo = self.calibs["Tr_imu_to_velo"].copy()  # 3x4
            imu2velo = np.vstack((imu2velo, np.array([0, 0, 0, 1])))
            velo2imu = np.linalg.inv(imu2velo)
            # cam to imu
            cam2imu = velo2imu @ cam2velo

            # note that the boxes are in the camera coordinate system of the ego-vehicle
            for box in sorted(track, key=lambda x: x["frame"]):
                if self.config.use_sensor_timestamps:
                    timestamps.append(self.timestamp_per_sensor["velodyne"][box["frame"]])
                else:
                    timestamps.append(box["frame"] / DATA_FREQUENCY)
                # The actors are annotated in the LIDAR point cloud, so we use those timestamps.
                ego_pose = self._get_ego_pose(box["frame"], "velodyne").numpy()
                # cam to world
                cam2world = ego_pose @ cam2imu
                obj_pose_cam = np.eye(4)
                obj_pose_cam[:3, 3] = np.array([box["x"], box["y"], box["z"]])
                obj_pose_cam[:3, :3] = Quaternion(axis=[0, 1, 0], angle=np.pi / 2 + box["rotation_y"]).rotation_matrix
                obj_pose_world = cam2world @ obj_pose_cam @ RIGHT_FRONT_UP2RIGHT_DOWN_FRONT
                poses.append(obj_pose_world)
                wlh.append(np.array([box["width"], box["length"], box["height"]]))

            poses = np.array(poses)
            # dynamic if we move more that 1m in any direction
            dynamic = np.any(np.std(poses[:, :3, 3], axis=0) > 0.5)
            # we skip all stationary objects
            if not dynamic:
                continue

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
        # load the ego_poses
        root = self.config.data / self.config.split
        ego_pose_path = root / "oxts" / f"{self.config.sequence}.txt"
        self.ego_poses = get_poses_from_oxts(ego_pose_path)
        self.calibs = get_calib(root / "calib" / f"{self.config.sequence}.txt")

        # we only use timestamps if all sensors have them.
        if self.config.use_sensor_timestamps:
            self._setup_sensor_timestamps()

        out = super()._generate_dataparser_outputs(split=split)
        del self.ego_poses
        del self.calibs
        return out

    def _setup_sensor_timestamps(self) -> None:
        self.timestamp_per_sensor = {}
        for sensor in self.config.cameras + self.config.lidars + ("oxts",):
            timestamp_file = (
                self.config.data
                / self.config.split
                / sensor
                / "timestamps"
                / f"{self.config.sequence}"
                / "timestamps.txt"
            )
            assert timestamp_file.exists(), f"Trying to use sensor timestamps but file {timestamp_file} does not exist."
            with open(timestamp_file, "r") as f:
                lines = f.readlines()
            # parse the timestamps: 2011-09-26 13:13:32.322364840
            timestamps = []
            for line in lines:
                line = line.strip()[:-3]  # remove nanosecond precision
                dt = datetime.strptime(line, "%Y-%m-%d %H:%M:%S.%f")
                timestamp = dt.timestamp()
                timestamps.append(timestamp)

            self.timestamp_per_sensor[sensor] = np.array(timestamps)

    def _get_ego_pose(self, frame_idx: int, sensor: str):
        if self.config.use_sensor_timestamps:
            ego_pose_timestamps = self.timestamp_per_sensor["oxts"]
            sensor_timestamp = self.timestamp_per_sensor[sensor][frame_idx]
            # get closest ego-pose timestamp
            next_ego_pose_idx = np.searchsorted(ego_pose_timestamps, sensor_timestamp)
            prev_ego_pose_idx = next_ego_pose_idx - 1
            # lets not extrapolate
            if prev_ego_pose_idx < 0:
                return torch.from_numpy(self.ego_poses[0])
            if next_ego_pose_idx >= len(ego_pose_timestamps):
                return torch.from_numpy(self.ego_poses[-1])

            prev_ego_pose = torch.from_numpy(self.ego_poses[prev_ego_pose_idx].copy())  # 4x4
            next_ego_pose = torch.from_numpy(self.ego_poses[next_ego_pose_idx].copy())  # 4x4
            fraction = (sensor_timestamp - ego_pose_timestamps[prev_ego_pose_idx]) / (
                ego_pose_timestamps[next_ego_pose_idx] - ego_pose_timestamps[prev_ego_pose_idx]
            )

            # position
            pos_prev = prev_ego_pose[:3, 3]
            pos_next = next_ego_pose[:3, 3]
            interp_pos = pos_prev + (pos_next - pos_prev) * fraction

            # rotation
            quat_prev = rotmat_to_unitquat(prev_ego_pose[:3, :3])
            quat_next = rotmat_to_unitquat(next_ego_pose[:3, :3])
            interp_quat = unitquat_slerp(quat_prev.float(), quat_next.float(), torch.tensor(fraction))
            interp_rot = unitquat_to_rotmat(interp_quat)

            # concat
            interp_pose = torch.eye(4)
            interp_pose[:3, 3] = interp_pos.float()
            interp_pose[:3, :3] = interp_rot
            return interp_pose
        else:
            return torch.from_numpy(self.ego_poses[frame_idx].copy())

    def _get_linspaced_indices(self, sensor_idxs: Tensor) -> Tuple[Tensor, Tensor]:
        # if we are using all the samples, i.e., optimizing poses, we can use the same for eval
        if self.config.train_split_fraction == 1.0:
            return torch.arange(sensor_idxs.numel(), dtype=torch.int64), torch.arange(
                sensor_idxs.numel(), dtype=torch.int64
            )

        if self.config.use_mars_nvs_50_split:
            # based on the implementation found here:
            # https://github.com/OPEN-AIR-SUN/mars/blob/master/mars/data/mars_kitti_dataparser.py#L1090-L1114
            start_frame = MARS_SEQ_TO_START_FRAME[self.config.sequence]
            end_frame = MARS_SEQ_TO_END_FRAME[self.config.sequence]
            if end_frame != -1:
                end_frame += 1  # they are inclusive

            # for each camera the logic is:
            # [train, train, skip, eval] and repeat this
            first_sensor_idxes = sensor_idxs[sensor_idxs == sensor_idxs.unique()[0]]
            train_mask = torch.zeros_like(first_sensor_idxes)
            train_mask[start_frame:end_frame:4] = 1
            train_mask[start_frame + 1 : end_frame : 4] = 1

            eval_mask = torch.zeros_like(first_sensor_idxes)
            eval_mask[start_frame + 3 : end_frame : 4] = 1

            # repeat this for the number of unique sensors
            train_mask = torch.concat([train_mask] * len(sensor_idxs.unique()))
            eval_mask = torch.concat([eval_mask] * len(sensor_idxs.unique()))
            assert (train_mask + eval_mask).max() == 1, "Something went wrong with the indices."

            idxes = torch.arange(len(sensor_idxs))
            train_indices = idxes[train_mask.bool()]
            eval_indices = idxes[eval_mask.bool()]
            return train_indices, eval_indices
        else:
            return super()._get_linspaced_indices(sensor_idxs)


def lat_lon_to_mercator(
    lat: npt.NDArray[np.float64], lon: npt.NDArray[np.float64], scale: float
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Converts lat/lon coordinates to mercator coordinates.

    Reimplemented from the kitti dataset devkit.
    """

    earth_radius = 6378137.0
    mx = lon * scale * earth_radius * np.pi / 180.0
    my = np.log(np.tan((90 + lat) * np.pi / 360.0)) * scale * earth_radius
    return mx, my


def get_poses_from_oxts(ego_pose_path: Path) -> npt.NDArray[np.float64]:
    """Takes the path to the KITTI oxts file and returns the poses for each timestamp.

    Reimplemented based on the KITTI dataset devkit.
    """
    oxts = np.loadtxt(ego_pose_path, dtype=np.float64)
    lat, lon, alt = oxts[:, 0], oxts[:, 1], oxts[:, 2]
    roll, pitch, yaw = oxts[:, 3], oxts[:, 4], oxts[:, 5]

    scale = np.cos(lat[0] * np.pi / 180.0)
    x, y = lat_lon_to_mercator(lat, lon, scale)
    z = alt

    Rx = np.array(
        [
            [np.ones_like(roll), np.zeros_like(roll), np.zeros_like(roll)],
            [np.zeros_like(roll), np.cos(roll), -np.sin(roll)],
            [np.zeros_like(roll), np.sin(roll), np.cos(roll)],
        ]
    )  # 3x3xN
    Ry = np.array(
        [
            [np.cos(pitch), np.zeros_like(roll), np.sin(pitch)],
            [np.zeros_like(roll), np.ones_like(roll), np.zeros_like(roll)],
            [-np.sin(pitch), np.zeros_like(roll), np.cos(pitch)],
        ]
    )  # 3x3xN
    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), np.zeros_like(roll)],
            [np.sin(yaw), np.cos(yaw), np.zeros_like(roll)],
            [np.zeros_like(roll), np.zeros_like(roll), np.ones_like(roll)],
        ]
    )  # 3x3xN
    # make into N x 3 x 3
    Rx = np.transpose(Rx, (2, 0, 1))
    Ry = np.transpose(Ry, (2, 0, 1))
    Rz = np.transpose(Rz, (2, 0, 1))

    R = Rz @ Ry @ Rx

    poses = np.zeros((len(oxts), 4, 4))
    poses[:, :3, :3] = R
    poses[:, :3, 3] = np.array([x, y, z]).T
    poses[:, 3, 3] = 1

    # normalize to the first pose
    poses = np.linalg.inv(poses[0]) @ poses

    return poses


def get_calib(calib_path: Path) -> Dict[str, npt.NDArray[np.float32]]:
    with open(str(calib_path), "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("P0:"):
            P0 = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        elif line.startswith("P1:"):
            P1 = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        elif line.startswith("P2:"):
            P2 = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        elif line.startswith("P3:"):
            P3 = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        elif line.startswith("R_rect"):
            R_rect = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 3)
        elif line.startswith("Tr_velo_cam"):
            Tr_velo_to_cam = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        elif line.startswith("Tr_imu_velo"):
            Tr_imu_to_velo = np.array(line.split(" ")[1:], dtype=np.float32).reshape(3, 4)
        else:
            raise ValueError(f"Unknown calibration line: {line}")
    return {
        "P0": P0,  # type: ignore
        "P1": P1,  # type: ignore
        "P2": P2,  # type: ignore
        "P3": P3,  # type: ignore
        "R_rect": R_rect,  # type: ignore
        "Tr_velo_to_cam": Tr_velo_to_cam,  # type: ignore
        "Tr_imu_to_velo": Tr_imu_to_velo,  # type: ignore
    }


def get_mock_timestamps(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Get mock relative timestamps for the velodyne points."""
    # the velodyne has x forward, y left, z up and the sweep is split behind the car.
    # it is also rotating counter-clockwise, meaning that the angles close to -pi are the
    # first ones in the sweep and the ones close to pi are the last ones in the sweep.
    angles = np.arctan2(points[:, 1], points[:, 0])  # N, [-pi, pi]
    angles += np.pi  # N, [0, 2pi]
    # see how much of the rotation have finished
    fraction_of_rotation = angles / (2 * np.pi)  # N, [0, 1]
    # get the pseudo timestamps based on the total rotation time
    timestamps = fraction_of_rotation * LIDAR_ROTATION_TIME
    return timestamps
