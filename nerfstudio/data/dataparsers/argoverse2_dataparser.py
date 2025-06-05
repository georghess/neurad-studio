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

"""Dataparser for the argoverse2 dataset."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type

import av2.utils.io as io_utils
import numpy as np
import torch
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader, convert_pose_dataframe_to_SE3
from av2.datasets.sensor.constants import AnnotationCategories
from av2.geometry.geometry import quat_to_mat
from av2.structures.sweep import Sweep
from av2.utils.io import read_ego_SE3_sensor, read_feather

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType, transform_points
from nerfstudio.data.dataparsers.ad_dataparser import (
    DUMMY_DISTANCE_VALUE,
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
)
from nerfstudio.data.utils.lidar_elevation_mappings import ARGOVERSE2_VELODYNE_VLP32C_ELEVATION_MAPPING
from nerfstudio.utils import poses as pose_utils

MAX_REFLECTANCE_VALUE = 255.0
HOOD_HEIGHT = 250  # number of pixels to crop from the bottom due to the hood

WLH_TO_LWH = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
HORIZONTAL_BEAM_DIVERGENCE = 3e-3  # radians
VERTICAL_BEAM_DIVERGENCE = 1.5e-3  # radians

SYMMETRIC_CATEGORIES = {
    AnnotationCategories.ARTICULATED_BUS,
    AnnotationCategories.BUS,
    AnnotationCategories.BOLLARD,
    AnnotationCategories.BOX_TRUCK,
    AnnotationCategories.CONSTRUCTION_CONE,
    AnnotationCategories.LARGE_VEHICLE,
    AnnotationCategories.RAILED_VEHICLE,
    AnnotationCategories.REGULAR_VEHICLE,
    AnnotationCategories.SCHOOL_BUS,
    AnnotationCategories.TRUCK,
    AnnotationCategories.TRUCK_CAB,
    AnnotationCategories.VEHICULAR_TRAILER,
}

DEFORMABLE_CATEGORIES = {
    AnnotationCategories.PEDESTRIAN,
    AnnotationCategories.OFFICIAL_SIGNALER,
}

ALLOWED_CATEGORIES = {
    AnnotationCategories.ARTICULATED_BUS,
    AnnotationCategories.BOX_TRUCK,
    AnnotationCategories.BUS,
    AnnotationCategories.LARGE_VEHICLE,
    AnnotationCategories.RAILED_VEHICLE,
    AnnotationCategories.REGULAR_VEHICLE,
    AnnotationCategories.SCHOOL_BUS,
    AnnotationCategories.TRUCK,
    AnnotationCategories.TRUCK_CAB,
    AnnotationCategories.VEHICULAR_TRAILER,
    # motor cycles
    AnnotationCategories.MOTORCYCLE,
    AnnotationCategories.MOTORCYCLIST,
    # bicycles
    AnnotationCategories.BICYCLE,
    AnnotationCategories.BICYCLIST,
    # wheel devices
    AnnotationCategories.WHEELED_DEVICE,
    AnnotationCategories.WHEELED_RIDER,
}

CAMERA_TO_BOTTOM_RIGHT_CROP = {
    "ring_front_center": (250, 0),
    "ring_front_left": (0, 0),
    "ring_front_right": (0, 0),
    "ring_rear_left": (250, 0),
    "ring_rear_right": (250, 0),
    "ring_side_left": (0, 0),
    "ring_side_right": (0, 0),
}

AVAILABLE_CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
)

AVAILABLE_LIDARS = ("lidar_up", "lidar_down")


# lidar_up is the upper lidar, lidar_down is the lower lidar, and lidar is short for any of them
AV2_VELO_VLP32C_ELEVATION_MAPPING = ARGOVERSE2_VELODYNE_VLP32C_ELEVATION_MAPPING
AV2_ELEVATION_MAPPING = {
    "lidar_down": AV2_VELO_VLP32C_ELEVATION_MAPPING,
    "lidar_up": AV2_VELO_VLP32C_ELEVATION_MAPPING,
    "lidar": AV2_VELO_VLP32C_ELEVATION_MAPPING,
}
AV2_AZIMUTH_RESOLUTION = {
    "lidar_down": 0.2,
    "lidar_up": 0.2,
    "lidar": 0.2,
}  # degrees
AV2_SKIP_ELEVATION_CHANNELS = {
    "lidar": (0, 1, 2),
    "lidar_down": (4, 15, 0),  # corresponding to 15,10.333,7.0 degrees as lidar_down is upside down
    "lidar_up": (0, 3, 4),  # corresponding to -25,-15.639,-11.31 degrees
}
AV2_IGNORE_REGIONS = {
    "lidar_down": [[140, 200, -30.0, 20.0], [-200.0, -170.0, -30.0, 20.0]],
    "lidar_up": [],
    "lidar": [],
}


@dataclass
class Argoverse2DataParserConfig(ADDataParserConfig):
    """Argoverse 2 dataset config."""

    _target: Type = field(default_factory=lambda: Argoverse2)
    """target class to instantiate"""
    sequence: str = "2b044433-ddc1-3580-b560-d46474934089"
    """name of the sequence to use"""
    data: Path = Path("data/av2")
    """path to the dataset"""
    cameras: Tuple[
        Literal[
            "ring_front_center",
            "ring_front_left",
            "ring_front_right",
            "ring_rear_left",
            "ring_rear_right",
            "ring_side_left",
            "ring_side_right",
            "none",
            "all",
        ],
        ...,
    ] = ("all",)
    """what cameras to use"""
    lidars: Tuple[Literal["lidar_up", "lidar_down", "all", "none"], ...] = ("all",)
    """what lidars to use. only one lidar is available."""
    annotation_interval: float = 0.1
    """interval between annotations in seconds"""
    split: str = "val"
    """what split to use. options are: train, val, test, mini, mini_val, mini_test"""
    add_missing_points: bool = True
    """whether to add missing points to the point clouds"""
    lidar_elevation_mapping: Dict[str, Dict[int, float]] = field(default_factory=lambda: AV2_ELEVATION_MAPPING)
    """Elevation mapping for each lidar."""
    lidar_azimuth_resolution: Dict[str, float] = field(default_factory=lambda: AV2_AZIMUTH_RESOLUTION)
    """Azimuth resolution for each lidar."""
    skip_elevation_channels: Dict[str, Tuple] = field(default_factory=lambda: AV2_SKIP_ELEVATION_CHANNELS)
    """Channels to skip when adding missing points."""
    output_lidars_separately: bool = True
    """whether to output the two combined lidars as separate instances"""


@dataclass
class Argoverse2(ADDataParser):
    """Argoverse2 DatasetParser."""

    config: Argoverse2DataParserConfig

    @property
    def actor_transform(self) -> torch.Tensor:
        """Argo uses x-forward, so we need to rotate to x-right."""
        wlh_to_lwh = np.eye(4)
        wlh_to_lwh[:3, :3] = WLH_TO_LWH
        return torch.from_numpy(wlh_to_lwh)

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        """Returns camera info and image filenames."""
        if "all" in self.config.cameras:
            self.config.cameras = AVAILABLE_CAMERAS
        filenames, times, intrinsics, poses, idxs = [], [], [], [], []
        heights, widths = [], []

        for camera_idx, cam_name in enumerate(self.config.cameras):
            pinhole_camera = self.av2.get_log_pinhole_camera(self.config.sequence, cam_name)

            # get all camera paths
            camera_paths = self.av2.get_ordered_log_cam_fpaths(self.config.sequence, cam_name)
            # timestamps is the filename without the extension
            timestamps = [int(p.stem) for p in camera_paths]

            # compute the poses of each camera at each timestamp
            for t, fp in zip(timestamps, camera_paths):
                # get the ego-pose
                ego2world = self.av2.get_city_SE3_ego(self.config.sequence, t)
                cam_extrinsics = pinhole_camera.ego_SE3_cam.transform_matrix.copy()
                cam_extrinsics[:3, :3] = cam_extrinsics[:3, :3] @ OPENCV_TO_NERFSTUDIO
                poses.append(ego2world.transform_matrix @ cam_extrinsics)
                # add the camera intrinsics
                intrinsics.append(pinhole_camera.intrinsics.K)
                # add the camera height and width, which differ between cameras
                # the ring_front_center is in portrait mode and the others are in landscape mode
                heights.append(pinhole_camera.intrinsics.height_px - CAMERA_TO_BOTTOM_RIGHT_CROP[cam_name][0])
                widths.append(pinhole_camera.intrinsics.width_px - CAMERA_TO_BOTTOM_RIGHT_CROP[cam_name][1])
                # append the times and filenames to the lists
                times.append(t / 1e9)  # convert to seconds
                filenames.append(fp)
                # add the camera index
                idxs.append(camera_idx)

        # To tensors
        intrinsics = torch.from_numpy(np.array(intrinsics)).float()
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
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={"sensor_idxs": idxs},
        )

        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        """Returns lidar info and loaded point clouds."""

        (
            lidar_filenames,
            times,
            poses_up,
            poses_down,
            idxs,
        ) = ([], [], [], [], [])

        if "all" in self.config.lidars:
            self.config.lidars = AVAILABLE_LIDARS

        lidar_times = self.av2.get_ordered_log_lidar_timestamps(self.config.sequence)

        for t in lidar_times:
            # get the path to the lidar point cloud
            pc_path = self.av2.get_lidar_fpath(self.config.sequence, t)
            lidar_filenames.append(pc_path)

            log_dir = pc_path.parent.parent.parent
            sensor_name_to_pose = read_ego_SE3_sensor(log_dir=log_dir)
            ego_SE3_up_lidar = sensor_name_to_pose["up_lidar"]
            ego_SE3_down_lidar = sensor_name_to_pose["down_lidar"]

            # get the ego-vehicle pose at the lidar timestamp
            ego_pose = self.av2.get_city_SE3_ego(self.config.sequence, t)
            # move to the actual lidar position
            ego_pose_up = ego_pose.compose(ego_SE3_up_lidar)
            ego_pose_down = ego_pose.compose(ego_SE3_down_lidar)
            # add the pose
            poses_up.append(ego_pose_up.transform_matrix)
            poses_down.append(ego_pose_down.transform_matrix)
            idxs.append(0)  # add the lidar index (there is only one so lets use 0)
            times.append(t / 1e9)  # convert to seconds

        poses_up = torch.tensor(np.array(poses_up), dtype=torch.float32)
        poses_down = torch.tensor(np.array(poses_down), dtype=torch.float32)
        down2up = pose_utils.inverse(poses_up[0, ...]) @ poses_down[0, ...]
        times = torch.tensor(np.array(times, dtype=np.float64), dtype=torch.float64)  # need higher precision
        idxs = torch.tensor(np.array(idxs)).int().unsqueeze(-1)
        # we will store the down2up transformation in the metadata, as we will need it later
        down2up = down2up.repeat(len(poses_up), 1, 1).view(len(poses_up), -1)

        if self.config.output_lidars_separately:
            lidars = Lidars(
                lidar_to_worlds=torch.cat([poses_up, poses_down], dim=0)[:, :3, :4],
                lidar_type=LidarType.VELODYNE_VLP32C,
                times=torch.cat([times, times], dim=0),
                metadata={
                    "sensor_idxs": torch.cat([idxs, idxs + (idxs.max() + 1)], dim=0),
                    "down2up": torch.cat([down2up, down2up], dim=0),  # will be removed later, here for compatibility
                    "poses_down": torch.cat(
                        [poses_down.view(len(poses_up), -1), poses_down.view(len(poses_up), -1)], dim=0
                    ),  # will be removed later, here for compatibility
                },
                horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
                vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
                valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
            )
        else:
            lidars = Lidars(
                lidar_to_worlds=poses_up[:, :3, :4],
                lidar_type=LidarType.VELODYNE_VLP32C,
                times=times,
                metadata={
                    "sensor_idxs": idxs,
                    "down2up": down2up,  # will be removed later
                    "poses_down": poses_down.view(len(poses_up), -1),  # will be removed later
                },
                horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
                vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
                valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
            )

        return lidars, lidar_filenames

    def _read_lidars(self, lidars: Lidars, filepaths: List[Path]) -> List[torch.Tensor]:
        point_clouds_up, point_clouds_down = [], []

        for pc_path in filepaths:
            sweep = Sweep.from_feather(pc_path)
            # get laser_number, 0-31 is up_lidar, 32-63 is down_lidar according to
            laser_number = sweep.laser_number
            # transform the points back to the sensor position
            xyz_up = sweep.ego_SE3_up_lidar.inverse().transform_point_cloud(sweep.xyz[laser_number < 32])
            xyz_down = sweep.ego_SE3_down_lidar.inverse().transform_point_cloud(sweep.xyz[laser_number >= 32])
            # normalize the intensity values to be in [0, 1]
            intensity = sweep.intensity / MAX_REFLECTANCE_VALUE
            # Add relative time
            time = sweep.offset_ns / 1e9
            # add the intensity as the last channel
            xyz_intensity_up = np.concatenate(
                [
                    xyz_up,
                    intensity[laser_number < 32, None],
                    time[laser_number < 32, None],
                    laser_number[laser_number < 32, None],
                ],
                axis=1,
            )
            xyz_intensity_down = np.concatenate(
                [
                    xyz_down,
                    intensity[laser_number >= 32, None],
                    time[laser_number >= 32, None],
                    laser_number[laser_number >= 32, None],
                ],
                axis=1,
            )
            # add the point cloud
            point_clouds_up.append(torch.from_numpy(xyz_intensity_up))
            point_clouds_down.append(torch.from_numpy(xyz_intensity_down))

        missing_points_up = []
        missing_points_down = []

        assert lidars.times is not None  # typehints
        assert lidars.metadata is not None  # typehints
        down2up = lidars.metadata["down2up"].view(len(lidars.times), 3, 4)[0]
        if self.config.add_missing_points:
            poses_down = lidars.metadata["poses_down"].view(len(lidars.times), 4, 4)
            poses_up = lidars.lidar_to_worlds
            times = lidars.times
            log_pose_df = io_utils.read_feather(
                self.av2._data_dir / self.config.sequence / "city_SE3_egovehicle.feather"
            )
            all_ego2w = [convert_pose_dataframe_to_SE3(pose_df) for _, pose_df in log_pose_df.iterrows()]
            # calibration is constant, just use the last one
            assert sweep is not None
            uplidar2ego = sweep.ego_SE3_up_lidar
            all_lup2w = torch.tensor(
                np.array([e2w.compose(uplidar2ego).transform_matrix for e2w in all_ego2w]), dtype=torch.float32
            )
            downlidar2ego = sweep.ego_SE3_down_lidar
            all_ldown2w = torch.tensor(
                np.array([e2w.compose(downlidar2ego).transform_matrix for e2w in all_ego2w]), dtype=torch.float32
            )
            all_times = torch.from_numpy(log_pose_df["timestamp_ns"].to_numpy() / 1e9)

            for i, (pc_up, pc_down, lup2w, ldown2w, time) in enumerate(
                zip(point_clouds_up, point_clouds_down, poses_up, poses_down, times)
            ):
                for j, (pc_, l2w) in enumerate(zip((pc_up, pc_down), (lup2w, ldown2w))):
                    pc = pc_.clone().to(torch.float64)
                    # relative -> absolute time
                    pc[:, -2] += time
                    # project to world frame
                    pc[..., :3] = transform_points(pc[..., :3], l2w.unsqueeze(0).to(pc))
                    # remove ego motion compensation and move to sensor frame
                    if j == 0:  # up_lidar
                        l2ws = all_lup2w
                        lidar_name = "lidar_up"
                    else:  # down_lidar
                        l2ws = all_ldown2w
                        lidar_name = "lidar_down"
                    pc, interpolated_poses = self._remove_ego_motion_compensation(pc, l2ws, all_times)
                    # reset time to relative
                    pc[:, 4] = pc_[:, 4].clone()
                    # transform to common lidar frame again, interpolated_poses is l2w for each point at its timestamp
                    interpolated_poses = torch.matmul(
                        pose_utils.inverse(l2w.unsqueeze(0)).float(), pose_utils.to4x4(interpolated_poses).float()
                    )
                    # move channel from index 5 to 3
                    pc = pc[..., [0, 1, 2, 5, 3, 4]]
                    if j == 1:  # down_lidar
                        pc[..., 3] -= 32
                    # add missing points
                    miss_points = self._get_missing_points(
                        pc,
                        interpolated_poses,
                        lidar_name,
                        dist_cutoff=0.0,
                        outlier_thresh=0.2,
                        ignore_regions=AV2_IGNORE_REGIONS[lidar_name],
                    )
                    if j == 1:  # down_lidar
                        miss_points[..., 3] += 32
                        if not self.config.output_lidars_separately:
                            miss_points[..., :3] = transform_points(
                                miss_points[..., :3], down2up.unsqueeze(0).to(miss_points)
                            )
                    # move channel back from index 3 to 5
                    miss_points = miss_points[..., [0, 1, 2, 4, 5, 3]]
                    if j == 0:  # up_lidar
                        missing_points_up.append(miss_points.float())
                    else:  # down_lidar
                        missing_points_down.append(miss_points.float())
        if self.config.output_lidars_separately:
            if self.config.add_missing_points:
                point_clouds_up = [torch.cat([pc, mp], dim=0) for pc, mp in zip(point_clouds_up, missing_points_up)]
                point_clouds_down = [
                    torch.cat([pc, mp], dim=0) for pc, mp in zip(point_clouds_down, missing_points_down)
                ]
            point_clouds = point_clouds_up + point_clouds_down
        else:
            # transform all points to common lidar frame (up_lidar)
            point_clouds = [
                torch.cat([pc_up, transform_points(pc_down, down2up.unsqueeze(0).to(pc_down))], dim=0)
                for pc_up, pc_down in zip(point_clouds_up, point_clouds_down)
            ]
            if self.config.add_missing_points:
                missing_points = [
                    torch.cat([mp_up, mp_down], dim=0) for mp_up, mp_down in zip(missing_points_up, missing_points_down)
                ]
                point_clouds = [torch.cat([pc, mp], dim=0) for pc, mp in zip(point_clouds, missing_points)]

        point_clouds = [pc.float() for pc in point_clouds]

        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        assert len(lidars) == len(point_clouds)
        if lidars.metadata:
            del lidars.metadata["down2up"]  # we dont need this anymore
            del lidars.metadata["poses_down"]  # we dont need this anymore
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
        trajs = []

        # frames are annotated at the lidar timestamp
        anno_times = self.av2.get_ordered_log_lidar_timestamps(self.config.sequence)
        annos = read_feather(self.av2._data_dir / self.config.sequence / "annotations.feather")
        unique_tracks = annos["track_uuid"].unique()
        all_ego_poses = {t: self.av2.get_city_SE3_ego(self.config.sequence, t).transform_matrix for t in anno_times}

        for track_uuid in unique_tracks:
            rows = annos[annos["track_uuid"] == track_uuid]
            # TODO: how to handle this?
            if len(rows) < 2:
                # only keep tracks that have at least 2 annotations
                continue

            first_row = rows.iloc[0]
            w, l, h = first_row["width_m"], first_row["length_m"], first_row["height_m"]  # noqa: E741
            wlh = np.array([w, l, h], dtype=np.float32)
            label = first_row["category"]
            # remove all objects that are not in the allowed categories
            allowed_cats = ALLOWED_CATEGORIES
            if self.config.include_deformable_actors:
                allowed_cats = allowed_cats.union(DEFORMABLE_CATEGORIES)
            if label not in ALLOWED_CATEGORIES:
                continue
            # check if the actor is symmetric
            symmetric = label in SYMMETRIC_CATEGORIES
            # check if the actor is deformable
            deformable = label in DEFORMABLE_CATEGORIES

            rotations = quat_to_mat(rows.loc[:, ["qw", "qx", "qy", "qz"]].to_numpy())
            # rotate 90 deg around z counter clockwise to get the correct orientation
            rotations = np.matmul(rotations, WLH_TO_LWH)
            translation_xyz_m = rows.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy()
            # get the ego-vehicle poses at the annotation times as the annotations are relative to the ego-vehicle
            ego_poses = np.array([all_ego_poses[t] for t in rows["timestamp_ns"]])
            # get the poses in the ego-vehicle frame
            poses = np.zeros((len(rows), 4, 4), dtype=np.float32)
            poses[:, :3, :3] = rotations
            poses[:, :3, 3] = translation_xyz_m
            poses[:, 3, 3] = 1

            # transform the actor poses to the ego-vehicle frame
            actor_world_poses = np.matmul(ego_poses, poses)

            # check if the actor is stationary, if it has moved more than 0.5m in any direction
            dynamic = np.any(np.std(actor_world_poses[:, :3, 3], axis=0) > 0.5)

            # remove all static objects
            if not dynamic:
                continue

            trajs.append(
                {
                    "poses": torch.tensor(actor_world_poses, dtype=torch.float32),
                    "timestamps": torch.tensor(rows["timestamp_ns"].to_numpy(), dtype=torch.float64) / 1e9,
                    "dims": torch.tensor(wlh, dtype=torch.float32),
                    "label": label,
                    "stationary": not dynamic,
                    "symmetric": symmetric,
                    "deformable": deformable,
                }
            )

        return trajs

    def _generate_dataparser_outputs(self, split="train"):
        datapath = self.config.data / "sensor" / self.config.split
        self.av2 = AV2SensorDataLoader(datapath, datapath)

        assert self.config.sequence in self.av2.get_log_ids(), f"Sequence {self.config.sequence} not found in dataset."
        out = super()._generate_dataparser_outputs(split=split)

        del self.av2
        return out
