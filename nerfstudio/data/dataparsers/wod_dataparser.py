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
"""Data parser for Waymo Open Dataset"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
import transforms3d
from waymo_open_dataset.v2.perception import camera_image

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import DUMMY_DISTANCE_VALUE, ADDataParser, ADDataParserConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.wod_utils import ExportImages, ExportLidar, ObjectsID, ParquetReader, SelectedTimestamp
from nerfstudio.data.utils.lidar_elevation_mappings import WOD64_ELEVATION_MAPPING

WOD_ELEVATION_MAPPING = {"Wod64": WOD64_ELEVATION_MAPPING}
WOD_AZIMUT_RESOLUTION = {"Wod64": 0.140625}
WOD_SKIP_ELEVATION_CHANNELS = {"Wod64": ()}

HORIZONTAL_BEAM_DIVERGENCE = 2.4e-3  # radians
VERTICAL_BEAM_DIVERGENCE = 1.5e-3  # radians

ALLOWED_DEFORMABLE_CLASSES = (
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)

ALLOWED_RIGID_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_SIGN",
)
WOD_CAMERA_NAME_2_ID = {e.name: e.value for e in camera_image.CameraName if e.name != "UNKNOWN"}


@dataclass
class WoDParserConfig(ADDataParserConfig):
    """Waymo Open Dataset config."""

    _target: Type = field(default_factory=lambda: WoD)
    """target class to instantiate"""
    sequence: str = "10588771936253546636_2300_000_2320_000"
    """Name of the scene (ie: so-called context_name)."""
    data: Path = Path("/data/dataset/wod/")
    """Raw dataset path to WOD"""
    parquet_dir: str = "training"
    """Change to validation when some sequence is in validation"""
    output_folder: Path = Path("/data/dataset/wod/images")
    """Output saving folder for images, by defaut it will be set with wod dataset path."""
    train_split_fraction: float = 0.5
    """The percent of images to use for training. The remaining images are for eval."""
    start_frame: int = 0
    """Start frame"""
    end_frame: Optional[int] = None
    """End frame. When set to known end frame will be the last one."""
    dataset_end_fraction: float = 1.0
    """At what fraction of the dataset to end. Different value than 1.0 not supported with current implementation of wod dataset."""
    cameras: Tuple[Literal["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"], ...] = (
        "FRONT",
        "FRONT_LEFT",
        "FRONT_RIGHT",
        # "SIDE_LEFT",
        # "SIDE_RIGHT",
    )
    """Which cameras to use."""
    lidars: Tuple[Literal["Top"], ...] = ("Top",)
    """Which lidars to use, only lidar TOP is supported."""
    load_cuboids: bool = True
    """Whether to load cuboid annotations."""
    cuboids_ids: Optional[Tuple[int, ...]] = None
    """Selection of cuboids_ids if cuboid_annotations is set to True. If None, all dynamic cuboids will be exported."""
    annotation_interval: float = 0.1  # 10 Hz of capture
    """Interval between annotations in seconds."""
    correct_cuboid_time: bool = True
    """Whether to correct the cuboid time to match the actual time of observation, not the end of the lidar sweep."""
    min_lidar_dist: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Wod Top lidar is x-forward, y-left, z-up."""
    add_missing_points: bool = True
    """Whether to add missing points (rays that did not return) to the point clouds."""
    lidar_elevation_mapping: Dict[str, Dict] = field(default_factory=lambda: WOD_ELEVATION_MAPPING)
    """Elevation mapping for each lidar."""
    skip_elevation_channels: Dict[str, Tuple] = field(default_factory=lambda: WOD_SKIP_ELEVATION_CHANNELS)
    """Channels to skip when adding missing points."""
    lidar_azimuth_resolution: Dict[str, float] = field(default_factory=lambda: WOD_AZIMUT_RESOLUTION)
    """Azimuth resolution for each lidar."""
    rolling_shutter_time: float = 0.044
    """The rolling shutter time for the cameras (seconds)."""
    time_to_center_pixel: float = 0.0
    """The time offset for the center pixel, relative to the image timestamp (seconds)."""
    paint_points: bool = True
    """Whether to paint the points in the point cloud."""


@dataclass
class WoD(ADDataParser):
    """Waymo Open Dataset DatasetParser"""

    config: WoDParserConfig

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        """Images are exported from parquet files to jpg in the dataset folder, and filepaths are returns with Cameras."""

        output_folder_name = f"{self.config.sequence}_start{self.config.start_frame}_end{self.config.end_frame}"
        output_folder_name += "_cameras_" + "_".join([str(id) for id in self.cameras_ids])
        images_output_folder: Path = Path(self.config.output_folder) / output_folder_name  # type: ignore

        export_images = ExportImages(
            self.parquet_reader,
            output_folder=str(images_output_folder),
            select_ts=self.select_ts,
            cameras_ids=self.cameras_ids,
        )

        data_out, (rolling_shutter_half, rolling_shutter_direction) = export_images.process()
        rolling_shutter = round(rolling_shutter_half * 2, 3)

        self.config.rolling_shutter_time = rolling_shutter
        self.config.time_to_center_pixel = 0.0
        rs_direction = "Horizontal" if rolling_shutter_direction in (2, 4) else "Vertical"
        rs_direction = "Horizontal_reversed" if rolling_shutter_direction == 4 else rs_direction
        img_filenames = []
        intrinsics = []
        poses = []
        idxs = []
        heights = []
        widths = []
        times = []
        for frame in data_out["frames"]:
            img_filenames.append(str(images_output_folder / frame["file_path"]))
            poses.append(frame["transform_matrix"])
            intrinsic = np.array(
                [
                    [frame["f_u"], 0, frame["c_u"]],
                    [0, frame["f_v"], frame["c_v"]],
                    [0, 0, 1],
                ]
            )
            intrinsics.append(intrinsic)
            idxs.append(frame["sensor_id"])
            heights.append(frame["h"])
            widths.append(frame["w"])
            times.append(frame["time"])

        intrinsics = torch.tensor(np.array(intrinsics), dtype=torch.float32)
        poses = torch.tensor(np.array(poses), dtype=torch.float32)
        times = torch.tensor(times, dtype=torch.float64)
        idxs = torch.tensor(idxs).int().unsqueeze(-1)
        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=torch.tensor(heights),
            width=torch.tensor(widths),
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={
                "sensor_idxs": idxs,
                "rs_direction": rs_direction,
            },
        )
        return cameras, img_filenames

    def _get_lidars(self) -> Tuple[Lidars, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """The implementation of _get_lidar for WoD dataset actually returns directly tensors for pts_lidar and pts_missing, while
        other dataparser provide links to files containing the point-cloud which are then processed with _read_lidar function in
        _generate_dataparser_output. With WoD all lidar point-cloud are stored in parquet files, and points cloud are eventually
        stored in memory in DataParserOutput object. So most of the job is done within _get_lidars function.

        :return: Tuple[Lidars, Tuple[List[Point-clouds],List[MissingPointsPcd]]]
        """
        if self.config.load_cuboids:
            objects_id_to_extract = (
                list(self.config.cuboids_ids) if self.config.cuboids_ids is not None else self.objects_id.dynamic_id
            )
        else:
            objects_id_to_extract = []

        export_lidar = ExportLidar(self.parquet_reader, self.select_ts, self.objects_id, self.config.output_folder)
        poses, pts_lidar_list, missing_pts_list, times, actors = export_lidar.process(
            objects_id_to_extract=objects_id_to_extract
        )

        # save actors for later trajectories calculation
        self.actors = actors

        pts_lidar_list = [torch.from_numpy(pts) for pts in pts_lidar_list]
        missing_pts_list = [torch.from_numpy(pts) for pts in missing_pts_list]

        times = torch.tensor(times, dtype=torch.float64)
        idxs = torch.zeros_like(times).int().unsqueeze(-1)

        poses = torch.from_numpy(np.array(poses))
        lidars = Lidars(
            lidar_to_worlds=poses[:, :3, :4],
            lidar_type=LidarType.WOD64,
            times=times,
            metadata={"sensor_idxs": idxs},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
            valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
        )
        return lidars, (pts_lidar_list, missing_pts_list)

    def _read_lidars(
        self, lidars: Lidars, pts_list_tuple: Tuple[List[torch.Tensor], List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Reads the point clouds from the given filenames. Should be in x,y,z,r,t order. t is optional."""

        pts_lidar_list, missing_pts_list = pts_list_tuple
        if self.config.add_missing_points:
            """Currently this part has been done during wod_export, here we only concatenate together.
               For future modification, refer to _read_lidars method from pandaset_dataparser.py
            """
            point_clouds = [torch.cat([pc, missing], dim=0) for pc, missing in zip(pts_lidar_list, missing_pts_list)]
        else:
            point_clouds = pts_lidar_list

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
        trajs_list = []
        allowed_classes = ALLOWED_RIGID_CLASSES
        if self.config.include_deformable_actors:
            allowed_classes += ALLOWED_DEFORMABLE_CLASSES

        rot_minus_90 = np.eye(4)
        rot_minus_90[:3, :3] = transforms3d.euler.euler2mat(0.0, 0.0, -np.pi / 2)

        for index, actor in self.actors.items():
            actor_type = actor["label"]

            if actor_type not in allowed_classes:
                continue
            poses = np.array(actor["poses"]) @ rot_minus_90
            timestamps = actor["timestamps"]
            actor_dimensions = self.objects_id.id2box_dimensions[index]  # (length, width, height)
            lenght, width, height = actor_dimensions.values()
            dims = np.array([width, lenght, height], dtype=np.float32)

            symmetric = actor_type == "TYPE_VEHICLE"
            deformable = actor_type in ALLOWED_DEFORMABLE_CLASSES

            trajs_list.append(
                {
                    "poses": torch.tensor(poses).float(),
                    "timestamps": torch.tensor(timestamps, dtype=torch.float64),
                    "dims": torch.tensor(dims, dtype=torch.float32),
                    "label": actor_type,
                    "stationary": False,  # Only 'export' dynamic objects from ExportLidar
                    "symmetric": symmetric,
                    "deformable": deformable,
                }
            )
        return trajs_list

    def _generate_dataparser_outputs(self, split="train") -> DataparserOutputs:
        assert (
            self.config.dataset_end_fraction == 1.0
        ), f"Wod data parser only support dataset_end_fraction == 1.0, value received {self.config.dataset_end_fraction}"
        self.cameras_ids = [WOD_CAMERA_NAME_2_ID[cam] for cam in self.config.cameras]
        parquet_dir = str(self.config.data / self.config.parquet_dir)
        self.parquet_reader = ParquetReader(self.config.sequence, dataset_dir=parquet_dir)
        self.select_ts = SelectedTimestamp(self.parquet_reader, self.config.start_frame, self.config.end_frame)
        self.objects_id = ObjectsID(self.parquet_reader, self.select_ts)

        return super()._generate_dataparser_outputs(split)


if __name__ == "__main__":
    wod_test = WoD(config=WoDParserConfig())
    do = wod_test._generate_dataparser_outputs()
    print(do)
