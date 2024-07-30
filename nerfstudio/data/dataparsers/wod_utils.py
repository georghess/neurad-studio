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
from __future__ import annotations

import glob
import os
import warnings
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, TypedDict

import dask.dataframe as dd
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import transforms3d
from tqdm import tqdm
from waymo_open_dataset import v2
from waymo_open_dataset.utils import box_utils, transform_utils
from waymo_open_dataset.v2.perception import (
    box as _v2_box,
    camera_image as _v2_camera_image,
    context as _v2_context,
    lidar as _v2_lidar,
    pose as _v2_pose,
)
from waymo_open_dataset.v2.perception.utils.lidar_utils import convert_range_image_to_cartesian

tf.config.set_visible_devices([], "GPU")  # Not useful for parsing data.

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action="ignore", category=FutureWarning)

DATA_FREQUENCY = 10.0  # 10 Hz
DUMMY_DISTANCE_VALUE = 2e3  # meters, used for missing points
TIME_OFFSET = 50e-3  # => 50ms ,time offset in sec; half scanning period


class ActorsDict(TypedDict):
    poses: List[np.ndarray]
    timestamps: List[float]
    label: str


class ImageFrame(TypedDict):
    file_path: str
    transform_matrix: List[List[float]]
    frame_id: int
    time: float
    sensor_id: int
    f_u: float
    f_v: float
    c_u: float
    c_v: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    h: int
    w: int


def get_camera_names():
    return [f"{e.value}:{e.name}" for e in _v2_camera_image.CameraName if e.name != "UNKNOWN"]


def get_mock_timestamps(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Get mock relative timestamps for the wod points."""
    # the wod has x forward, y left, z up and the sweep is split behind the car.
    # it is also rotating clockwise, meaning that the angles close to -pi are the
    # first ones in the sweep and the ones close to pi are the last ones in the sweep.
    angles = -np.arctan2(points[:, 1], points[:, 0])  # N, [-pi, pi]
    # angles += np.pi  # N, [0, 2pi]
    # see how much of the rotation have finished
    fraction_of_rotation = angles / (2 * np.pi)  # N, [0, 1]
    # get the pseudo timestamps based on the total rotation time
    timestamps = fraction_of_rotation * 1.0 / DATA_FREQUENCY
    return timestamps


class ParquetReader:
    def __init__(self, context_name: str, dataset_dir: str = "/data/dataset/wod/training", nb_partitions: int = 120):
        self.context_name = context_name
        self.dataset_dir = dataset_dir
        self.nb_partitions = nb_partitions

    def read(self, tag: str) -> dd.DataFrame:
        """Creates a Dask DataFrame for the component specified by its tag."""
        paths = glob.glob(f"{self.dataset_dir}/{tag}/{self.context_name}.parquet")
        return dd.read_parquet(paths, npartitions=self.nb_partitions)  # type: ignore

    def __call__(self, tag: str) -> dd.DataFrame:
        return self.read(tag)


class SelectedTimestamp:
    def __init__(self, reader: ParquetReader, start_frame: int = 0, end_frame: Optional[int] = None):
        cam_image_df = reader("camera_image")
        cam_image_df = cam_image_df["key.frame_timestamp_micros"]
        self.ts_list = np.unique(np.array(cam_image_df.compute()))
        self.ts_selected = self.ts_list[start_frame:end_frame]
        pass

    def __len__(self) -> int:
        return len(self.ts_selected)

    def sequence_len(self) -> int:
        return len(self.ts_list)

    def get_selected_ts(self) -> List[int]:
        return self.ts_selected.tolist()

    def is_selected(self, ts: int) -> bool:
        return ts in self.ts_selected

    def ts2frame_idx(self, ts: int) -> int:
        if ts not in self.ts_selected:
            raise IndexError(f"{ts} is not in selected timestamps")
        return np.where(self.ts_selected == ts)[0][0]


class ObjectsID:
    """Helper extraction object static/dynamic IDs to be processed by ExportLidar class."""

    def __init__(self, reader: ParquetReader, selected_ts: SelectedTimestamp, speed_static_threshold: float = 0.2):
        self.reader = reader
        self.speed_static_threshold = speed_static_threshold
        self.dynamic_id: list[int] = []
        self.dynamic_uuid: list[str] = []
        self.dynamic_type: list[str] = []
        self.id2uuid: dict[int, str] = {}
        self.uuid2id: dict[str, int] = {}
        self.id2box_dimensions: dict[int, dict[str, float]] = {}
        self.selected_ts = selected_ts
        self.keep_id_after_lidar_extraction = []
        self.build_dict()

    def build_dict(self):
        lidar_box_df = self.reader("lidar_box")

        lidar_box_df2 = (
            lidar_box_df.groupby(["key.segment_context_name", "key.laser_object_id"]).agg(list).reset_index()
        )

        for object_id, (_, r) in enumerate(lidar_box_df2.iterrows()):
            LiDARBoxCom = v2.LiDARBoxComponent.from_dict(r)
            ts_mask = np.isin(np.array(LiDARBoxCom.key.frame_timestamp_micros), self.selected_ts.get_selected_ts())
            if not np.any(ts_mask):
                continue
            dimensions = LiDARBoxCom.box.size

            length, width, height = (
                np.array(dimensions.x)[ts_mask][0],
                np.array(dimensions.y)[ts_mask][0],
                np.array(dimensions.z)[ts_mask][0],
            )

            self.id2box_dimensions[object_id] = {"length": length, "width": width, "height": height}
            object_uuid = LiDARBoxCom.key.laser_object_id
            self.id2uuid[object_id] = object_uuid
            # object is considered static if static in frames selection ( < speed threshold )
            speed = np.array(
                [
                    np.array(LiDARBoxCom.speed.x)[ts_mask],  # type: ignore
                    np.array(LiDARBoxCom.speed.y)[ts_mask],  # type: ignore
                    np.array(LiDARBoxCom.speed.z)[ts_mask],  # type: ignore
                ]
            )
            speed = speed[~np.isnan(speed).any(axis=1)]
            speed = np.linalg.norm(speed, axis=0)
            dynamic = np.any(speed > self.speed_static_threshold)
            if dynamic:
                self.dynamic_id.append(object_id)
                self.dynamic_uuid.append(object_uuid)
                self.dynamic_type.append(_v2_box.BoxType(LiDARBoxCom.type[0]).name)  # type: ignore

        for id, uuid in self.id2uuid.items():
            self.uuid2id[uuid] = id

    def is_dynamic(self, id: int | str):
        if isinstance(id, int):
            return id in self.dynamic_id
        if isinstance(id, str):
            return self.uuid2id[id] in self.dynamic_id

    def get_box_dimensions(self, id: int | str):
        if isinstance(id, int):
            return self.id2box_dimensions[id]
        if isinstance(id, str):
            return self.id2box_dimensions[self.uuid2id[id]]

    def get_box_coordinates(self, dynamic_only: bool = True) -> Dict[str, np.ndarray]:
        lidar_box_df = self.reader("lidar_box")

        lidar_box_df2 = (
            lidar_box_df.groupby(["key.segment_context_name", "key.laser_object_id"]).agg(list).reset_index()
        )

        objects_coordinates = {}
        for object_id, (_, r) in enumerate(lidar_box_df2.iterrows()):
            LiDARBoxCom = v2.LiDARBoxComponent.from_dict(r)
            ts_mask = np.isin(np.array(LiDARBoxCom.key.frame_timestamp_micros), self.selected_ts.get_selected_ts())
            if not np.any(ts_mask):
                continue

            object_uuid = LiDARBoxCom.key.laser_object_id
            object_id = self.uuid2id[object_uuid]

            if dynamic_only:
                if object_id in self.dynamic_id:
                    objects_coordinates[object_id] = LiDARBoxCom.box.center
            else:
                objects_coordinates[object_id] = LiDARBoxCom.box.center

        return objects_coordinates

    def print_dynamic(self):
        for id, type in zip(self.dynamic_id, self.dynamic_type):
            print(f"{id}:{type}, ", end="")


class ExportImages:
    """
    Used to create folder and save image in images, and returns a tuple with:
     - a list of images dict with (image path, frame_id, time, pose (nerf), sensor_id, intrinsic))
     - a tuple of rolling shutter information (duration and direction)

    :param reader: ParquetReader object
    :param select_ts: SelectedTimestamp object
    :param output_folder: Root folder images where will be saved.
    :param cameras_ids: Select which cameras_ids to export, defaults to list(range(1, len(get_camera_names()) + 1))
    """

    IMAGE_FOLDER = "images"

    def __init__(
        self,
        reader: ParquetReader,
        select_ts: SelectedTimestamp,
        output_folder: str,
        cameras_ids: List[int] = list(range(1, len(get_camera_names()) + 1)),
    ):
        self.reader: ParquetReader = reader
        self.select_ts = select_ts
        self.cameras_ids = cameras_ids

        self.output_folder = os.path.join(output_folder, self.IMAGE_FOLDER)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def process(self) -> Tuple[dict[str, List[ImageFrame]], Tuple[float, int]]:
        cam_calib = self.reader("camera_calibration")
        camera_calib = {}
        data_out: dict[str, List[ImageFrame]] = {}

        data_out["frames"] = []
        for i, (_, r) in enumerate(cam_calib.iterrows()):
            calib = v2.CameraCalibrationComponent.from_dict(r)
            camera_calib["cam" + v2.perception.camera_image.CameraName(calib.key.camera_name).name] = (  # type: ignore
                calib.extrinsic.transform.reshape(4, 4)  # type: ignore
            )
            camera_calib["cam" + v2.perception.camera_image.CameraName(calib.key.camera_name).name + "_intrinsics"] = (  # type: ignore
                asdict(calib.intrinsic) | {"h": calib.height, "w": calib.width}  # type: ignore
            )
        # rolling shutter direction for offset calculation
        rolling_shutter_direction = calib.rolling_shutter_direction

        print("Camera processing...")
        cam_image_df = self.reader("camera_image")
        cam_image_df = cam_image_df[
            (cam_image_df["key.camera_name"].isin(self.cameras_ids))  # type: ignore
            & (cam_image_df["key.frame_timestamp_micros"].isin(self.select_ts.get_selected_ts()))  # type: ignore
        ]
        camera_poses = []
        rolling_shutter_list = []
        for i, (_, r) in tqdm(enumerate(cam_image_df.iterrows())):  # type: ignore
            CamComp = v2.CameraImageComponent.from_dict(r)
            tr_image = CamComp.pose.transform.reshape(4, 4)  # type: ignore
            delta_time = (
                CamComp.rolling_shutter_params.camera_readout_done_time
                + CamComp.rolling_shutter_params.camera_trigger_time
            ) / 2 - CamComp.pose_timestamp

            rolling_shutter = (
                CamComp.rolling_shutter_params.camera_readout_done_time
                - CamComp.rolling_shutter_params.camera_trigger_time
            ) / 2
            rolling_shutter_list.append(rolling_shutter)

            avx, avy, avz = (
                CamComp.velocity.angular_velocity.x,
                CamComp.velocity.angular_velocity.y,
                CamComp.velocity.angular_velocity.z,
            )
            skm = np.array([[0, -avz, avy], [avz, 0, -avx], [-avy, avx, 0]])
            r_image = tr_image[:3, :3]

            r_updated = (
                (np.eye(3) + delta_time * skm) @ r_image
            )  # probably another way to do it : R_derivative = skm@r_image; r_image + delta_time * R_derivative
            t_updated = tr_image[:3, 3] + delta_time * np.array(
                [
                    CamComp.velocity.linear_velocity.x,
                    CamComp.velocity.linear_velocity.y,
                    CamComp.velocity.linear_velocity.z,
                ]
            )
            tr_updated = np.eye(4)
            tr_updated[:3, 3] = t_updated
            tr_updated[:3, :3] = r_updated

            frame_id = self.select_ts.ts2frame_idx(CamComp.key.frame_timestamp_micros)
            filename = f"{v2.perception.camera_image.CameraName(CamComp.key.camera_name).name}_{frame_id:08d}.jpg"  # type: ignore

            nerfstudio2waymo = np.eye(4)
            nerfstudio2waymo[:3, :3] = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).T
            # opencv2waymo = np.eye(4)
            # opencv2waymo[:3,:3] = np.array([[0,-1,0],[0,0,-1],[1,0,0]]).T
            calib = camera_calib["cam" + v2.perception.camera_image.CameraName(CamComp.key.camera_name).name]  # type: ignore
            camera_poses.append(tr_updated @ calib @ nerfstudio2waymo)
            data_out["frames"].append(
                {
                    "file_path": os.path.join(self.IMAGE_FOLDER, filename),
                    "transform_matrix": (camera_poses[-1]).tolist(),
                    "frame_id": int(frame_id),
                    "time": delta_time + CamComp.pose_timestamp,
                    "sensor_id": CamComp.key.camera_name - 1,  # sensor_id for NeuRAD, WOD 0 == Unkown
                }
                | camera_calib[
                    "cam" + v2.perception.camera_image.CameraName(CamComp.key.camera_name).name + "_intrinsics"  # type: ignore
                ]
            )

            save_file = os.path.join(self.output_folder, filename)
            if not os.path.exists(save_file):
                with open(save_file, "wb") as binary_file:
                    binary_file.write(CamComp.image)

        # get the mean value for rolling shutter
        rolling_shutter = sum(rolling_shutter_list) / (i + 1)
        return (data_out, (rolling_shutter, rolling_shutter_direction))


class ExportLidar:
    """Utility class for extracting lidar point-cloud and objects from parquet files of WoD v2 dataset."""

    def __init__(
        self,
        reader: ParquetReader,
        select_ts: SelectedTimestamp,
        objects_id: ObjectsID,
        output_folder: str,
        extract_objects=True,
        cameras_ids: List[int] = list(range(1, len(get_camera_names()) + 1)),
    ):
        self.reader: ParquetReader = reader
        self.select_ts = select_ts
        self.cameras_ids = cameras_ids

        self.output_folder = output_folder
        self.extract_objects = extract_objects
        self.objects_id = objects_id
        self.cameras_calibration = None

    def convert_range_image_to_point_cloud(
        self,
        range_image: _v2_lidar.RangeImage,
        calibration: _v2_context.LiDARCalibrationComponent,
        pixel_pose: Optional[_v2_lidar.PoseRangeImage] = None,
        frame_pose: Optional[_v2_pose.VehiclePoseComponent] = None,
        keep_polar_features=False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Converts one range image from polar coordinates to point cloud.
            same as in wod api, but return the mask in addition plus channel id

        Args:
            range_image: One range image return captured by a LiDAR sensor.
            calibration: Parameters for calibration of a LiDAR sensor.
            pixel_pose: If not none, it sets pose for each range image pixel.
            frame_pose: This must be set when `pose` is set.
            keep_polar_features: If true, keep the features from the polar range image
            (i.e. range, intensity, and elongation) as the first features in the
            output range image.

        Returns:
            A 3 [N, D] tensor of 3D LiDAR points. D will be 3 if keep_polar_features is
            False (x, y, z) and 6 if keep_polar_features is True (range, intensity,
            elongation, x, y, z).
                1. Lidar points-cloud
                2. Missing points points-cloud
                3. Range image mask above dummy distance.

        """

        # missing points are found directly from range image
        val_clone = deepcopy(range_image.tensor.numpy())  # type: ignore
        no_return = val_clone[..., 0] == -1  # where range is -1
        val_clone[..., 0][no_return] = DUMMY_DISTANCE_VALUE
        # re-assign the field
        object.__setattr__(range_image, "values", val_clone.flatten())

        # From range image, missing points do not have a pose.
        # So we replace their pose with the vehicle pose.
        # pixel pose & frame pose
        pixel_pose_clone = deepcopy(pixel_pose.tensor.numpy())  # type: ignore
        pixel_pose_mask = pixel_pose_clone[..., 0] == 0
        tr_orig = frame_pose.world_from_vehicle.transform.reshape(4, 4)  # type: ignore
        rot = tr_orig[:3, :3]
        x, y, z = tr_orig[:3, 3]
        yaw, pitch, roll = transforms3d.euler.mat2euler(rot, "szyx")
        # ` [roll, pitch, yaw, x, y, z]`
        pixel_pose_clone[..., 0][pixel_pose_mask] = roll
        pixel_pose_clone[..., 1][pixel_pose_mask] = pitch
        pixel_pose_clone[..., 2][pixel_pose_mask] = yaw
        pixel_pose_clone[..., 3][pixel_pose_mask] = x
        pixel_pose_clone[..., 4][pixel_pose_mask] = y
        pixel_pose_clone[..., 5][pixel_pose_mask] = z
        # re-assign the field
        object.__setattr__(pixel_pose, "values", pixel_pose_clone.flatten())

        range_image_cartesian = convert_range_image_to_cartesian(
            range_image=range_image,
            calibration=calibration,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            keep_polar_features=keep_polar_features,
        )

        range_image_tensor = range_image.tensor
        range_image_mask = DUMMY_DISTANCE_VALUE / 2 > range_image_tensor[..., 0]  # 0  # type: ignore
        points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(range_image_mask))
        missing_points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(~range_image_mask))

        return points_tensor, missing_points_tensor, range_image_mask

    def is_within_box_3d(self, point, box, name=None):
        """Checks whether a point is in a 3d box given a set of points and boxes.

        Args:
            point: [N, 3] tensor. Inner dims are: [x, y, z].
            box: [M, 7] tensor. Inner dims are: [center_x, center_y, center_z, length,
            width, height, heading].
            name: tf name scope.

        Returns:
            point_in_box; [N, M] boolean tensor.

        """

        with tf.compat.v1.name_scope(name, "IsWithinBox3D", [point, box]):
            center = box[:, 0:3]
            dim = box[:, 3:6]
            heading = box[:, 6]
            # [M, 3, 3]
            rotation = transform_utils.get_yaw_rotation(heading)
            # [M, 4, 4]
            transform = transform_utils.get_transform(rotation, center)
            # [M, 4, 4]
            transform = tf.linalg.inv(transform)
            # [M, 3, 3]
            rotation = transform[:, 0:3, 0:3]  # type: ignore
            # [M, 3]
            translation = transform[:, 0:3, 3]  # type: ignore

            # [N, M, 3]
            point_in_box_frame = tf.einsum("nj,mij->nmi", point, rotation) + translation
            # [N, M, 3]
            point_in_box = tf.logical_and(
                tf.logical_and(point_in_box_frame <= dim * 0.5, point_in_box_frame >= -dim * 0.5),
                tf.reduce_all(tf.not_equal(dim, 0), axis=-1, keepdims=True),
            )
            # [N, M]
            point_in_box = tf.cast(
                tf.reduce_prod(input_tensor=tf.cast(point_in_box, dtype=tf.uint8), axis=-1), dtype=tf.bool
            )

            return point_in_box, point_in_box_frame[point_in_box]

    def _load_camera_calibration(self):
        """Loads camera calibration from parquet file to dictionnary."""
        cam_calib_df = self.reader("camera_calibration").compute()
        self.cameras_calibration = {}
        for i, (_, r) in enumerate(cam_calib_df.iterrows()):
            calib = v2.CameraCalibrationComponent.from_dict(r)
            self.cameras_calibration[calib.key.camera_name] = calib

    def process(
        self, objects_id_to_extract: List[int] = []
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float], Dict[int, ActorsDict]]:
        print("Lidar processing...")
        objects_uuid_to_extract = [
            self.objects_id.id2uuid[object_id_to_extract] for object_id_to_extract in objects_id_to_extract
        ]

        self._load_camera_calibration()
        lidar_calib = self.reader("lidar_calibration").compute()

        lidar_df = self.reader("lidar").compute()
        lidar_df = lidar_df[
            (lidar_df["key.laser_name"] == _v2_lidar.LaserName.TOP.value)  # Only lidar TOP is used
            & (lidar_df["key.frame_timestamp_micros"].isin(self.select_ts.get_selected_ts()))
        ]

        lidar_pose_df = self.reader("lidar_pose").compute()

        vehicle_pose_df = self.reader("vehicle_pose").compute()
        vehicle_pose_df = vehicle_pose_df[
            vehicle_pose_df["key.frame_timestamp_micros"].isin(self.select_ts.get_selected_ts())
        ]

        lidar_box_df = self.reader("lidar_box").compute()
        lidar_box_df = lidar_box_df[lidar_box_df["key.frame_timestamp_micros"].isin(self.select_ts.get_selected_ts())]

        pts_lidar_list = []
        missing_pts_list = []
        poses = []
        times = []

        # Neurad actor trajectories
        actors: Dict[int, ActorsDict] = {}

        for i, (_, r) in tqdm(enumerate(lidar_df.iterrows())):
            LidarComp = v2.LiDARComponent.from_dict(r)
            lidar_pose_df_ = lidar_pose_df[
                (lidar_pose_df["key.frame_timestamp_micros"] == LidarComp.key.frame_timestamp_micros)
                & (lidar_pose_df["key.laser_name"] == _v2_lidar.LaserName.TOP.value)
            ]
            LidarPoseComp = v2.LiDARPoseComponent.from_dict(lidar_pose_df_.iloc[0])
            lidar_calib_ = lidar_calib[lidar_calib["key.laser_name"] == _v2_lidar.LaserName.TOP.value]
            LidarCalibComp = v2.LiDARCalibrationComponent.from_dict(lidar_calib_.iloc[0])
            vehicle_pose_df_ = vehicle_pose_df[
                vehicle_pose_df["key.frame_timestamp_micros"] == LidarComp.key.frame_timestamp_micros
            ]
            VehiclePoseCom = v2.VehiclePoseComponent.from_dict(vehicle_pose_df_.iloc[0])

            lidar_box_df_ = lidar_box_df[
                (lidar_box_df["key.frame_timestamp_micros"] == LidarComp.key.frame_timestamp_micros)
                & (lidar_box_df["key.laser_object_id"].isin(self.objects_id.dynamic_uuid))
            ]

            pts_lidar, missing_pts, _ = self.convert_range_image_to_point_cloud(
                LidarComp.range_image_return1,
                LidarCalibComp,
                LidarPoseComp.range_image_return1,
                VehiclePoseCom,
                keep_polar_features=True,
            )
            missing_pts = missing_pts.numpy()

            # compute timestamp for each lidar frame
            time = LidarComp.key.frame_timestamp_micros / 1e6 + TIME_OFFSET  # convert to seconds
            times.append(time)

            timestamps = get_mock_timestamps(pts_lidar[:, 3:6])  # (N, 6)->(..., x,y,z)
            timestamps = np.expand_dims(timestamps, axis=1)

            timestamps_miss = get_mock_timestamps(missing_pts[:, 3:6])  # (N, 6)->(..., x,y,z)
            timestamps_miss = np.expand_dims(timestamps_miss, axis=1)

            pts_lidar = pts_lidar.numpy()
            intensity = pts_lidar[:, 1:2]  # (range, intensity, elongation, x, y, z) => (N, 1)
            intensity = self._normalize(intensity)  # => [0.0, 1.0]

            pts_lidar = np.hstack((pts_lidar[:, 3:6], np.ones((pts_lidar.shape[0], 1))))

            pts_lidar_in_vehicle = pts_lidar
            l2v = LidarCalibComp.extrinsic.transform.reshape(4, 4)  # type: ignore
            pts_lidar_sensor = (np.linalg.inv(l2v) @ pts_lidar_in_vehicle.T).T[:, :3]
            v2w = VehiclePoseCom.world_from_vehicle.transform.reshape(4, 4)  # type: ignore
            l2w = v2w @ l2v

            pts_lidar_world = (v2w @ pts_lidar_in_vehicle.T).T[:, :3]

            lidar_box_df_selected_boxes = lidar_box_df_[
                lidar_box_df_["key.laser_object_id"].isin(objects_uuid_to_extract)
            ]
            for _, lidar_box in lidar_box_df_selected_boxes.iterrows():
                v1_box = tf.transpose(
                    tf.constant(
                        [
                            lidar_box["[LiDARBoxComponent].box.center.x"],
                            lidar_box["[LiDARBoxComponent].box.center.y"],
                            lidar_box["[LiDARBoxComponent].box.center.z"],
                            lidar_box["[LiDARBoxComponent].box.size.x"],
                            lidar_box["[LiDARBoxComponent].box.size.y"],
                            lidar_box["[LiDARBoxComponent].box.size.z"],
                            lidar_box["[LiDARBoxComponent].box.heading"],
                        ],
                        dtype=tf.float32,
                    )
                )
                v1_box = tf.reshape(v1_box, (1, -1))
                v1_box_world = box_utils.transform_box(
                    v1_box,
                    VehiclePoseCom.world_from_vehicle.transform.reshape((4, 4)).astype("float32"),
                    tf.eye(4),  # type: ignore
                )
                mask_object = box_utils.is_within_box_3d(pts_lidar_world[:, :3], v1_box_world).numpy()  # type: ignore
                mask_object = np.any(mask_object, axis=1)

                mean_ts_from_lidar_pts = timestamps[
                    mask_object
                ].mean()  # timestamp of actor is taken from mean of lidar points inside the bbox
                object_timestamp = (
                    time + mean_ts_from_lidar_pts if np.any(mask_object) else time
                )  # If no lidar points in box, timestamp of frame

                # actor pose
                # actor ids
                uuids = lidar_box["key.laser_object_id"]
                actor_id = self.objects_id.uuid2id[uuids]

                # actor type
                type_ = lidar_box["[LiDARBoxComponent].type"]
                type_names = _v2_box.BoxType(type_).name

                tr_object = np.eye(4)
                tr_object[:3, :3] = transforms3d.euler.euler2mat(0, 0, v1_box_world.numpy().ravel()[6])  # type: ignore
                tr_object[:3, 3] = v1_box_world.numpy().ravel()[:3]  # type: ignore

                if actor_id in actors:
                    actors[actor_id]["poses"].append(tr_object)
                    actors[actor_id]["timestamps"].append(object_timestamp)
                else:
                    actors[actor_id] = {"poses": [tr_object], "timestamps": [object_timestamp], "label": type_names}

            pts_lidar = np.hstack((pts_lidar_sensor, intensity, timestamps))  # => (N, 5) == (x, y, z, int, t)
            pts_lidar_list.append(pts_lidar)

            missing_intensity = np.zeros_like(missing_pts[:, 1:2])  # 0 for missing point intensity
            missing_pts_list.append(np.hstack((missing_pts[:, 3:6], missing_intensity, timestamps_miss)))

            poses.append(l2w)

        return poses, pts_lidar_list, missing_pts_list, times, actors

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        max_ = points.max()
        min_ = points.min()

        points = (points - min_) / (max_ - min_)
        return points
