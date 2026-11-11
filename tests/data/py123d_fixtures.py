# Copyright 2026 the authors of NeuRAD and contributors.
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

"""Synthetic py123d Arrow log for dataparser unit / smoke tests."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import msgpack
import numpy as np
import pyarrow as pa
from PIL import Image
from py123d.api.utils.arrow_metadata_utils import parse_log_directory_metadata
from py123d.datatypes.custom.custom_modality import CustomModalityMetadata
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.base_camera import CameraID
from py123d.datatypes.sensors.lidar import LidarID, LidarMergedMetadata, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeDistortion, PinholeIntrinsics
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry.pose import PoseSE3

MINI_LOG_ID = "mini_log_001"
MINI_SPLIT = "train"
NUM_SAMPLES = 3
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48
CAMERAS: Tuple[Tuple[str, CameraID], ...] = (
    ("CAM_FRONT", CameraID.PCAM_F0),
    ("CAM_FRONT_LEFT", CameraID.PCAM_L0),
)
LIDAR_NAME = "LIDAR_TOP"
IDENTITY_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def _pose(x: float, y: float = 0.0, z: float = 0.0) -> List[float]:
    return [x, y, z, 1.0, 0.0, 0.0, 0.0]


def _write_arrow(path: Path, table: pa.Table, metadata_dict: Dict) -> None:
    schema = table.schema.with_metadata({"metadata": msgpack.packb(metadata_dict, use_bin_type=True)})
    table = table.replace_schema_metadata(schema.metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            writer.write_table(table)


def _jpeg_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _lidar_ipc_bytes(num_points: int = 32, points_per_sensor: Optional[Sequence[int]] = None) -> bytes:
    """Encode a cloud with sensor IDs, optionally concatenating several sensors' point blocks."""
    if points_per_sensor is not None:
        num_points = int(sum(points_per_sensor))
        sensor_ids = np.concatenate(
            [np.full(count, sensor_id, dtype=np.uint8) for sensor_id, count in enumerate(points_per_sensor)]
        )
    else:
        sensor_ids = np.zeros(num_points, dtype=np.uint8)
    xyz = np.stack(
        [
            np.linspace(1.0, 10.0, num_points, dtype=np.float32),
            np.zeros(num_points, dtype=np.float32),
            np.ones(num_points, dtype=np.float32),
        ],
        axis=1,
    )
    intensity = np.full(num_points, 128, dtype=np.uint8)
    channel = np.zeros(num_points, dtype=np.uint16)
    table = pa.table(
        {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "intensity": intensity,
            "channel": channel,
            "ids": sensor_ids,
        }
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def build_mini_py123d_log(data_root: Path, *, num_samples: int = NUM_SAMPLES) -> Path:
    """Write a tiny synthetic py123d log under ``data_root/logs/{split}/{log_id}/``.

    Returns the log directory path. Clears py123d's log-metadata LRU cache so the
    new files are visible to subsequent ``parse_log_directory_metadata`` calls.
    """
    log_dir = data_root / "logs" / MINI_SPLIT / MINI_LOG_ID
    log_dir.mkdir(parents=True, exist_ok=True)

    base_ts = 1_700_000_000_000_000
    timestamps = [base_ts + i * 500_000 for i in range(num_samples)]

    camera_meta: Dict[str, PinholeCameraMetadata] = {}
    for camera_name, camera_id in CAMERAS:
        camera_meta[camera_name] = PinholeCameraMetadata(
            camera_name=camera_name,
            camera_id=camera_id,
            intrinsics=PinholeIntrinsics(fx=50.0, fy=50.0, cx=IMAGE_WIDTH / 2, cy=IMAGE_HEIGHT / 2, skew=0.0),
            distortion=PinholeDistortion(k1=-0.1, k2=0.05, p1=0.0, p2=0.0, k3=0.0),
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            camera_to_imu_se3=PoseSE3.from_list([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        )

    # Cameras
    for camera_name, metadata in camera_meta.items():
        modality = metadata.modality_key
        rows_ts = []
        rows_data = []
        rows_pose = []
        for i, ts in enumerate(timestamps):
            rows_ts.append(ts)
            rows_data.append(_jpeg_bytes(hash((camera_name, i)) & 0xFFFF))
            # Camera pose in world: translate with ego.
            rows_pose.append(_pose(float(i)))
        table = pa.table(
            {
                f"{modality}.timestamp_us": pa.array(rows_ts, type=pa.int64()),
                f"{modality}.data": pa.array(rows_data, type=pa.binary()),
                f"{modality}.camera_to_global_se3": pa.array(rows_pose, type=pa.list_(pa.float64(), 7)),
            }
        )
        _write_arrow(log_dir / f"{modality}.arrow", table, metadata.to_dict())

    # Ego
    ego_meta = EgoStateSE3Metadata(
        vehicle_name="test_vehicle",
        width=1.8,
        length=4.5,
        height=1.5,
        wheel_base=2.7,
        center_to_imu_se3=PoseSE3.from_list([1.2, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0]),
        rear_axle_to_imu_se3=PoseSE3.from_list(IDENTITY_POSE),
    )
    ego_table = pa.table(
        {
            "ego_state_se3.timestamp_us": pa.array(timestamps, type=pa.int64()),
            "ego_state_se3.imu_se3": pa.array(
                [_pose(float(i)) for i in range(num_samples)], type=pa.list_(pa.float64(), 7)
            ),
            "ego_state_se3.dynamic_state_se3": pa.array(
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(num_samples)],
                type=pa.list_(pa.float64(), 9),
            ),
        }
    )
    _write_arrow(log_dir / "ego_state_se3.arrow", ego_table, ego_meta.to_dict())

    # Lidar (merged modality file; channel name is user-facing LIDAR_TOP)
    lidar_meta = LidarMergedMetadata(
        {
            LidarID.LIDAR_MERGED: LidarMetadata(
                lidar_name=LIDAR_NAME,
                lidar_id=LidarID.LIDAR_MERGED,
                lidar_to_imu_se3=PoseSE3.from_list([0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0]),
            )
        }
    )
    lidar_table = pa.table(
        {
            "lidar.lidar_merged.timestamp_us": pa.array(timestamps, type=pa.int64()),
            "lidar.lidar_merged.end_timestamp_us": pa.array([ts + 50_000 for ts in timestamps], type=pa.int64()),
            "lidar.lidar_merged.data": pa.array(
                [_lidar_ipc_bytes(24 + i) for i in range(num_samples)], type=pa.binary()
            ),
        }
    )
    _write_arrow(log_dir / "lidar.lidar_merged.arrow", lidar_table, lidar_meta.to_dict())

    # Capture sidecar: per-camera offsets from the shared anchor.
    capture_meta = CustomModalityMetadata(
        modality_id="capture_metadata",
        metadata={"schema_version": 1, "description": "synthetic capture timestamps"},
    )
    capture_payloads = []
    for i, ts in enumerate(timestamps):
        payload = {
            "schema_version": 1,
            "cameras": {
                "CAM_FRONT": {"capture_timestamp_us": ts + 10_000},
                "CAM_FRONT_LEFT": {"capture_timestamp_us": ts + 25_000},
            },
        }
        capture_payloads.append(msgpack.packb(payload, use_bin_type=True))
    capture_table = pa.table(
        {
            "custom.capture_metadata.timestamp_us": pa.array(timestamps, type=pa.int64()),
            "custom.capture_metadata.data": pa.array(capture_payloads, type=pa.binary()),
        }
    )
    _write_arrow(log_dir / "custom.capture_metadata.arrow", capture_table, capture_meta.to_dict())

    # Boxes: one moving vehicle track (xy moves > 0.5 m std across samples).
    box_meta = BoxDetectionsSE3Metadata(DefaultBoxDetectionLabel)
    boxes: List[List[List[float]]] = []
    tokens: List[List[str]] = []
    labels: List[List[int]] = []
    velocities: List[List[List[float]]] = []
    num_pts: List[List[int]] = []
    for i in range(num_samples):
        # length, width, height after xyz + quat
        box = [float(i) * 2.0, 5.0, 0.5, 1.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.5]
        boxes.append([box])
        tokens.append(["track_vehicle_0"])
        labels.append([int(DefaultBoxDetectionLabel.VEHICLE)])
        velocities.append([[2.0, 0.0, 0.0]])
        num_pts.append([10])
    box_table = pa.table(
        {
            "box_detections_se3.timestamp_us": pa.array(timestamps, type=pa.int64()),
            "box_detections_se3.bounding_box_se3": pa.array(boxes, type=pa.list_(pa.list_(pa.float64(), 10))),
            "box_detections_se3.track_token": pa.array(tokens, type=pa.list_(pa.string())),
            "box_detections_se3.label": pa.array(labels, type=pa.list_(pa.uint16())),
            "box_detections_se3.velocity_3d": pa.array(velocities, type=pa.list_(pa.list_(pa.float64(), 3))),
            "box_detections_se3.num_lidar_points": pa.array(num_pts, type=pa.list_(pa.int32())),
        }
    )
    _write_arrow(log_dir / "box_detections_se3.arrow", box_table, box_meta.to_dict())

    # Sync
    sync_cols: Dict[str, Sequence] = {
        "sync.uuid": [uuid.uuid4().bytes for _ in range(num_samples)],
        "sync.timestamp_us": timestamps,
        "box_detections_se3": list(range(num_samples)),
        "ego_state_se3": list(range(num_samples)),
        "lidar.lidar_merged": list(range(num_samples)),
        "custom.capture_metadata": list(range(num_samples)),
    }
    for metadata in camera_meta.values():
        sync_cols[metadata.modality_key] = list(range(num_samples))

    arrays = {
        "sync.uuid": pa.array(sync_cols["sync.uuid"], type=pa.uuid()),
        "sync.timestamp_us": pa.array(sync_cols["sync.timestamp_us"], type=pa.int64()),
    }
    for key, values in sync_cols.items():
        if key in arrays:
            continue
        arrays[key] = pa.array(values, type=pa.int64())
    sync_table = pa.table(arrays)
    log_meta = LogMetadata(
        dataset="example",
        split=MINI_SPLIT,
        log_name=MINI_LOG_ID,
        location="test",
        version="0.0.1",
        map_metadata=None,
    )
    _write_arrow(log_dir / "sync.arrow", sync_table, log_meta.to_dict())

    parse_log_directory_metadata.cache_clear()
    return log_dir
