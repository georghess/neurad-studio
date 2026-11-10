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

"""Utilities for reading py123d Arrow logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple, Type, Union

import msgpack
import numpy as np
import pyarrow as pa
from py123d.api.utils.arrow_metadata_utils import get_metadata_from_arrow_schema, parse_log_directory_metadata
from py123d.datatypes.detections.box_detection_label import BoxDetectionLabel
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata, camera_metadata_from_dict
from py123d.datatypes.sensors.lidar import LidarMergedMetadata, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeDistortion
from py123d.geometry.bounding_box import BoundingBoxSE3
from py123d.geometry.pose import PoseSE3

LIDAR_ROW_PATH_PREFIX = "__py123d_lidar_row__"
DEFAULT_CAPTURE_METADATA_MODALITY = "custom.capture_metadata"

# py123d / nuScenes actor frame is x-forward, y-left, z-up. Neurad uses x-right,
# y-forward, z-up, so rotate actor poses by 90 degrees about z (same as nuScenes).
WLH_TO_LWH = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def read_arrow_table(path: Path) -> pa.Table:
    """Read all record batches from a py123d Arrow IPC file."""
    reader = pa.ipc.open_file(path)
    return reader.read_all()


def pose_list_to_matrix(pose_values: Sequence[float]) -> np.ndarray:
    """Convert a py123d ``[x, y, z, qw, qx, qy, qz]`` pose to a 4x4 matrix."""
    return PoseSE3.from_list(list(pose_values)).transformation_matrix


def pinhole_distortion_to_neurad(
    distortion: Optional[Union[PinholeDistortion, Sequence[float]]],
) -> np.ndarray:
    """Map py123d pinhole distortion to neurad's 6-element Brown model."""
    if distortion is None:
        return np.zeros(6, dtype=np.float32)
    if isinstance(distortion, PinholeDistortion):
        coeffs = [distortion.k1, distortion.k2, distortion.p1, distortion.p2, distortion.k3]
    else:
        coeffs = list(distortion)
        while len(coeffs) < 5:
            coeffs.append(0.0)
    k1, k2, p1, p2, k3 = coeffs[:5]
    return np.array([k1, k2, k3, 0.0, p1, p2], dtype=np.float32)


def build_camera_name_to_modality_key(log_dir: Path) -> Dict[str, str]:
    """Map ``camera_name`` values (e.g. ``CAM_FRONT``) to modality keys (e.g. ``camera.pcam_f0``)."""
    log_metadata = parse_log_directory_metadata(log_dir)
    mapping: Dict[str, str] = {}
    for modality_key, metadata in log_metadata.modality_metadatas.items():
        if not isinstance(metadata, BaseCameraMetadata):
            continue
        mapping[metadata.camera_name] = modality_key
    return mapping


def get_pinhole_camera_metadata(arrow_path: Path) -> PinholeCameraMetadata:
    """Load and validate pinhole camera metadata from an Arrow file schema."""
    schema = pa.ipc.open_file(arrow_path).schema
    metadata = camera_metadata_from_dict(msgpack.unpackb(schema.metadata[b"metadata"], raw=False))
    if not isinstance(metadata, PinholeCameraMetadata):
        raise TypeError(f"Only pinhole cameras are supported; got {type(metadata).__name__} for '{arrow_path.name}'")
    return metadata


def get_lidar_merged_metadata(arrow_path: Path) -> LidarMergedMetadata:
    """Load merged lidar metadata from an Arrow file schema."""
    schema = pa.ipc.open_file(arrow_path).schema
    return get_metadata_from_arrow_schema(schema, LidarMergedMetadata)


def find_lidar_entry(merged_metadata: LidarMergedMetadata, lidar_name: str) -> Tuple[str, LidarMetadata]:
    """Return the metadata entry for a configured lidar name."""
    for lidar_id, entry in merged_metadata.lidar_metadatas.items():
        if entry.lidar_name == lidar_name:
            return str(lidar_id), entry
    available = [entry.lidar_name for entry in merged_metadata.lidar_metadatas.values()]
    raise KeyError(f"Lidar '{lidar_name}' not found in merged metadata. Available: {available}")


def decode_lidar_ipc(data: bytes) -> np.ndarray:
    """Decode embedded lidar IPC bytes to ``[x, y, z, intensity, channel]``."""
    table = pa.ipc.open_stream(data).read_all()
    x = table.column("x").to_numpy().astype(np.float32, copy=False)
    y = table.column("y").to_numpy().astype(np.float32, copy=False)
    z = table.column("z").to_numpy().astype(np.float32, copy=False)
    intensity = table.column("intensity").to_numpy().astype(np.float32) / 255.0
    channel = table.column("channel").to_numpy().astype(np.float32)
    return np.stack([x, y, z, intensity, channel], axis=1)


def lidar_ipc_to_numpy(
    data: bytes,
    sweep_duration_s: float,
) -> np.ndarray:
    """Decode lidar IPC bytes to neurad array format ``[x, y, z, intensity, time, channel]``."""
    points = decode_lidar_ipc(data)
    num_points = points.shape[0]
    if num_points == 0:
        return np.zeros((0, 6), dtype=np.float32)
    time_offsets = np.linspace(-sweep_duration_s, 0.0, num_points, dtype=np.float32)
    return np.concatenate([points[:, :4], time_offsets[:, None], points[:, 4:5]], axis=1).astype(np.float32)


def modality_data_column(table: pa.Table) -> pa.ChunkedArray:
    """Return the payload column for a py123d modality table."""
    for name in table.column_names:
        if name.endswith(".data"):
            return table.column(name)
    raise KeyError(f"No '.data' column found in table with columns: {table.column_names}")


def modality_pose_column(table: pa.Table) -> Optional[pa.ChunkedArray]:
    """Return the global pose column for a camera modality table, if present."""
    for name in table.column_names:
        if name.endswith(".camera_to_global_se3"):
            return table.column(name)
    return None


def modality_timestamp_column(table: pa.Table) -> pa.ChunkedArray:
    """Return the timestamp column for a py123d modality table."""
    for name in table.column_names:
        if name.endswith(".timestamp_us"):
            return table.column(name)
    raise KeyError(f"No '.timestamp_us' column found in table with columns: {table.column_names}")


def modality_end_timestamp_column(table: pa.Table) -> Optional[pa.ChunkedArray]:
    """Return the sweep end timestamp column for a lidar modality table, if present."""
    for name in table.column_names:
        if name.endswith(".end_timestamp_us"):
            return table.column(name)
    return None


def lidar_row_path(row_idx: int) -> Path:
    """Create a synthetic path token that encodes a lidar row index."""
    return Path(f"{LIDAR_ROW_PATH_PREFIX}{row_idx}")


def parse_lidar_row_path(path: Path) -> int:
    """Parse a synthetic lidar row path created by :func:`lidar_row_path`."""
    return int(path.name.removeprefix(LIDAR_ROW_PATH_PREFIX))


def iter_sync_rows(sync_table: pa.Table) -> Iterable[int]:
    """Iterate row indices present in ``sync.arrow``."""
    return range(sync_table.num_rows)


def get_box_detections_metadata(arrow_path: Path) -> BoxDetectionsSE3Metadata:
    """Load box-detection schema metadata, including the label enum class."""
    schema = pa.ipc.open_file(arrow_path).schema
    return get_metadata_from_arrow_schema(schema, BoxDetectionsSE3Metadata)


def bounding_box_se3_from_list(values: Sequence[float]) -> BoundingBoxSE3:
    """Convert a py123d 10-float box array to ``BoundingBoxSE3``."""
    return BoundingBoxSE3.from_array(np.asarray(values, dtype=np.float64))


def box_label_to_category(label_class: Type[BoxDetectionLabel], label_value: int) -> str:
    """Map a stored label enum int to a lowercase default-taxonomy category string.

    Dataset-specific enums are converted via ``to_default()`` so the public dataparser
    can filter with generic prefixes like ``vehicle`` / ``person``.
    """
    return label_class(label_value).to_default().name.lower()


def is_label_prefix_allowed(label: str, allowed_prefixes: Set[str]) -> bool:
    """Return True if ``label`` equals or starts with any allowed prefix.

    Examples:
        ``vehicle.emergency`` matches ``vehicle``; ``two_wheeler`` matches ``two_wheeler``.
    """
    label = label.lower()
    for prefix in allowed_prefixes:
        prefix = prefix.lower()
        if label == prefix or label.startswith(prefix + ".") or label.startswith(prefix + "_"):
            return True
    return False


def decode_capture_metadata_payload(data: Any) -> Dict[str, Any]:
    """Decode a capture-metadata row payload from msgpack bytes, JSON bytes/str, or dict."""
    if isinstance(data, dict):
        return data
    if isinstance(data, (bytes, bytearray)):
        try:
            return msgpack.unpackb(data, raw=False)
        except Exception:
            return json.loads(data.decode("utf-8"))
    if isinstance(data, str):
        return json.loads(data)
    raise TypeError(f"Unsupported capture metadata payload type: {type(data)}")


def load_capture_metadata_table(log_dir: Path, modality_key: str) -> Optional[pa.Table]:
    """Load an optional capture-metadata custom modality table, or ``None`` if absent."""
    if not modality_key:
        return None
    path = log_dir / f"{modality_key}.arrow"
    if not path.exists():
        return None
    return read_arrow_table(path)


def get_camera_capture_timestamp_us(
    capture_table: Optional[pa.Table],
    sync_table: pa.Table,
    modality_key: str,
    sync_row: int,
    camera_name: str,
) -> Optional[int]:
    """Return per-camera ``capture_timestamp_us`` from the sidecar, or ``None`` to fall back.

    Looks up the capture-metadata row via ``sync_table[modality_key][sync_row]``, decodes
    the payload, and reads ``payload["cameras"][camera_name]["capture_timestamp_us"]``.
    """
    if capture_table is None or modality_key not in sync_table.column_names:
        return None
    capture_row = sync_table.column(modality_key)[sync_row].as_py()
    if capture_row is None:
        return None
    data_column = modality_data_column(capture_table)
    payload = decode_capture_metadata_payload(data_column[int(capture_row)].as_py())
    cameras = payload.get("cameras") or {}
    entry = cameras.get(camera_name)
    if not entry:
        return None
    capture_ts = entry.get("capture_timestamp_us")
    return int(capture_ts) if capture_ts is not None else None
