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

"""Data parser for py123d Arrow logs."""

from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import (
    DUMMY_DISTANCE_VALUE,
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
)
from nerfstudio.data.dataparsers.py123d_utils import (
    DEFAULT_CAPTURE_METADATA_MODALITY,
    WLH_TO_LWH,
    bounding_box_se3_from_list,
    box_label_to_category,
    build_camera_name_to_modality_key,
    find_lidar_entry,
    get_box_detections_metadata,
    get_camera_capture_timestamp_us,
    get_lidar_merged_metadata,
    get_pinhole_camera_metadata,
    is_label_prefix_allowed,
    iter_sync_rows,
    lidar_ipc_to_numpy,
    lidar_row_path,
    load_capture_metadata_table,
    modality_data_column,
    modality_end_timestamp_column,
    modality_pose_column,
    modality_timestamp_column,
    parse_lidar_row_path,
    pinhole_distortion_to_neurad,
    pose_list_to_matrix,
    read_arrow_table,
)

# Reflectance is assumed normalized to [0, 1] in the decoded lidar IPC.
MAX_REFLECTANCE_VALUE = 1.0


@dataclass
class Py123dDataParserConfig(ADDataParserConfig):
    """py123d dataset config.

    Reads Apache Arrow logs produced by exporters that follow the py123d log layout.
    Camera names and lidar names are user-supplied values from the Arrow schema metadata.
    """

    _target: Type = field(default_factory=lambda: Py123dDataParser)
    data: Path = Path("data/py123d")
    """Root directory containing ``logs/{split}/{log_id}/``."""
    log_id: str = ""
    """Log directory name under ``{data}/logs/{split}/``."""
    split: str = "train"
    """Split subdirectory under ``{data}/logs/``."""
    cameras: Tuple[str, ...] = ()
    """Camera names from Arrow metadata (e.g. channel names stored in ``camera_name``)."""
    lidars: Tuple[str, ...] = ()
    """Lidar names from Arrow metadata (e.g. ``LIDAR_TOP``).

    Currently only a single name is supported: it selects one entry from the
    merged ``lidar.lidar_merged`` modality. Passing multiple names raises.
    """
    lidar_type: LidarType = LidarType.VELODYNE_VLP32C
    """Lidar model used for rendering metadata.

    Defaults to neurad's ``Lidars`` default (``VELODYNE_VLP32C``). Override to match
    the sensor that produced your export.
    """
    load_cuboids: bool = False
    """Whether to load dynamic actor trajectories from box detections."""
    include_deformable_actors: bool = True
    """Whether to include deformable classes (e.g. person) from the allowlist."""
    allowed_rigid_classes: Tuple[str, ...] = (
        "vehicle",
        "two_wheeler",
        "train",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
        "trailer",
    )
    """Prefix allowlist for rigid actors. Matched against default-taxonomy label names."""
    allowed_deformable_classes: Tuple[str, ...] = ("person", "human", "pedestrian")
    """Prefix allowlist for deformable actors. Matched against default-taxonomy label names."""
    add_missing_points: bool = False
    """Disable missing-point synthesis until elevation maps are validated."""
    annotation_interval: float = 0.5
    """Approximate sync interval between samples (seconds)."""
    rolling_shutter_time: float = 0.03
    """Rolling shutter duration for cameras (seconds). Set to 0.0 to disable."""
    # Same public defaults as ZOD / PandaSet dataparsers (radians at 1 m).
    horizontal_beam_divergence: float = 3e-3
    """Horizontal lidar beam divergence used for rendering metadata."""
    vertical_beam_divergence: float = 1.5e-3
    """Vertical lidar beam divergence used for rendering metadata."""
    use_capture_timestamps: bool = True
    """Use per-camera capture timestamps from an optional custom modality sidecar."""
    capture_metadata_modality: str = DEFAULT_CAPTURE_METADATA_MODALITY
    """Custom modality key for capture metadata (e.g. ``custom.capture_metadata``).

    When ``use_capture_timestamps`` is True, the dataparser loads
    ``{modality}.arrow`` if present and uses ``cameras[name].capture_timestamp_us``.
    Missing sidecar / camera entries fall back to the camera-table anchor timestamp.
    """

    def __post_init__(self) -> None:
        if self.log_id:
            self.sequence = self.log_id
        # Allow empty cameras/lidars at registration time (dataparser_configs.py /
        # tyro defaults). Sensors must be supplied on the CLI before loading data.
        if not (self.cameras or self.lidars or self.radars):
            assert self.annotation_interval > 1e-6, "Child classes must specify the annotation interval"
            assert self.dataset_start_fraction >= 0.0, "Dataset start fraction must be >= 0.0"
            assert self.dataset_end_fraction <= 1.0, "Dataset end fraction must be <= 1.0"
            assert self.dataset_start_fraction < self.dataset_end_fraction, "Dataset start must be < dataset end"
            return
        super().__post_init__()


@dataclass
class Py123dDataParser(ADDataParser):
    """py123d dataset parser."""

    config: Py123dDataParserConfig

    def __post_init__(self) -> None:
        self._log_dir: Path = Path()
        self._sync_table = None
        self._ego_table = None
        self._camera_tables: Dict[str, object] = {}
        self._camera_name_to_modality: Dict[str, str] = {}
        self._camera_metadata: Dict[str, object] = {}
        self._lidar_table = None
        self._lidar_modality_key = "lidar.lidar_merged"
        self._lidar_to_imu = np.eye(4)
        self._box_table = None
        self._box_label_class = None
        self._capture_table = None
        self._image_tmpdir = None
        self._image_dir: Path = Path()
        self._image_paths: Dict[Tuple[str, int], Path] = {}

    @property
    def actor_transform(self) -> torch.Tensor:
        """Convert from neurad actor frame (x-right) to py123d/nuScenes (x-forward)."""
        return torch.from_numpy(WLH_TO_LWH).float()[:3, :]

    def _generate_dataparser_outputs(self, split="train"):
        self._load_arrow_tables()
        self._extract_camera_images()
        return super()._generate_dataparser_outputs(split)

    def _load_arrow_tables(self) -> None:
        log_dir = self.config.data / "logs" / self.config.split / self.config.log_id
        if not log_dir.exists():
            raise FileNotFoundError(f"py123d log not found: {log_dir}")
        self._log_dir = log_dir

        self._sync_table = read_arrow_table(log_dir / "sync.arrow")
        self._ego_table = read_arrow_table(log_dir / "ego_state_se3.arrow")
        self._camera_name_to_modality = build_camera_name_to_modality_key(log_dir)

        missing_cameras = [name for name in self.config.cameras if name not in self._camera_name_to_modality]
        if missing_cameras:
            available = sorted(self._camera_name_to_modality.keys())
            raise KeyError(f"Configured cameras not found in log metadata: {missing_cameras}. Available: {available}")

        for camera_name in self.config.cameras:
            modality_key = self._camera_name_to_modality[camera_name]
            arrow_path = log_dir / f"{modality_key}.arrow"
            self._camera_tables[modality_key] = read_arrow_table(arrow_path)
            self._camera_metadata[camera_name] = get_pinhole_camera_metadata(arrow_path)

        if self.config.lidars:
            if len(self.config.lidars) != 1:
                raise ValueError(
                    "py123d-data currently supports exactly one lidar name selecting an entry "
                    f"from lidar.lidar_merged; got {list(self.config.lidars)}"
                )
            lidar_path = log_dir / f"{self._lidar_modality_key}.arrow"
            self._lidar_table = read_arrow_table(lidar_path)
            merged_metadata = get_lidar_merged_metadata(lidar_path)
            _, lidar_entry = find_lidar_entry(merged_metadata, self.config.lidars[0])
            self._lidar_to_imu = pose_list_to_matrix(lidar_entry.lidar_to_imu_se3.array)

        if self.config.load_cuboids:
            box_path = log_dir / "box_detections_se3.arrow"
            if not box_path.exists():
                raise FileNotFoundError(f"load_cuboids=True but box file not found: {box_path}")
            try:
                box_metadata = get_box_detections_metadata(box_path)
            except ValueError as exc:
                raise ValueError(
                    f"Could not resolve box detection label class from {box_path}. "
                    "The label enum referenced in Arrow schema metadata must be importable "
                    "(e.g. DefaultBoxDetectionLabel, or a dataset-specific enum on PYTHONPATH)."
                ) from exc
            self._box_label_class = box_metadata.box_detection_label_class
            self._box_table = read_arrow_table(box_path)

        if self.config.use_capture_timestamps:
            self._capture_table = load_capture_metadata_table(log_dir, self.config.capture_metadata_modality)

        # Keep TemporaryDirectory alive for the parser lifetime so images remain
        # readable, then clean up automatically when the parser is discarded.
        self._image_tmpdir = tempfile.TemporaryDirectory(prefix="py123d_images_")
        self._image_dir = Path(self._image_tmpdir.name)

    def _extract_camera_images(self) -> None:
        """Extract image payloads for rows referenced by ``sync.arrow``."""
        for camera_name in self.config.cameras:
            modality_key = self._camera_name_to_modality[camera_name]
            table = self._camera_tables[modality_key]
            data_column = modality_data_column(table)
            needed_rows = {
                int(self._sync_table.column(modality_key)[sync_row].as_py())
                for sync_row in iter_sync_rows(self._sync_table)
            }
            for row_idx in sorted(needed_rows):
                image_bytes = data_column[row_idx].as_py()
                if not image_bytes:
                    raise ValueError(
                        f"Missing image payload for camera '{camera_name}' "
                        f"(modality '{modality_key}', row {row_idx})"
                    )
                image_path = self._image_dir / f"{modality_key.replace('.', '_')}_{row_idx:05d}.jpg"
                image_path.write_bytes(image_bytes)
                self._image_paths[(modality_key, row_idx)] = image_path

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        filenames: List[Path] = []
        times: List[float] = []
        intrinsics: List[np.ndarray] = []
        distortions: List[np.ndarray] = []
        poses: List[np.ndarray] = []
        idxs: List[int] = []
        heights: List[int] = []
        widths: List[int] = []

        for cam_idx, camera_name in enumerate(self.config.cameras):
            modality_key = self._camera_name_to_modality[camera_name]
            metadata = self._camera_metadata[camera_name]
            table = self._camera_tables[modality_key]
            pose_column = modality_pose_column(table)
            timestamp_column = modality_timestamp_column(table)
            assert pose_column is not None

            for sync_row in iter_sync_rows(self._sync_table):
                row_idx = int(self._sync_table.column(modality_key)[sync_row].as_py())
                pose = pose_list_to_matrix(pose_column[row_idx].as_py())
                pose[:3, :3] = pose[:3, :3] @ OPENCV_TO_NERFSTUDIO

                anchor_ts_us = timestamp_column[row_idx].as_py()
                capture_ts_us = None
                if self.config.use_capture_timestamps:
                    capture_ts_us = get_camera_capture_timestamp_us(
                        self._capture_table,
                        self._sync_table,
                        self.config.capture_metadata_modality,
                        sync_row,
                        camera_name,
                    )
                times.append((capture_ts_us if capture_ts_us is not None else anchor_ts_us) / 1e6)

                filenames.append(self._image_paths[(modality_key, row_idx)])
                intrinsics.append(
                    np.array(
                        [
                            [metadata.intrinsics.fx, 0.0, metadata.intrinsics.cx],
                            [0.0, metadata.intrinsics.fy, metadata.intrinsics.cy],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    )
                )
                distortions.append(pinhole_distortion_to_neurad(metadata.distortion))
                poses.append(pose)
                idxs.append(cam_idx)
                heights.append(metadata.height)
                widths.append(metadata.width)

        intrinsics_tensor = torch.tensor(np.array(intrinsics), dtype=torch.float32)
        distortion_tensor = torch.tensor(np.array(distortions), dtype=torch.float32)
        poses_tensor = torch.tensor(np.array(poses), dtype=torch.float32)
        times_tensor = torch.tensor(times, dtype=torch.float64)
        idxs_tensor = torch.tensor(idxs, dtype=torch.int32).unsqueeze(-1)

        cameras = Cameras(
            fx=intrinsics_tensor[:, 0, 0],
            fy=intrinsics_tensor[:, 1, 1],
            cx=intrinsics_tensor[:, 0, 2],
            cy=intrinsics_tensor[:, 1, 2],
            height=torch.tensor(heights, dtype=torch.int32),
            width=torch.tensor(widths, dtype=torch.int32),
            distortion_params=distortion_tensor,
            camera_to_worlds=poses_tensor[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=times_tensor,
            metadata={"sensor_idxs": idxs_tensor},
        )
        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        filenames: List[Path] = []
        poses: List[np.ndarray] = []
        times: List[float] = []
        idxs: List[int] = []

        ego_pose_column = None
        for name in self._ego_table.column_names:
            if name.endswith(".imu_se3"):
                ego_pose_column = self._ego_table.column(name)
                break
        assert ego_pose_column is not None

        timestamp_column = modality_timestamp_column(self._lidar_table)

        ego_column_name = "ego_state_se3"
        for lidar_idx, _lidar_name in enumerate(self.config.lidars):
            for sync_row in iter_sync_rows(self._sync_table):
                row_idx = int(self._sync_table.column(self._lidar_modality_key)[sync_row].as_py())
                ego_row_idx = int(self._sync_table.column(ego_column_name)[sync_row].as_py())
                ego_pose = pose_list_to_matrix(ego_pose_column[ego_row_idx].as_py())
                lidar_pose = ego_pose @ self._lidar_to_imu

                filenames.append(lidar_row_path(row_idx))
                poses.append(lidar_pose)
                times.append(timestamp_column[row_idx].as_py() / 1e6)
                idxs.append(lidar_idx)

        poses_tensor = torch.tensor(np.array(poses), dtype=torch.float32)
        times_tensor = torch.tensor(times, dtype=torch.float64)
        idxs_tensor = torch.tensor(idxs, dtype=torch.int32).unsqueeze(-1)

        lidars = Lidars(
            lidar_to_worlds=poses_tensor[:, :3, :4],
            lidar_type=self.config.lidar_type,
            assume_ego_compensated=False,
            times=times_tensor,
            metadata={"sensor_idxs": idxs_tensor},
            horizontal_beam_divergence=self.config.horizontal_beam_divergence,
            vertical_beam_divergence=self.config.vertical_beam_divergence,
            valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
        )
        return lidars, filenames

    def _read_lidars(self, lidars: Lidars, filepaths: List[Path]) -> List[torch.Tensor]:
        point_clouds: List[torch.Tensor] = []
        data_column = modality_data_column(self._lidar_table)
        timestamp_column = modality_timestamp_column(self._lidar_table)
        end_timestamp_column = modality_end_timestamp_column(self._lidar_table)

        for filepath, lidar_time in zip(filepaths, lidars.times.squeeze(-1)):
            row_idx = parse_lidar_row_path(filepath)
            start_us = timestamp_column[row_idx].as_py()
            end_us = end_timestamp_column[row_idx].as_py() if end_timestamp_column is not None else start_us
            sweep_duration_s = max((end_us - start_us) / 1e6, 0.0)

            # lidar_ipc_to_numpy already returns relative sweep time offsets in
            # [-sweep_duration, 0]; do not subtract absolute lidar timestamps.
            pc = torch.from_numpy(lidar_ipc_to_numpy(data_column[row_idx].as_py(), sweep_duration_s))
            point_clouds.append(pc)

        return point_clouds

    def _get_actor_trajectories(self) -> List[Dict]:
        """Load dynamic actor trajectories from ``box_detections_se3.arrow``.

        Boxes are grouped by ``track_token``, filtered by the configured class-prefix
        allowlist (matched against py123d default-taxonomy names via ``to_default()``),
        and converted to neurad's actor frame with ``WLH_TO_LWH``. Stationary tracks
        (xy std < 0.5 m) are dropped, matching the nuScenes parser.
        """
        assert self._box_table is not None and self._box_label_class is not None

        timestamp_col = self._box_table.column("box_detections_se3.timestamp_us")
        boxes_col = self._box_table.column("box_detections_se3.bounding_box_se3")
        tokens_col = self._box_table.column("box_detections_se3.track_token")
        labels_col = self._box_table.column("box_detections_se3.label")

        trajs: Dict[str, List[Dict]] = defaultdict(list)
        for row_idx in range(self._box_table.num_rows):
            timestamp_s = timestamp_col[row_idx].as_py() / 1e6
            boxes = boxes_col[row_idx].as_py()
            tokens = tokens_col[row_idx].as_py()
            labels = labels_col[row_idx].as_py()
            for box_values, track_token, label_value in zip(boxes, tokens, labels):
                if not track_token:
                    continue
                box = bounding_box_se3_from_list(box_values)
                pose = box.center_se3.transformation_matrix @ WLH_TO_LWH
                # neurad dims are documented as wlh; py123d stores length/width/height.
                wlh = np.array([box.width, box.length, box.height], dtype=np.float32)
                category = box_label_to_category(self._box_label_class, int(label_value))
                trajs[track_token].append(
                    {
                        "pose": pose.astype(np.float32),
                        "wlh": wlh,
                        "label": category,
                        "time": timestamp_s,
                    }
                )

        return self._traj_dict_to_list(trajs)

    def _traj_dict_to_list(self, traj: Dict[str, List[Dict]]) -> List[Dict]:
        """Convert per-track observation lists into neurad trajectory dicts."""
        allowed: Set[str] = set(self.config.allowed_rigid_classes)
        if self.config.include_deformable_actors:
            allowed.update(self.config.allowed_deformable_classes)

        traj_out: List[Dict] = []
        for track_token, traj_list in traj.items():
            if len(traj_list) < 2:
                continue
            label = traj_list[0]["label"]
            if not is_label_prefix_allowed(label, allowed):
                continue

            poses = torch.from_numpy(np.stack([t["pose"] for t in traj_list]).astype(np.float32))
            times = torch.from_numpy(np.array([t["time"] for t in traj_list], dtype=np.float64))
            dims = torch.from_numpy(np.stack([t["wlh"] for t in traj_list]).astype(np.float32))
            dims = dims.max(0).values

            dynamic = bool((poses[:, :2, 3].std(dim=0) > 0.50).any())
            if not dynamic:
                continue

            deformable = is_label_prefix_allowed(label, set(self.config.allowed_deformable_classes))
            traj_out.append(
                {
                    "uuid": track_token,
                    "label": label,
                    "poses": poses,
                    "timestamps": times,
                    "dims": dims,
                    "stationary": False,
                    "symmetric": not deformable,
                    "deformable": deformable,
                }
            )
        return traj_out
