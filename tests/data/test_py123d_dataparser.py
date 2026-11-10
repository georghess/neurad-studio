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

# Optional imports must follow the importorskip guard.
# ruff: noqa: E402

from pathlib import Path

import numpy as np
import pytest

# py123d is an optional dependency; skip these tests when it is not installed.
pytest.importorskip("py123d", reason="py123d is an optional dependency for these tests")

from nerfstudio.data.dataparsers.py123d_utils import (
    DEFAULT_CAPTURE_METADATA_MODALITY,
    WLH_TO_LWH,
    bounding_box_se3_from_list,
    build_camera_name_to_modality_key,
    decode_capture_metadata_payload,
    get_camera_capture_timestamp_us,
    get_pinhole_camera_metadata,
    is_label_prefix_allowed,
    lidar_ipc_to_numpy,
    load_capture_metadata_table,
    pinhole_distortion_to_neurad,
    pose_list_to_matrix,
    read_arrow_table,
)


def test_pinhole_distortion_to_neurad() -> None:
    coeffs = pinhole_distortion_to_neurad([-0.4, 0.22, 0.0007, 0.0004, -0.096])
    assert coeffs.tolist() == pytest.approx([-0.4, 0.22, -0.096, 0.0, 0.0007, 0.0004])


def test_pose_list_to_matrix_is_valid_rigid_transform() -> None:
    matrix = pose_list_to_matrix([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    assert matrix.shape == (4, 4)
    assert matrix[3, 3] == 1.0
    rotation = matrix[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)


def test_is_label_prefix_allowed() -> None:
    allowed = {"vehicle", "person", "two_wheeler"}
    assert is_label_prefix_allowed("vehicle", allowed)
    assert is_label_prefix_allowed("vehicle.emergency", allowed)
    assert is_label_prefix_allowed("person", allowed)
    assert is_label_prefix_allowed("two_wheeler", allowed)
    assert not is_label_prefix_allowed("animal", allowed)
    assert not is_label_prefix_allowed("other", allowed)


def test_bounding_box_se3_from_list_and_wlh_rotation() -> None:
    box = bounding_box_se3_from_list([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.5])
    assert box.length == pytest.approx(4.0)
    assert box.width == pytest.approx(2.0)
    assert box.height == pytest.approx(1.5)
    pose = box.center_se3.transformation_matrix @ WLH_TO_LWH
    assert pose.shape == (4, 4)
    assert np.allclose(pose[:3, 3], [1.0, 2.0, 3.0])


def test_decode_capture_metadata_payload_dict_passthrough() -> None:
    payload = {"cameras": {"CAM_FRONT": {"capture_timestamp_us": 123}}}
    assert decode_capture_metadata_payload(payload) is payload


def test_mini_capture_metadata_sidecar(mini_log: Path) -> None:
    capture = load_capture_metadata_table(mini_log, DEFAULT_CAPTURE_METADATA_MODALITY)
    assert capture is not None
    sync = read_arrow_table(mini_log / "sync.arrow")
    ts = get_camera_capture_timestamp_us(
        capture, sync, DEFAULT_CAPTURE_METADATA_MODALITY, sync_row=0, camera_name="CAM_FRONT"
    )
    assert ts is not None
    assert (
        get_camera_capture_timestamp_us(
            capture, sync, DEFAULT_CAPTURE_METADATA_MODALITY, sync_row=0, camera_name="MISSING"
        )
        is None
    )
    assert load_capture_metadata_table(mini_log, "custom.does_not_exist") is None


def test_mini_log_camera_metadata_and_lidar_decode(mini_log: Path) -> None:
    mapping = build_camera_name_to_modality_key(mini_log)
    assert "CAM_FRONT" in mapping

    metadata = get_pinhole_camera_metadata(mini_log / f"{mapping['CAM_FRONT']}.arrow")
    assert metadata.camera_name == "CAM_FRONT"
    assert metadata.intrinsics.fx > 0

    lidar_table = read_arrow_table(mini_log / "lidar.lidar_merged.arrow")
    data = lidar_table.column(2)[0].as_py()
    points = lidar_ipc_to_numpy(data, sweep_duration_s=0.05)
    assert points.shape[1] == 6
    assert points.shape[0] > 0


def test_py123d_dataparser_smoke(mini_log: Path) -> None:
    torch = pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    config = Py123dDataParserConfig(
        data=mini_log.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=("CAM_FRONT", "CAM_FRONT_LEFT"),
        lidars=("LIDAR_TOP",),
        lidar_type=LidarType.VELODYNE_VLP32C,
        train_split_fraction=1.0,
    )
    parser = Py123dDataParser(config)
    outputs = parser.get_dataparser_outputs("train")

    assert len(outputs.image_filenames) == 3 * len(config.cameras)
    assert outputs.cameras.shape[0] == len(outputs.image_filenames)
    assert outputs.metadata["lidars"].shape[0] == 3
    assert len(outputs.metadata["point_clouds"]) == 3
    assert outputs.metadata["point_clouds"][0].shape[1] == 6
    assert outputs.cameras.distortion_params is not None
    assert torch.all(outputs.cameras.distortion_params[:, 3] == 0)


def test_py123d_rejects_multiple_lidars(mini_log: Path) -> None:
    pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    config = Py123dDataParserConfig(
        data=mini_log.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=("CAM_FRONT",),
        lidars=("LIDAR_TOP", "LIDAR_REAR"),
        lidar_type=LidarType.VELODYNE_VLP32C,
        train_split_fraction=1.0,
    )
    with pytest.raises(ValueError, match="exactly one lidar name"):
        Py123dDataParser(config).get_dataparser_outputs("train")


def test_py123d_actor_trajectories(mini_log: Path) -> None:
    torch = pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    config = Py123dDataParserConfig(
        data=mini_log.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=("CAM_FRONT",),
        lidars=("LIDAR_TOP",),
        lidar_type=LidarType.VELODYNE_VLP32C,
        load_cuboids=True,
        train_split_fraction=1.0,
    )
    parser = Py123dDataParser(config)
    outputs = parser.get_dataparser_outputs("train")

    trajectories = outputs.metadata["trajectories"]
    assert len(trajectories) > 0
    traj = trajectories[0]
    assert traj["poses"].ndim == 3 and traj["poses"].shape[-2:] == (4, 4)
    assert traj["timestamps"].ndim == 1
    assert traj["dims"].shape == (3,)
    assert traj["label"] == "vehicle"
    assert traj["stationary"] is False
    assert traj["deformable"] is False
    assert torch.is_floating_point(traj["poses"])


def test_py123d_per_camera_capture_timestamps(mini_log: Path) -> None:
    torch = pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    cameras = ("CAM_FRONT", "CAM_FRONT_LEFT")
    common = dict(
        data=mini_log.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=cameras,
        lidars=("LIDAR_TOP",),
        lidar_type=LidarType.VELODYNE_VLP32C,
        train_split_fraction=1.0,
    )

    with_capture = Py123dDataParser(
        Py123dDataParserConfig(**common, use_capture_timestamps=True)
    ).get_dataparser_outputs("train")
    without_capture = Py123dDataParser(
        Py123dDataParserConfig(**common, use_capture_timestamps=False)
    ).get_dataparser_outputs("train")

    n = 3
    capture_f = with_capture.cameras.times[:n].squeeze(-1)
    capture_l = with_capture.cameras.times[n : 2 * n].squeeze(-1)
    anchor_f = without_capture.cameras.times[:n].squeeze(-1)
    anchor_l = without_capture.cameras.times[n : 2 * n].squeeze(-1)

    assert not torch.allclose(capture_f, capture_l)
    assert torch.allclose(anchor_f, anchor_l)
    assert not torch.allclose(capture_f, anchor_f)
    delta_ms = (capture_f - anchor_f).abs() * 1e3
    assert float(delta_ms.mean()) == pytest.approx(10.0, abs=0.1)
