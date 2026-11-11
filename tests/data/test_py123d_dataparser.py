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


def test_lidar_point_times_are_built_per_sensor(make_lidar_ipc_bytes) -> None:
    """Each sensor's points must span the full sweep in firing order."""
    points_per_sensor = (40, 10, 25)
    sweep_duration_s = 0.05
    points = lidar_ipc_to_numpy(
        make_lidar_ipc_bytes(points_per_sensor=points_per_sensor), sweep_duration_s=sweep_duration_s
    )

    assert points.shape[0] == sum(points_per_sensor)
    times = points[:, 4]
    start = 0
    for count in points_per_sensor:
        block = times[start : start + count]
        assert block.min() == pytest.approx(-sweep_duration_s)
        assert block.max() == pytest.approx(0.0)
        assert np.all(np.diff(block) > 0), "times must still advance with firing order within a sensor"
        start += count


def test_lidar_point_times_unchanged_for_a_single_sensor(make_lidar_ipc_bytes) -> None:
    """One sensor, or no ``ids`` column at all, keeps the plain uniform ramp."""
    points = lidar_ipc_to_numpy(make_lidar_ipc_bytes(num_points=8), sweep_duration_s=0.05)
    np.testing.assert_allclose(points[:, 4], np.linspace(-0.05, 0.0, 8), atol=1e-7)


def _make_parser(log_dir: Path, **overrides):
    """Common synthetic-log configuration; callers retain the parser while reading its images."""
    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    config = dict(
        data=log_dir.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=("CAM_FRONT", "CAM_FRONT_LEFT"),
        lidars=("LIDAR_TOP",),
        lidar_type=LidarType.VELODYNE_VLP32C,
        train_split_fraction=1.0,
    )
    config.update(overrides)
    return Py123dDataParser(Py123dDataParserConfig(**config))


def test_py123d_handles_null_sync_references(make_py123d_log) -> None:
    """Missing camera frames and lidar sweeps are skipped while real images stay readable."""
    log_dir = make_py123d_log(null_sync_samples={"CAM_FRONT_LEFT": [2], "lidar": [0]})
    parser = _make_parser(log_dir)
    outputs = parser.get_dataparser_outputs("train")

    # 3 CAM_FRONT frames + 2 CAM_FRONT_LEFT frames, and 2 of 3 lidar sweeps.
    assert len(outputs.image_filenames) == 5
    assert outputs.cameras.shape[0] == 5
    assert all(path.exists() for path in outputs.image_filenames)
    assert outputs.metadata["lidars"].shape[0] == 2
    assert len(outputs.metadata["point_clouds"]) == 2


def test_py123d_image_paths_survive_a_second_split(mini_log: Path) -> None:
    """Paths handed out for one split must still be readable after the next split parses."""
    parser = _make_parser(mini_log, cameras=("CAM_FRONT",), train_split_fraction=0.5)

    train = parser.get_dataparser_outputs("train")
    train_paths = list(train.image_filenames)
    assert train_paths and all(path.exists() for path in train_paths)

    val = parser.get_dataparser_outputs("val")

    assert all(path.exists() for path in train_paths), "train images deleted by the eval parse"
    assert all(path.exists() for path in val.image_filenames)


def test_py123d_georeferenced_poses_keep_their_precision(make_py123d_log) -> None:
    """Recentered camera, lidar and actor poses agree within 1 mm at global-map coordinates."""
    torch = pytest.importorskip("torch")

    def parse(origin):
        log_dir = make_py123d_log(num_samples=6, global_origin=origin)
        return _make_parser(log_dir, load_cuboids=True).get_dataparser_outputs("train")

    at_origin = parse((0.0, 0.0, 0.0))
    georeferenced = parse((3.8e6, 1.2e6, 40.0))

    cam_error = (georeferenced.cameras.camera_to_worlds[:, :3, 3] - at_origin.cameras.camera_to_worlds[:, :3, 3]).abs()
    lidar_error = (
        georeferenced.metadata["lidars"].lidar_to_worlds[:, :3, 3]
        - at_origin.metadata["lidars"].lidar_to_worlds[:, :3, 3]
    ).abs()
    actor_error = (
        georeferenced.metadata["trajectories"][0]["poses"][:, :3, 3]
        - at_origin.metadata["trajectories"][0]["poses"][:, :3, 3]
    ).abs()

    assert float(cam_error.max()) < 1e-3
    assert float(lidar_error.max()) < 1e-3
    assert float(actor_error.max()) < 1e-3

    # dataparser_transform still maps the exporter's global frame into the world frame.
    transform = georeferenced.dataparser_transform.double()
    global_pose = torch.tensor([3.8e6, 1.2e6, 40.0, 1.0], dtype=torch.float64)
    world_pose = transform @ global_pose
    assert float(world_pose.abs().max()) < 10.0, world_pose


def test_py123d_lidar_sensor_offset_is_a_frame_change_only(mini_log: Path) -> None:
    """`lidar_sensor_offset` must move points into the sensor frame without moving the world."""
    torch = pytest.importorskip("torch")

    offset = (0.0, 0.0, 1.5)

    stored_frame = _make_parser(mini_log, cameras=("CAM_FRONT",)).get_dataparser_outputs("train")
    sensor_frame = _make_parser(mini_log, cameras=("CAM_FRONT",), lidar_sensor_offset=offset).get_dataparser_outputs(
        "train"
    )

    offset_tensor = torch.tensor(offset)
    for stored_pc, sensor_pc in zip(stored_frame.metadata["point_clouds"], sensor_frame.metadata["point_clouds"]):
        # The points themselves are shifted into the sensor frame...
        assert torch.allclose(sensor_pc[:, :3], stored_pc[:, :3] - offset_tensor, atol=1e-6)

    def world_points(outputs):
        # Camera 0 is a fixed landmark across the two differently recentered parses.
        lidar_to_worlds = outputs.metadata["lidars"].lidar_to_worlds
        anchor = outputs.cameras.camera_to_worlds[0, :3, 3]
        out = []
        for idx, pc in enumerate(outputs.metadata["point_clouds"]):
            rotation = lidar_to_worlds[idx, :3, :3]
            translation = lidar_to_worlds[idx, :3, 3]
            out.append(pc[:, :3] @ rotation.T + translation - anchor)
        return torch.cat(out)

    assert torch.allclose(world_points(sensor_frame), world_points(stored_frame), atol=1e-5)


@pytest.mark.parametrize("sensor", ["CAM_FRONT", "lidar"])
def test_singleton_sensor_keeps_zero_velocities(make_py123d_log, sensor):
    """Missing sync entries may leave just one frame: keep its pose finite and motion zero."""
    torch = pytest.importorskip("torch")
    log = make_py123d_log(null_sync_samples={sensor: [0, 1]})
    parser = _make_parser(log)
    outputs = parser.get_dataparser_outputs("train")
    sensors = outputs.cameras if sensor == "CAM_FRONT" else outputs.metadata["lidars"]
    sensor_idx = 0 if sensor == "CAM_FRONT" else len(parser.config.cameras)
    mask = (sensors.metadata["sensor_idxs"] == sensor_idx).squeeze(-1)
    assert mask.sum() == 1
    for key in ("velocities", "linear_velocities_local", "angular_velocities_local"):
        torch.testing.assert_close(sensors.metadata[key][mask], torch.zeros(1, 3))
    assert torch.isfinite(outputs.cameras.metadata["velocities"]).all()
    assert torch.isfinite(outputs.metadata["lidars"].metadata["velocities"]).all()
    assert torch.isfinite(outputs.dataparser_transform).all()
    assert torch.isfinite(outputs.cameras.camera_to_worlds).all()
    assert torch.isfinite(outputs.metadata["lidars"].lidar_to_worlds).all()


def _encode_lidar_table(table):
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


@pytest.mark.parametrize("has_timestamps", [True, False])
def test_real_lidar_schema_selects_sensor_and_preserves_time(has_timestamps):
    import pyarrow as pa

    anchor = 1_700_000_000_000_000
    table = pa.table(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [0.0] * 4,
            "z": [0.0] * 4,
            "intensity": pa.array([255] * 4, type=pa.uint8()),
            "ids": [2, 3, 2, 2],
        }
    )
    if has_timestamps:
        table = table.append_column("timestamps", pa.array([anchor + t for t in (10_001, 30_000, 60_002, 90_003)]))
    points = lidar_ipc_to_numpy(
        _encode_lidar_table(table),
        0.1,
        lidar_id=2,
        reference_timestamp_us=anchor + 50_000,
        start_timestamp_us=anchor,
    )
    np.testing.assert_array_equal(points[:, 0], [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(points[:, 3], 1.0)
    np.testing.assert_array_equal(points[:, 5], -1.0)  # unknown channel, not a fabricated ring
    expected = [-0.039999, 0.010002, 0.040003] if has_timestamps else [-0.05, 0.0, 0.05]
    np.testing.assert_allclose(points[:, 4], expected, rtol=0, atol=1e-8)


def test_sensor_selection_requires_ids_for_merged_cloud(make_lidar_ipc_bytes):
    import pyarrow as pa

    table = pa.ipc.open_stream(make_lidar_ipc_bytes()).read_all().drop(["ids"])
    with pytest.raises(ValueError, match="ids"):
        lidar_ipc_to_numpy(_encode_lidar_table(table), 0.1, lidar_id=2)


@pytest.mark.parametrize("points_frame", ["imu", "sensor"])
def test_lidar_export_preserves_world_points_and_capture_times(make_py123d_log, points_frame):
    import pyarrow as pa
    import torch
    from py123d.datatypes.sensors.lidar import LidarID, LidarMergedMetadata, LidarMetadata
    from py123d.geometry.pose import PoseSE3
    from py123d_fixtures import _write_arrow

    log = make_py123d_log()
    # A real merged export has distinct sensor IDs, IMU-frame xyz, point timestamps, and no ring field.
    metadata = LidarMergedMetadata(
        {
            LidarID.LIDAR_TOP: LidarMetadata(
                "LIDAR_TOP", LidarID.LIDAR_TOP, PoseSE3.from_list([1.0, 2.0, 0.5, 2**-0.5, 0.0, 0.0, 2**-0.5])
            ),
            LidarID.LIDAR_FRONT: LidarMetadata("LIDAR_FRONT", LidarID.LIDAR_FRONT),
        }
    )
    path = log / "lidar.lidar_merged.arrow"
    table = read_arrow_table(path)
    starts = table["lidar.lidar_merged.timestamp_us"].to_pylist()
    xyz = np.array([[8.0, 3.0, 1.0], [99.0, 99.0, 1.0], [10.0, 4.0, 1.0]])
    if points_frame == "sensor":
        extrinsic = metadata.lidar_metadatas[LidarID.LIDAR_TOP].lidar_to_imu_se3.transformation_matrix
        xyz = (xyz - extrinsic[:3, 3]) @ extrinsic[:3, :3]
    payloads = []
    for start in starts:
        cloud = pa.table(
            {
                "x": xyz[:, 0],
                "y": xyz[:, 1],
                "z": xyz[:, 2],
                "intensity": pa.array([128] * 3, type=pa.uint8()),
                "ids": [2, 3, 2],
                "timestamps": [start + 10_001, start + 30_000, start + 40_003],
            }
        )
        payloads.append(_encode_lidar_table(cloud))
    table = table.set_column(2, "lidar.lidar_merged.data", pa.array(payloads))
    _write_arrow(path, table, metadata.to_dict())
    # The cloud starts before its compensated pose's reference time, as in the official exporter.
    ego_path = log / "ego_state_se3.arrow"
    ego = read_arrow_table(ego_path)
    ego = ego.set_column(0, "ego_state_se3.timestamp_us", pa.array([s + 25_000 for s in starts]))
    with pa.ipc.new_file(ego_path, ego.schema) as writer:
        writer.write_table(ego)

    parser = _make_parser(
        log, min_lidar_dist=(0.0, 0.0, 0.0), lidar_points_frame=points_frame, lidar_sensor_offset=(0.0, 0.0, 0.25)
    )
    outputs = parser.get_dataparser_outputs("train")
    lidars = outputs.metadata["lidars"]
    assert lidars.assume_ego_compensated
    for i, points in enumerate(outputs.metadata["point_clouds"]):
        assert len(points) == 2
        np.testing.assert_allclose(points[:, 4].numpy(), [-0.014999, 0.015003], rtol=0, atol=1e-8)
        assert float(lidars.times[i]) + outputs.time_offset == pytest.approx((starts[i] + 25_000) / 1e6, abs=1e-6)
        actual = points[:, :3] @ lidars.lidar_to_worlds[i, :3, :3].T + lidars.lidar_to_worlds[i, :3, 3]
        world = torch.tensor([[8.0 + i, 3.0, 1.0], [10.0 + i, 4.0, 1.0]])
        transform = outputs.dataparser_transform
        expected = world @ transform[:, :3].T + transform[:, 3]
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)


@pytest.mark.parametrize("is_undistorted", [True, False])
def test_camera_distortion_respects_preprocessed_images(mini_log, is_undistorted):
    """Calibration coefficients must not cause already-undistorted images to be corrected twice."""
    import torch
    from py123d_fixtures import _write_arrow

    mapping = build_camera_name_to_modality_key(mini_log)
    path = mini_log / f"{mapping['CAM_FRONT']}.arrow"
    metadata = get_pinhole_camera_metadata(path).to_dict()
    metadata["is_undistorted"] = is_undistorted
    _write_arrow(path, read_arrow_table(path), metadata)

    parser = _make_parser(mini_log, cameras=("CAM_FRONT",))
    cameras = parser.get_dataparser_outputs("train").cameras
    expected = torch.tensor([0.0] * 6 if is_undistorted else [-0.1, 0.05, 0.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(cameras.distortion_params, expected.expand(len(cameras), -1))
