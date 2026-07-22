# py123d data (`py123d-data`)

Train NeuRAD / SplatAD on [py123d](https://github.com/kesai-labs/py123d) Apache Arrow
driving logs. The dataparser reads a public log layout and takes camera / lidar names
from config — no dataset-specific sensor defaults are baked in.

| | |
|---|---|
| CLI subcommand | `py123d-data` |
| Implementation | [`py123d_dataparser.py`](../nerfstudio/data/dataparsers/py123d_dataparser.py), [`py123d_utils.py`](../nerfstudio/data/dataparsers/py123d_utils.py) |
| Extra dependency | [`py123d`](https://pypi.org/project/py123d/) (`pip install py123d`) |
| Example config | [`docs/examples/py123d_example.yaml`](./examples/py123d_example.yaml) |

## Expected log layout

Point `--data` at a root that contains `logs/{split}/{log_id}/`:

```text
{data}/
  logs/
    {split}/
      {log_id}/
        sync.arrow
        ego_state_se3.arrow
        box_detections_se3.arrow          # optional; needed for --load-cuboids True
        camera.{modality_key}.arrow       # one file per camera modality
        lidar.{modality_key}.arrow        # e.g. lidar.lidar_merged.arrow
        custom.capture_metadata.arrow     # optional per-camera capture times
```

Camera **names** passed on the CLI (e.g. `CAM_FRONT`) are resolved via Arrow schema
metadata (`camera_name`) to modality keys (e.g. `camera.pcam_f0`). Lidar names (e.g.
`LIDAR_TOP`) resolve the same way.

### Supported cameras

Only **pinhole** cameras (`PinholeCameraMetadata`) are supported today. Distortion is
mapped to neurad's 6-element Brown model and undistorted by the image datamanager.

### Optional capture-time sidecar

When `--use-capture-timestamps True` (default), the dataparser loads
`custom.capture_metadata.arrow` if present and uses per-camera
`capture_timestamp_us` values. Missing sidecar or camera entries fall back to the
camera-table anchor timestamp.

Payload shape (MsgPack / JSON inside the custom modality):

```json
{
  "schema_version": 1,
  "cameras": {
    "CAM_FRONT": { "capture_timestamp_us": 1234567890 }
  }
}
```

Override the modality key with `--capture-metadata-modality` if your exporter uses a
different custom id (file name is `{modality}.arrow`).

## Quickstart

```bash
pip install py123d

# Inspect CLI options
python nerfstudio/scripts/train.py splatad py123d-data --help

# Smoke train (replace paths / sensor names with your export)
TORCHDYNAMO_DISABLE=1 python nerfstudio/scripts/train.py splatad \
  --max-num-iterations 100 \
  --vis tensorboard \
  py123d-data \
  --data data/py123d \
  --log-id my_log_001 \
  --split train \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
  --lidars LIDAR_TOP \
  --lidar-type VELODYNE_VLP32C
```

Dataparser flags sit **flat** under the `py123d-data` subcommand (not under
`--pipeline.datamanager.dataparser.*`).

See [`docs/examples/py123d_example.yaml`](./examples/py123d_example.yaml) for a filled-in
reference of common knobs.

## Config knobs

| Flag | Default | Notes |
|---|---|---|
| `--data` | `data/py123d` | Root containing `logs/` |
| `--log-id` | `""` | Log directory name under `logs/{split}/` |
| `--split` | `train` | Split subdirectory |
| `--cameras` | `()` | Required for camera training; names from Arrow `camera_name` |
| `--lidars` | `()` | Exactly one name selecting an entry from `lidar.lidar_merged` (e.g. `LIDAR_TOP`) |
| `--lidar-type` | `VELODYNE_VLP32C` | Neurad `Lidars` default; override to match your sensor |
| `--load-cuboids` | `False` | Load actor trajectories from `box_detections_se3.arrow` |
| `--include-deformable-actors` | `True` | Include person / pedestrian-style classes |
| `--allowed-rigid-classes` | vehicle, … | Prefix allowlist vs default-taxonomy label names |
| `--allowed-deformable-classes` | person, human, pedestrian | Same, for deformable actors |
| `--add-missing-points` | `False` | Keep off for merged multi-lidar clouds |
| `--rolling-shutter-time` | `0.03` | Camera readout duration (seconds); `0.0` disables |
| `--time-to-center-pixel` | `0.0` | Offset from capture timestamp to center-row time (seconds) |
| `--horizontal-beam-divergence` | `0.003` | Same public default as ZOD / PandaSet |
| `--vertical-beam-divergence` | `0.0015` | Same public default as ZOD / PandaSet |
| `--use-capture-timestamps` | `True` | Prefer per-camera times from capture sidecar |
| `--capture-metadata-modality` | `custom.capture_metadata` | Custom modality key for that sidecar |
| `--camera-rows` | `sync` | `sync` = one frame per sync row; `all` = every row in each `camera.*.arrow` (dense / sweep exports) |

When `--camera-rows all`, times come from the camera table timestamps (the exporter
should write capture times there for dense logs). The capture-metadata sidecar is not
used for timing in that mode. Lidar and boxes still follow `sync.arrow`.

### Rolling shutter

`--rolling-shutter-time` and `--time-to-center-pixel` are **training CLI overrides**.
They are not read from the Arrow files. Leave `--time-to-center-pixel` at `0.0` when
your capture timestamps already represent center-of-exposure (or when RS compensation
is not needed). If your exporter stores **start-of-exposure** times, set
`--time-to-center-pixel` to roughly half the sensor readout.

### Dynamic actors

With `--load-cuboids True`, boxes are grouped by `track_token`, filtered by the class
prefix allowlists (matched against py123d's default label taxonomy via `to_default()`
when available), transformed with the same WLH→LWH convention as nuScenes, and
stationary tracks (xy std &lt; 0.5 m) are dropped.

Box Arrow metadata must name an importable `BoxDetectionLabel` enum class. Prefer
py123d's public `DefaultBoxDetectionLabel` in exports meant for this dataparser so
no private label modules are required at train time.

## Eval and render

After training, point `--load-config` at the run's `config.yml`:

```bash
export CONFIG=outputs/<experiment>/splatad/<timestamp>/config.yml

python nerfstudio/scripts/eval.py \
  --load-config "$CONFIG" \
  --output-path outputs/<experiment>/eval.json

python nerfstudio/scripts/render.py dataset \
  --load-config "$CONFIG" \
  --output-path renders/<experiment> \
  --pose-source val \
  --rendered-output-names rgb gt-rgb depth
```

On GPUs with limited VRAM, dense lidar evaluation can OOM. In that case disable
mid-train eval (`--steps-per-eval-image 999999`, etc.) and / or downsample lidar
(`--pipeline.datamanager.downsample-factor 0.25`), then evaluate cameras separately.

## Limitations / future work

- Fisheye / wide-FOV cameras are not supported yet.
- Lidar times use a linear sweep offset across the scan; true per-point timestamps
  from the export are not consumed yet.
- Multi-log training (list of `--log-id` values) is not implemented; train one log
  at a time.

## References

- [py123d documentation](https://kesai-labs.github.io/py123d/)
- [py123d paper](https://arxiv.org/html/2605.08084)
- [NeuRAD](https://arxiv.org/abs/2311.15260) · [SplatAD](https://arxiv.org/abs/2411.16816)
