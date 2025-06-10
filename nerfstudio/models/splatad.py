# Copyright 2025 the authors of NeuRAD and contributors.
# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from pytorch_msssim import SSIM
from torch.nn import BCEWithLogitsLoss, Parameter
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import (
    CameraOptimizer,
    CameraOptimizerConfig,
    CameraVelocityOptimizer,
    CameraVelocityOptimizerConfig,
)
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import Lidars, transform_points, transform_points_pairwise
from nerfstudio.data.datamanagers.full_images_lidar_datamanager import AZIM_CHANNELS_PER_TILE, ELEV_CHANNELS_PER_TILE
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.data.utils.data_utils import points_in_box
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.cnns import BasicBlock
from nerfstudio.model_components.losses import L1Loss, MSELoss

# need following import for background color override
from nerfstudio.model_components.strategy import ADDefaultStrategy, ADMCMCStrategy
from nerfstudio.models.ad_model import ADModel, ADModelConfig
from nerfstudio.models.splatfacto import get_viewmat, resize_image
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import chamfer_distance
from nerfstudio.utils.poses import inverse as pose_inverse, to4x4
from nerfstudio.viewer.viewer_elements import ViewerSlider

try:
    from gsplat.rendering import lidar_rasterization, rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def get_ray_dirs_pinhole(cameras: Cameras, width: int, height: int, c2w: torch.Tensor):
    ys = (torch.arange(height, device=cameras.device, dtype=torch.float32) + (0.5 - cameras.cy[0, 0])) / cameras.fy[
        0, 0
    ]
    xs = (torch.arange(width, device=cameras.device, dtype=torch.float32) + (0.5 - cameras.cx[0, 0])) / cameras.fx[0, 0]
    image_coords = torch.meshgrid(ys, xs, indexing="ij")
    # flip y and z to align with nerfstudio convention
    directions = torch.stack(
        [image_coords[1], -image_coords[0], -torch.ones_like(image_coords[0])], dim=-1
    )  # (h, w, 3)
    directions = directions.view(-1, 3)
    directions = torch.matmul(directions, c2w[0, :3, :3].transpose(0, 1))
    directions = directions / directions.norm(dim=-1, keepdim=True)
    directions = directions.view(height, width, 3)

    return directions


class RGBDecoderCNN(torch.nn.Module):
    def __init__(
        self,
        in_dim=6,
        out_dim=6,
        skip_dim=3,
        weight_init_scale=1e-2,
        hidden_dim=32,
        kernel_size=3,
        num_hidden_blocks=1,
    ):
        super().__init__()
        last_layer = torch.nn.Conv2d(hidden_dim, out_dim, 1)
        last_layer.weight.data *= weight_init_scale
        layers = [BasicBlock(in_dim, hidden_dim, kernel_size, padding=kernel_size // 2, use_bn=False)]
        for _ in range(num_hidden_blocks):
            layers.append(BasicBlock(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, use_bn=False))
        layers.append(last_layer)
        self.net = torch.nn.Sequential(*layers)
        self.skip_dim = skip_dim
        self.out_dim = out_dim

    def forward(self, features, ray_dirs):
        features = features.view(1, *features.shape[-3:])
        albedo, spec = features.split([self.skip_dim, features.shape[-1] - self.skip_dim], dim=-1)

        spec = torch.cat([spec, ray_dirs], dim=-1)
        spec = spec.permute(0, 3, 1, 2)
        spec = self.net(spec)

        spec = spec.permute(0, 2, 3, 1)

        return albedo * (1 + spec[..., :3]) + spec[..., 3:]


@dataclass
class SplatADModelConfig(ADModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: SplatADModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    strategy: Literal["default", "mcmc"] = "mcmc"
    """Strategy to use for the optimization"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 500.0
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0006
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.5
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    use_absgrad: bool = True
    """If True, use absolute gradient for densification"""
    mcmc_cap_max: int = 5_000_000
    """Maximum number of GSs. Default to 1_000_000."""
    mcmc_noise_lr: float = 5e5
    """MCMC samping noise learning rate. Default to 5e5."""
    mcmc_min_opacity: float = 0.005
    """GSs with opacity below this value will be pruned. Default to 0.005."""
    verbose: bool = True
    """Whether to print verbose information. Default to False."""
    max_steps: int = 30_000
    """Number of training steps"""
    init_opacities: float = 0.1
    """Initial opacity of the gaussians"""
    init_scale: float = 1.0
    """Initial scale of the gaussians"""
    max_num_seed_points: int = 2_000_000
    """Maximum number of seed points to use for initialization. -1 means all seed points are used."""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    mcmc_scale_reg_lambda: float = 0.001
    """weight of scale regularization loss"""
    mcmc_opacity_reg_lambda: float = 0.005
    """weight of opacity regularization loss"""
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    camera_velocity_optimizer: CameraVelocityOptimizerConfig = field(
        default_factory=lambda: CameraVelocityOptimizerConfig(enabled=True)
    )
    """Config of the camera velocity optimizer to use"""
    feature_dim: int = 13
    """Dimension of the feature vector"""
    appearance_dim: int = 8
    """Dimension of the appearance vector"""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    actor_flip_probability: float = 0.5
    """Probability of flipping the actor gaussians around the y-axis"""
    flip_actors_at_init: bool = True
    """If True, duplicate the actor gaussians around the y-axis at initialization"""
    n_far_points: float = 300_000
    """Fraction of the seed points to add as extra points on the faces of the scene box."""
    depth_lambda: float = 0.1
    """Weight of the depth loss"""
    depth_loss_quantile_threshold: float = 0.95
    """Quantile threshold for the depth loss"""
    intensity_lambda: float = 1.0
    """Weight of the intensity loss"""
    ray_drop_lambda: float = 0.1
    """Weight of the ray drop loss"""
    compensate_rs_camera: bool = True
    """If True, compensate the camera for the RS camera"""
    compensate_rs_lidar: bool = True
    """If True, compensate the lidar for the RS camera"""
    radius_clip_pix: float = 0.0
    """Clip radius in pixels, 0.0 means no clipping"""
    radius_clip_lidar: float = 0.0
    """Clip radius in degrees, 0.0 means no clipping"""
    line_of_sight_lambda: float = 0.1
    """Weight of the line of sight loss"""
    line_of_sight_dist: float = 0.8
    """"""
    use_camopt_in_eval: bool = False
    """Use result of camera optimization also during evaluation. Only makes sense if trained with train_eval_split=1.0."""
    min_points_per_actor: int = 500
    """Minimum number of points per actor"""
    rgb_decoder_hidden_dim: int = 32
    """Hidden dimension of the RGB decoder"""
    rgb_decoder_kernel_size: int = 3
    """Kernel size of the RGB decoder"""
    rgb_decoder_num_hidden_blocks: int = 1
    """Number of hidden blocks of the RGB decoder"""

    def __post_init__(self):
        if self.strategy == "mcmc":
            self.init_opacities = 0.5
            self.init_scale = 0.2


class SplatADModel(ADModel):
    """Neurad-studio's implementation of Gaussian Splatting

    Args:
        config: SplatAD configuration to instantiate model
    """

    config: SplatADModelConfig

    def __init__(
        self,
        *args,
        seed_points: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        **kwargs,
    ):
        self.seed_points = seed_points
        self.last_size = (1, 1)
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        super().populate_modules()
        self.collider = None

        static_points, dynamic_points = self.split_seed_points(self.seed_points)
        (
            scene_width,
            scene_length,
        ) = (
            self.scene_box.aabb.diff(dim=0)[..., 0].item(),
            self.scene_box.aabb.diff(dim=0)[..., 1].item(),
        )
        random_directions = torch.rand(int(self.config.n_far_points), 3) - 0.5
        random_directions[:, -1] = torch.abs(random_directions[:, -1])
        random_directions = random_directions / random_directions.norm(dim=-1, keepdim=True)
        random_distances = torch.rand(int(self.config.n_far_points), 1)
        near = min(scene_width, scene_length) / 2
        far = 1e4
        random_distances = 1 / (1 / near * (1 - random_distances) + 1 / far * random_distances)
        far_points = random_directions * random_distances
        far_points = torch.cat([far_points, torch.randint_like(far_points, low=0, high=255)], dim=-1)

        # randomly sample points within the scene box
        close_points = torch.rand(int(self.config.n_far_points), 3) - 0.5
        close_points = close_points * torch.tensor([scene_width, scene_length, 50])
        close_points = torch.cat([close_points, torch.randint_like(close_points, low=0, high=255)], dim=-1)

        static_points = torch.cat([static_points, far_points, close_points], dim=0)

        self.gauss_params = self.create_gauss_param_dict(
            dynamic_points,
            [static_points],
            flip_actors_at_init=self.config.flip_actors_at_init,
        )

        dataset_metadata = self.kwargs["metadata"]
        num_sensors = len(dataset_metadata["sensor_idx_to_name"])
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.camera_velocity_optimizer: CameraVelocityOptimizer = self.config.camera_velocity_optimizer.setup(
            num_cameras=self.num_train_data,
            num_unique_cameras=num_sensors,
            device="cpu",
        )

        viewdir_dim = 3
        self.rgb_decoder = torch.compile(
            RGBDecoderCNN(
                self.config.feature_dim + self.config.appearance_dim + viewdir_dim,
                hidden_dim=self.config.rgb_decoder_hidden_dim,
                kernel_size=self.config.rgb_decoder_kernel_size,
                num_hidden_blocks=self.config.rgb_decoder_num_hidden_blocks,
            ),
            disable=True,  # TODO: enable automatically if we don't use the viewer
        )

        self.appearance_embedding = torch.nn.Embedding(num_sensors, self.config.appearance_dim)
        self.fallback_sensor_idx = ViewerSlider("fallback sensor idx", 0, 0, num_sensors - 1, step=1)

        self.setup_rs_editing()

        self.lidar_decoder = MLP(
            in_dim=self.config.feature_dim
            + self.config.appearance_dim
            + viewdir_dim,  # feature + appearance + view direction
            layer_width=32,
            out_dim=2,  # (intensity, ray_drop)
            num_layers=3,
            implementation=self.config.implementation,
            out_activation=None,
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        self.median_l2 = lambda pred, gt: torch.median((pred - gt) ** 2)
        self.mean_rel_l2 = lambda pred, gt: torch.mean(((pred - gt) / gt) ** 2)
        self.rmse = lambda pred, gt: torch.sqrt(torch.mean((pred - gt) ** 2))
        self.chamfer_distance = lambda pred, gt: chamfer_distance(pred, gt, 1_000, True)

        # losses
        self.depth_loss = L1Loss(reduction="none")
        self.intensity_loss = MSELoss()
        self.ray_drop_loss = BCEWithLogitsLoss()

        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        if self.config.strategy == "mcmc":
            self.strategy = ADMCMCStrategy(
                cap_max=self.config.mcmc_cap_max,
                noise_lr=self.config.mcmc_noise_lr,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                refine_every=self.config.refine_every,
                min_opacity=self.config.mcmc_min_opacity,
                verbose=self.config.verbose,
            )
            self.strategy_state = self.strategy.initialize_state()
            self.config.init_opacities = self.config.mcmc_min_opacity
        elif self.config.strategy == "default":
            self.strategy = ADDefaultStrategy(
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=self.num_train_data + self.config.refine_every,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=self.config.verbose,
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        else:
            raise NotImplementedError(f"Strategy {self.config.strategy} is not implemented.")

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    @property
    def id(self):
        return self.gauss_params["id"]

    def setup_rs_editing(self):
        # RS sliders
        self.rs_editing = {
            "rs_time": 0.0,
            "lin_vel_x": 0.0,
            "lin_vel_y": 0.0,
            "lin_vel_z": 0.0,
            "ang_vel_x": 0.0,
            "ang_vel_y": 0.0,
            "ang_vel_z": 0.0,
        }
        self.rs_time_slider = ViewerSlider(
            name="rs time",
            default_value=self.rs_editing["rs_time"],
            min_value=0.0,
            max_value=0.2,
            step=0.001,
            cb_hook=lambda obj: self.rs_editing.update({"rs_time": obj.value}),
        )
        self.rs_lin_vel_x_slider = ViewerSlider(
            name="rs lin vel x",
            default_value=self.rs_editing["lin_vel_x"],
            min_value=-30.0,
            max_value=30.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"lin_vel_x": obj.value}),
        )
        self.rs_lin_vel_y_slider = ViewerSlider(
            name="rs lin vel y",
            default_value=self.rs_editing["lin_vel_y"],
            min_value=-30.0,
            max_value=30.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"lin_vel_y": obj.value}),
        )
        self.rs_lin_vel_z_slider = ViewerSlider(
            name="rs lin vel z",
            default_value=self.rs_editing["lin_vel_z"],
            min_value=-30.0,
            max_value=30.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"lin_vel_z": obj.value}),
        )
        self.rs_ang_vel_x_slider = ViewerSlider(
            name="rs ang vel x",
            default_value=self.rs_editing["ang_vel_x"],
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"ang_vel_x": obj.value}),
        )
        self.rs_ang_vel_y_slider = ViewerSlider(
            name="rs ang vel y",
            default_value=self.rs_editing["ang_vel_y"],
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"ang_vel_y": obj.value}),
        )
        self.rs_ang_vel_z_slider = ViewerSlider(
            name="rs ang vel z",
            default_value=self.rs_editing["ang_vel_z"],
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"ang_vel_z": obj.value}),
        )

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def create_gauss_param_dict(
        self,
        dyn_seed_points_list: List[torch.Tensor],
        static_seed_points_list: List[torch.Tensor],
        flip_actors_at_init: bool = True,
    ):
        """Create a dict of parameters for the gaussians, created from a list with sets of seed points.

        seed_points_list - List of seed points. Each element in the list is a tensor of shape (N, 6).

        Returns:
            Parameter dict with the learnable parameters of the gaussians.
        """

        param_dicts = []
        self.xys_grad_norm = None
        self.max_2Dsize = None
        for i, seed_points in enumerate(dyn_seed_points_list + static_seed_points_list):
            assert seed_points is not None
            assert seed_points.shape[1] == 6
            flip = False
            if flip_actors_at_init and i < len(dyn_seed_points_list):
                flip = True

            means = torch.nn.Parameter(seed_points[:, :3])
            num_points = means.shape[0]

            if num_points < 4:
                warnings.warn(f"Actor {i} has less than 4 points, skipping")
                distances = torch.ones((num_points, 3))
            else:
                distances, _ = self.k_nearest_sklearn(means.data, 3)
                distances = torch.from_numpy(distances)
            # find the average of the three nearest neighbors for each point and use that as the scale
            avg_dist = distances.mean(dim=-1, keepdim=True)
            scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3) * self.config.init_scale))

            quats = torch.nn.Parameter(random_quat_tensor(num_points))

            features_dc = torch.nn.Parameter(seed_points[:, 3:] / 255)
            features_rest = torch.nn.Parameter(
                torch.randn(
                    features_dc.shape[0],
                    max(self.config.feature_dim, 0),
                    dtype=seed_points.dtype,
                    device=seed_points.device,
                )
            )

            opacities = torch.nn.Parameter(torch.logit(self.config.init_opacities * torch.ones(num_points, 1)))
            ids = torch.nn.Parameter(
                torch.full((num_points, 1), min(float(i), len(dyn_seed_points_list))), requires_grad=False
            )
            if flip:
                # duplicate all points and flip them around the y-axis
                mirrored_means = means.clone()
                mirrored_means[:, 0] *= -1
                mirrored_quats = quats.clone()
                mirrored_quats[:, 1] *= -1
                means = torch.nn.Parameter(torch.cat([means, mirrored_means], dim=0))
                scales = torch.nn.Parameter(torch.cat([scales, scales.clone()], dim=0))
                quats = torch.nn.Parameter(torch.cat([quats, mirrored_quats], dim=0))
                features_dc = torch.nn.Parameter(torch.cat([features_dc, features_dc.clone()], dim=0))
                features_rest = torch.nn.Parameter(torch.cat([features_rest, features_rest.clone()], dim=0))
                opacities = torch.nn.Parameter(torch.cat([opacities, opacities.clone()], dim=0))
                ids = torch.nn.Parameter(torch.cat([ids, ids.clone()], dim=0))

            param_dicts.append(
                {
                    "means": means,
                    "scales": scales,
                    "quats": quats,
                    "features_dc": features_dc,
                    "features_rest": features_rest,
                    "opacities": opacities,
                    "id": ids,
                }
            )
        return torch.nn.ParameterDict(
            {
                key: torch.cat(
                    [param_dict[key] for param_dict in param_dicts],
                    dim=0,
                )
                for key in param_dicts[0].keys()
            }
        )

    @torch.no_grad()
    def split_seed_points(self, seed_points: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Split the seed points into static and dynamic points using the dynamic actors information.

        seed_points - Tuple of (points, colors, times)

        Returns:
            Tuple of (static_points, dynamic_points). Points have shape (N, 6) where N is the number of points and the
            columns are x, y, z, r, g, b. Dynamic points is a list of tensors, one for each actor.
        """
        num_actors = self.dynamic_actors.n_actors
        static_points = []
        dynamic_points = [[] for _ in range(num_actors)]
        unique_seed_point_times = seed_points[2].unique()
        # begin by adding a few random points to all actors
        for actor_idx in range(num_actors):
            n_points = self.config.min_points_per_actor
            random_points = (
                torch.rand((n_points, 3), device=seed_points[0].device) - 0.5
            ) * self.dynamic_actors.actor_sizes[actor_idx]
            random_colors = torch.rand((n_points, 3), device=seed_points[0].device) * 255
            dynamic_points[actor_idx].append(torch.cat([random_points, random_colors], dim=-1))
        for current_time in unique_seed_point_times:
            points = seed_points[0][seed_points[2] == current_time]
            colors = seed_points[1][seed_points[2] == current_time]
            static_mask = torch.ones(points.shape[0], dtype=torch.bool)
            boxes2world, exists_at_time = self.dynamic_actors.get_boxes2world(current_time.unsqueeze(-1), flatten=False)
            boxes2world = boxes2world.squeeze(0)
            exists_at_time = exists_at_time.squeeze(0)
            assert boxes2world.shape[0] == num_actors
            for actor_idx in range(num_actors):
                if exists_at_time[actor_idx]:
                    actor_mask = points_in_box(
                        points,
                        boxes2world[actor_idx],
                        self.dynamic_actors.actor_sizes[actor_idx] + self.dynamic_actors.actor_padding,
                    )
                    if actor_mask.any():
                        world2box = pose_inverse(boxes2world[actor_idx]).reshape(-1, 3, 4)
                        points_in_local_box = transform_points(points[actor_mask].reshape(-1, 3), world2box)
                        mirrored_points = points_in_local_box.clone()
                        mirrored_points[:, 0] *= -1
                        points_in_local_box = torch.cat([points_in_local_box, mirrored_points], dim=0)
                        actor_colors = torch.cat([colors[actor_mask], colors[actor_mask]], dim=0)
                        dynamic_points[actor_idx].append(
                            torch.cat([points_in_local_box, actor_colors], dim=-1)
                        )  # x, y, z, r, g, b

                        static_mask = static_mask & ~actor_mask
                    else:
                        dynamic_points[actor_idx].append(torch.empty((0, 6), device=points.device))

            static_points.append(torch.cat([points[static_mask], colors[static_mask]], dim=-1))  # x, y, z, r, g, b

        # TODO(carlinds): Currently pruning per actor and static scene. Total number of points will be more than max_num_seed_points.
        dynamic_points = [self.prune_seed_points(torch.cat(points), is_dynamic=True) for points in dynamic_points]
        static_points = self.prune_seed_points(torch.cat(static_points))
        return static_points, dynamic_points

    def prune_seed_points(self, seed_points: torch.Tensor, is_dynamic: bool = False):
        """Prune the seed points to the maximum number of seed points allowed.

        seed_points - Seed points tensor of shape (N, 6).

        Returns:
            Pruned seed points tensor.
        """
        if seed_points is None or seed_points.shape[0] == 0:
            return seed_points

        if self.config.max_num_seed_points > 0 and seed_points.shape[0] > self.config.max_num_seed_points:
            n_seed_points = seed_points.shape[0]
            perm_idx = torch.randperm(n_seed_points)
            seed_points = seed_points[perm_idx][: self.config.max_num_seed_points]
        return seed_points

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def step_post_backward(self, step):
        assert step == self.step
        if isinstance(self.strategy, ADDefaultStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                packed=False,
                dynamic_actors=self.dynamic_actors,
            )
        elif isinstance(self.strategy, ADMCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                lr=self.optimizers["means"].param_groups[0]["lr"],
            )
        else:
            raise NotImplementedError(f"Strategy {self.config.strategy} is not implemented.")

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward,
            )
        )
        return cbs

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        param_groups = super().get_param_groups()
        param_groups.update(self.get_gaussian_param_groups())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        self.camera_velocity_optimizer.get_param_groups(param_groups=param_groups)
        param_groups["fields"] = []

        param_groups["fields"] += list(self.rgb_decoder.parameters())
        param_groups["fields"] += list(self.appearance_embedding.parameters())

        param_groups["fields"] += list(self.lidar_decoder.parameters())
        return param_groups

    def _get_downscale_factor(self) -> int:
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                self.background_color = self.background_color.to(self.device)
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def _get_actor_adjusted_means(
        self, means: torch.Tensor, times: torch.Tensor, ids: torch.Tensor, calc_vels: bool = True
    ):
        means_in_world = means.clone()
        boxes2world, _ = self.dynamic_actors.get_boxes2world(times, flatten=False)
        boxes2world = boxes2world.squeeze(0)
        if self.training:
            flip_matrix = torch.eye(4, device=boxes2world.device).unsqueeze(0).repeat(boxes2world.shape[0], 1, 1)
            flip_matrix[:, 0, 0] += (
                (torch.rand(boxes2world.shape[0], device=boxes2world.device)) < self.config.actor_flip_probability
            ) * -2
            boxes2world = boxes2world @ flip_matrix
        boxes2world = boxes2world[..., :3, :]  # remove the last row of the pose matrix
        num_actors = self.dynamic_actors.actor_sizes.shape[0]
        actor_idx = (ids < num_actors).squeeze().nonzero().squeeze()
        curr_ids = ids.index_select(0, actor_idx).squeeze().to(torch.long)

        vels_in_world = None
        b2w_per_pt = boxes2world.index_select(dim=0, index=curr_ids)
        if calc_vels and len(boxes2world) > 0:
            lin_vel, ang_vel = self.dynamic_actors.get_velocities(times).split([3, 3], dim=-1)
            lin_vel = lin_vel.squeeze(0).squeeze(0)
            ang_vel = ang_vel.squeeze(0).squeeze(0)

            vels_in_world = torch.zeros_like(means)
            angle_vel_in_box = torch.cross(
                ang_vel.index_select(dim=0, index=curr_ids), means.index_select(0, actor_idx)
            )
            vels_in_world[actor_idx] += lin_vel.index_select(dim=0, index=curr_ids) + transform_points_pairwise(
                angle_vel_in_box, b2w_per_pt, with_translation=False
            )

        means_in_world[actor_idx] = transform_points_pairwise(means_in_world.index_select(0, actor_idx), b2w_per_pt)

        return means_in_world, vels_in_world

    def get_camera_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training or self.config.use_camopt_in_eval:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        if camera_scale_fac != 1:
            camera.rescale_output_resolution(1 / camera_scale_fac)
        K = camera.get_intrinsics_matrices()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        ray_dirs = get_ray_dirs_pinhole(camera, W, H, optimized_camera_to_world)
        if camera_scale_fac != 1:
            camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        colors = torch.cat((self.features_dc, self.features_rest), dim=-1)

        # rolling shutter
        camera_linear_vel = None
        camera_angular_vel = None
        rolling_shutter_time = None
        camera_times = camera.times
        if camera.metadata is not None and self.config.compensate_rs_camera:
            rolling_shutter_time = camera.metadata.get(
                "rolling_shutter_time",
                torch.zeros(
                    (
                        1,
                        1,
                    ),
                    device=self.device,
                ),
            )[0]
            time_to_center_pixel = camera.metadata.get(
                "time_to_center_pixel",
                torch.zeros(
                    (
                        1,
                        1,
                    ),
                    device=self.device,
                ),
            )
            velocities = self.camera_velocity_optimizer.apply_to_camera_velocity(
                camera, return_init_only=(not self.training) and (not self.config.use_camopt_in_eval)
            )
            camera_linear_vel, camera_angular_vel = torch.split(velocities, 3, dim=-1)
            time_to_center_pixel = (
                time_to_center_pixel + self.camera_velocity_optimizer.get_time_to_center_pixel_adjustment(camera)
            )
            optimized_camera_to_world = torch.cat(
                [
                    optimized_camera_to_world[:, :3, :3],
                    optimized_camera_to_world[:, :3, 3:4]
                    + (
                        torch.matmul(camera_linear_vel, optimized_camera_to_world[0, :3, :3].transpose(0, 1))
                        * (time_to_center_pixel)
                    )[..., None],
                ],
                dim=-1,
            )
            camera_times = camera.times + time_to_center_pixel

            flip_tensor = torch.ones(3, device=camera_linear_vel.device, dtype=camera_linear_vel.dtype)
            flip_tensor[1:] = -1
            camera_linear_vel = camera_linear_vel * flip_tensor  # flip y and z
            camera_angular_vel = camera_angular_vel * flip_tensor  # flip y and z
        else:
            camera_linear_vel = torch.tensor(
                [[self.rs_editing["lin_vel_x"], self.rs_editing["lin_vel_y"], self.rs_editing["lin_vel_z"]]],
                device=self.device,
            )
            camera_angular_vel = torch.tensor(
                [[self.rs_editing["ang_vel_x"], self.rs_editing["ang_vel_y"], self.rs_editing["ang_vel_z"]]],
                device=self.device,
            )
            rolling_shutter_time = torch.tensor([self.rs_editing["rs_time"]], device=self.device)

        viewmat = get_viewmat(optimized_camera_to_world)
        means, vels = self._get_actor_adjusted_means(self.means, camera_times, self.id)

        render, alpha, self.info = rasterization(
            means=means,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=torch.sigmoid(self.opacities).squeeze(-1),
            colors=colors,
            velocities=vels,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            linear_velocity=camera_linear_vel,
            angular_velocity=camera_angular_vel,
            rolling_shutter_time=rolling_shutter_time,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.5,
            far_plane=1e10,
            radius_clip=self.config.radius_clip_pix,
            render_mode=render_mode,
            sh_degree=None,
            sparse_grad=False,
            absgrad=self.config.use_absgrad,
            rasterize_mode=self.config.rasterize_mode,
            channel_chunk=128,
            eps2d=0.3,
        )
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )

        background = self._get_background_color()

        rendered_features = render[..., :-1] if render_mode == "RGB+ED" else render
        appearance_features = self._get_appearance_embedding(camera, rendered_features)
        rendered_features = torch.cat((rendered_features, appearance_features), dim=-1)
        rgb = self.rgb_decoder(rendered_features, ray_dirs.unsqueeze(0))
        rgb = rgb + (1 - alpha) * background

        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        out = {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

        return out

    def get_lidar_outputs(self, lidar: Lidars) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(lidar, Lidars):
            print("Called get_outputs with not a camera")
            return {}
        assert (
            (lidar.azimuths is not None and lidar.elevations is not None)
            or lidar.metadata
            and "raster_pts" in lidar.metadata
        )
        if self.training or self.config.use_camopt_in_eval:
            assert lidar.shape[0] == 1, "Only one camera at a time"
            optimized_lidar_to_world = self.camera_optimizer.apply_to_camera(lidar)
        else:
            optimized_lidar_to_world = lidar.lidar_to_worlds

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if lidar.metadata and "raster_pts" in lidar.metadata:
            raster_pts = lidar.metadata["raster_pts"][
                ..., :-1
            ]  # omit intensity channel, only needed for metric/loss computation
            tile_elevation_boundaries = lidar.metadata["elevation_boundaries"]
            min_azimuth = -180
            max_azimuth = 180
            min_elevation = tile_elevation_boundaries.min()
            max_elevation = tile_elevation_boundaries.max()
            azimuth_resolution = lidar.metadata["azimuth_resolution"]
        else:
            elevs, azims = torch.meshgrid(
                torch.rad2deg(lidar.elevations.flatten()), torch.rad2deg(lidar.azimuths.flatten())
            )
            raster_pts = torch.stack([azims, elevs, torch.ones_like(azims), torch.zeros_like(azims)], dim=-1).to(
                self.device
            )[None]
            tile_elevation_boundaries = torch.rad2deg(lidar.elevations[0, ::ELEV_CHANNELS_PER_TILE]).to(self.device)
            tile_elevation_boundaries = tile_elevation_boundaries.flatten()
            tile_elevation_boundaries = torch.cat(
                [
                    tile_elevation_boundaries,
                    torch.tensor([tile_elevation_boundaries[..., -1].item() + 1], device=self.device),
                ]
            )
            azimuth_resolution = float(torch.rad2deg((lidar.azimuths[0, 1] - lidar.azimuths[0, 0])))
            min_azimuth = -180
            max_azimuth = 180
            min_elevation = tile_elevation_boundaries.min().item()
            max_elevation = tile_elevation_boundaries.max().item() + 1e-6

        # rolling shutter
        lidar_linear_vel = torch.zeros(1, 3, device=self.device)
        lidar_angular_vel = torch.zeros(1, 3, device=self.device)
        rolling_shutter_time = torch.zeros(1, device=self.device)
        lidar_times = lidar.times
        if lidar.metadata is not None and self.config.compensate_rs_lidar:
            max_offset, min_offset = raster_pts[..., 3].max(), raster_pts[..., 3].min()
            rolling_shutter_time = (max_offset - min_offset).unsqueeze(0)
            velocities = self.camera_velocity_optimizer.apply_to_camera_velocity(
                lidar, return_init_only=(not self.training) and (not self.config.use_camopt_in_eval)
            )
            lidar_linear_vel, lidar_angular_vel = torch.split(velocities, 3, dim=-1)

            time_to_center_adjustment = (max_offset + min_offset) / 2
            optimized_lidar_to_world = torch.cat(
                [
                    optimized_lidar_to_world[:, :3, :3],
                    optimized_lidar_to_world[:, :3, 3:4]
                    + (
                        (torch.einsum("bij,bj->bi", optimized_lidar_to_world[..., :3, :3], lidar_linear_vel))
                        * time_to_center_adjustment
                    )[..., None],
                ],
                dim=-1,
            )
            lidar_times = lidar.times + time_to_center_adjustment
            raster_pts[..., 3] = raster_pts[..., 3] - time_to_center_adjustment

        lidar_features = self.features_rest.unsqueeze(0)
        batch_size = raster_pts.shape[0]
        viewmat = to4x4(pose_inverse(optimized_lidar_to_world))
        if batch_size > 1:
            viewmat = viewmat.repeat(batch_size, 1, 1)
            lidar_features = lidar_features.repeat(batch_size, 1, 1)
            lidar_linear_vel = lidar_linear_vel.repeat(batch_size, 1)
            lidar_angular_vel = lidar_angular_vel.repeat(batch_size, 1)
            rolling_shutter_time = rolling_shutter_time.repeat(batch_size)

        means, vels = self._get_actor_adjusted_means(self.means, lidar_times, self.id)
        render, alpha, alpha_sum_until_points, self.info = lidar_rasterization(
            means=means,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=torch.sigmoid(self.opacities).squeeze(-1),
            lidar_features=lidar_features,  # [(C,) N, D]
            velocities=vels,
            viewmats=viewmat,  # [1, 4, 4]
            min_azimuth=min_azimuth,
            max_azimuth=max_azimuth,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            n_elevation_channels=raster_pts.shape[1],
            azimuth_resolution=azimuth_resolution,
            raster_pts=raster_pts,  # [C, H, W, 4]
            tile_width=AZIM_CHANNELS_PER_TILE,
            tile_height=ELEV_CHANNELS_PER_TILE,
            tile_elevation_boundaries=tile_elevation_boundaries,
            linear_velocity=lidar_linear_vel,
            angular_velocity=lidar_angular_vel,
            rolling_shutter_time=rolling_shutter_time,
            near_plane=0.2,
            far_plane=300,
            radius_clip=self.config.radius_clip_lidar,
            compute_alpha_sum_until_points=(self.config.line_of_sight_lambda > 0) and (self.training),
            compute_alpha_sum_until_points_threshold=self.config.line_of_sight_dist,
            sparse_grad=False,
            absgrad=self.config.use_absgrad,
            rasterize_mode=self.config.rasterize_mode,
            channel_chunk=128,
            eps2d=0.01718873385,
        )
        self.info["width"] = -1
        self.info["height"] = -1
        self.last_size = (self.last_size[0], self.last_size[1], -1)
        # TODO(carlin): Add this back if we want to start pruning based on 2D size in lidar image space.
        # self.xys = info["means2d"]  # [1, N, 2]
        # self.radii = info["radii"][0]  # [N]
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )

        depth_im = render[:, ..., -1:]

        rendered_features = render[..., :-1]
        appearance_features = self._get_appearance_embedding(lidar, rendered_features)
        rendered_features = torch.cat((rendered_features, appearance_features), dim=-1)
        raster_pts_degrees = torch.deg2rad(raster_pts[..., :2])
        lidar_ray_dir = torch.cat(
            [
                torch.cos(raster_pts_degrees[..., 0:1])
                * torch.cos(raster_pts_degrees[..., 1:2]),  # x = cos(azimuth) * cos(elevation)
                torch.sin(raster_pts_degrees[..., 0:1])
                * torch.cos(raster_pts_degrees[..., 1:2]),  # y = sin(azimuth) * cos(elevation)
                torch.sin(raster_pts_degrees[..., 1:2]),  # z = sin(elevation)
            ],
            dim=-1,
        )
        lidar_ray_dir_in_world = (
            optimized_lidar_to_world[:, :3, :3].reshape(1, 1, 1, 3, 3) @ lidar_ray_dir.unsqueeze(-1)
        ).squeeze(-1)

        intensity, ray_drop_logits = (
            self.lidar_decoder(
                torch.cat(
                    [
                        rendered_features.reshape(-1, rendered_features.shape[-1]),
                        lidar_ray_dir_in_world.reshape(-1, lidar_ray_dir_in_world.shape[-1]),
                    ],
                    dim=-1,
                )
            )
            .reshape((*lidar_ray_dir_in_world.shape[:-1], self.lidar_decoder.out_dim))
            .split([1, 1], dim=-1)
        )

        out = {
            "depth": depth_im,  # type: ignore
            "accumulation": alpha,  # type: ignore
            "median_depth": self.info["median_depths"]
            + (alpha <= 0.5)
            * (depth_im / alpha.clamp_min(1e-10)),  # add normalized expected depth where we did not reach alpha=0.5
        }  # type: ignore

        if intensity is not None:
            out["intensity"] = intensity.sigmoid().to(torch.float32)

        if ray_drop_logits is not None:
            out["ray_drop_logits"] = ray_drop_logits.to(torch.float32)
            out["ray_drop_prob"] = ray_drop_logits.sigmoid().to(torch.float32)

        if alpha_sum_until_points is not None:
            out["alpha_sum_until_points"] = alpha_sum_until_points

        return out  # type: ignore

    def get_outputs(self, sensor: Union[Cameras, Lidars]) -> Dict[str, Union[torch.Tensor, List]]:
        if isinstance(sensor, Cameras):
            return self.get_camera_outputs(sensor)
        elif isinstance(sensor, Lidars):
            return self.get_lidar_outputs(sensor)
        else:
            raise ValueError("Unknown sensor type")

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def filter_lidar_pred_and_gt(self, outputs, batch, output_point_cloud=False):
        gt_lidar = batch["raster_pts"]  # (azimuth, elev, depth, time, intensity)
        raster_pts_valid_and_did_return = batch["raster_pts_valid_depth_and_did_return"]
        raster_pts_did_return = batch["raster_pts_did_return"].flatten()
        raster_pts_valid_and_did_not_return = batch["raster_pts_valid_depth_and_did_not_return"]

        gt = {}
        gt["depth"] = gt_lidar[..., 2].flatten()[raster_pts_valid_and_did_return]
        gt["intensity"] = gt_lidar[..., 4].flatten()[raster_pts_valid_and_did_return]
        gt["ray_drop"] = ~raster_pts_did_return
        gt["valid"] = gt_lidar[..., 2].flatten() > 0

        pred = {}
        pred["depth"] = outputs["depth"].flatten()[raster_pts_valid_and_did_return]
        pred["depth_dropped"] = outputs["depth"].flatten()[raster_pts_valid_and_did_not_return]
        pred["intensity"] = outputs["intensity"].flatten()[raster_pts_valid_and_did_return]
        pred["intensity_dropped"] = outputs["intensity"].flatten()[raster_pts_valid_and_did_not_return]
        pred["ray_drop"] = outputs["ray_drop_logits"].flatten() * gt["valid"] - (~gt["valid"]) * 10_000
        pred["accumulation"] = outputs["accumulation"].flatten()[raster_pts_valid_and_did_return]
        pred["accumulation_dropped"] = outputs["accumulation"].flatten()[raster_pts_valid_and_did_not_return]
        pred["median_depth"] = outputs["median_depth"].flatten()[raster_pts_valid_and_did_return]

        if "alpha_sum_until_points" in outputs:
            pred["alpha_sum_until_points"] = outputs["alpha_sum_until_points"].flatten()[
                raster_pts_valid_and_did_return
            ]
            pred["alpha_sum_until_points_dropped"] = outputs["alpha_sum_until_points"].flatten()[
                raster_pts_valid_and_did_not_return
            ]

        if output_point_cloud:
            azimuth_angles = torch.deg2rad(gt_lidar[..., 0].flatten())
            elevation_angles = torch.deg2rad(gt_lidar[..., 1].flatten())
            directions = torch.stack(
                [
                    torch.cos(elevation_angles) * torch.cos(azimuth_angles),
                    torch.cos(elevation_angles) * torch.sin(azimuth_angles),
                    torch.sin(elevation_angles),
                ],
                dim=-1,
            )

            gt["point_cloud"] = batch["lidar"][batch["lidar_pts_did_return"].squeeze(), :3]
            pred["point_cloud"] = (
                outputs["depth"].view(-1, 1) * directions
                + batch["linear_velocities_local"] * gt_lidar[..., 3].view(-1, 1)
            )[((pred["ray_drop"].sigmoid() <= 0.5) * gt["valid"])]
            pred["median_point_cloud"] = (
                outputs["median_depth"].view(-1, 1) * directions
                + batch["linear_velocities_local"] * gt_lidar[..., 3].view(-1, 1)
            )[((pred["ray_drop"].sigmoid() <= 0.5) * gt["valid"])]

        return pred, gt

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}
        if "image" in batch:
            gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])

            predicted_rgb = outputs["rgb"]
            # slice gt_rgb to same shape as predicted_rgb
            if not gt_rgb.shape[:2] == predicted_rgb.shape[:2]:
                gt_rgb = gt_rgb[: predicted_rgb.shape[0], : predicted_rgb.shape[1], :]
                # raise user warning
                warnings.warn("GT image and predicted image have different shapes. Cropping GT image to match.")

            metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

            metrics_dict["gaussian_count"] = self.num_points

        if "raster_pts" in batch:
            pred, gt = self.filter_lidar_pred_and_gt(outputs, batch)

            metrics_dict["depth_median_l2"] = float(self.median_l2(pred["depth"], gt["depth"]))
            metrics_dict["depth_mean_rel_l2"] = float(self.mean_rel_l2(pred["depth"], gt["depth"]))
            metrics_dict["median_depth_median_l2"] = float(self.median_l2(pred["median_depth"], gt["depth"]))
            metrics_dict["median_depth_mean_rel_l2"] = float(self.mean_rel_l2(pred["median_depth"], gt["depth"]))
            metrics_dict["intensity_rmse"] = float(self.rmse(pred["intensity"], gt["intensity"]))
            metrics_dict["ray_drop_accuracy"] = (
                ((pred["ray_drop"].sigmoid() > 0.5) == gt["ray_drop"]) * gt["valid"]
            ).sum() / gt["valid"].sum()

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        self.camera_velocity_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = {}
        if "image" in batch:
            gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
            pred_img = outputs["rgb"]
            if not gt_img.shape[:2] == pred_img.shape[:2]:
                gt_img = gt_img[: pred_img.shape[0], : pred_img.shape[1], :]
                # raise user warning
                warnings.warn("GT image and predicted image have different shapes. Cropping GT image to match.")

            # Set masked part of both ground-truth and rendered image to black.
            # This is a little bit sketchy for the SSIM loss.
            if "mask" in batch:
                # batch["mask"] : [H, W, 1]
                mask = self._downscale_if_required(batch["mask"])
                mask = mask.to(self.device)
                assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
                gt_img = gt_img * mask
                pred_img = pred_img * mask

            Ll1 = torch.abs(gt_img - pred_img).mean()
            simloss = (
                1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
                if self.config.ssim_lambda > 0
                else 0
            )
            loss_dict["main_loss"] = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss

        if self.config.mcmc_scale_reg_lambda and isinstance(self.strategy, ADMCMCStrategy):
            mcmc_scale_reg = torch.abs(torch.exp(self.scales).mean()) * self.config.mcmc_scale_reg_lambda
        else:
            mcmc_scale_reg = torch.zeros(1, device=self.device)

        loss_dict["mcmc_scale_reg"] = mcmc_scale_reg

        if self.config.mcmc_opacity_reg_lambda and isinstance(self.strategy, ADMCMCStrategy):
            mcmc_opacity_reg = torch.abs(torch.sigmoid(self.opacities).mean()) * self.config.mcmc_opacity_reg_lambda
        else:
            mcmc_opacity_reg = torch.zeros(1, device=self.device)

        loss_dict["mcmc_opacity_reg"] = mcmc_opacity_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            self.camera_velocity_optimizer.get_loss_dict(loss_dict)

        if "raster_pts" in batch:
            pred, gt = self.filter_lidar_pred_and_gt(outputs, batch)

            unreduced_depth_loss = self.depth_loss(pred["depth"], gt["depth"])
            quantile = torch.quantile(unreduced_depth_loss, self.config.depth_loss_quantile_threshold)
            quantile_mask = unreduced_depth_loss < quantile
            loss_dict["depth_loss"] = self.config.depth_lambda * torch.mean(unreduced_depth_loss * quantile_mask)

            loss_dict["intensity_loss"] = self.config.intensity_lambda * self.intensity_loss(
                pred["intensity"] * quantile_mask,
                gt["intensity"] * quantile_mask,
            )
            loss_dict["ray_drop_loss"] = self.config.ray_drop_lambda * self.ray_drop_loss(
                pred["ray_drop"],
                gt["ray_drop"].to(pred["ray_drop"]),
            )

            if "alpha_sum_until_points" in pred and self.config.line_of_sight_lambda > 0:
                loss_dict["alpha_sum_until_points_loss"] = self.config.line_of_sight_lambda * torch.mean(
                    pred["alpha_sum_until_points"] * quantile_mask
                )

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Union[Cameras, Lidars], obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        images_dict = {}
        metrics_dict = {}
        if "image" in batch:
            gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
            predicted_rgb = outputs["rgb"]

            # slice gt_rgb to same shape as predicted_rgb
            if not gt_rgb.shape[:2] == predicted_rgb.shape[:2]:
                gt_rgb = gt_rgb[: predicted_rgb.shape[0], : predicted_rgb.shape[1], :]
                # raise user warning
                warnings.warn("GT image and predicted image have different shapes. Cropping GT image to match.")

            combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
            predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

            psnr = self.psnr(gt_rgb, predicted_rgb)
            ssim = self.ssim(gt_rgb, predicted_rgb)
            lpips = self.lpips(gt_rgb, predicted_rgb)

            # all of these metrics will be logged as scalars
            metrics_dict.update({"psnr": float(psnr), "ssim": float(ssim), "lpips": float(lpips)})  # type: ignore

            images_dict.update({"img": combined_rgb})

        if "raster_pts" in batch:
            pred, gt = self.filter_lidar_pred_and_gt(outputs, batch, output_point_cloud=True)

            metrics_dict["depth_median_l2"] = float(self.median_l2(pred["depth"], gt["depth"]))
            metrics_dict["depth_mean_rel_l2"] = float(self.mean_rel_l2(pred["depth"], gt["depth"]))
            metrics_dict["median_depth_median_l2"] = float(self.median_l2(pred["median_depth"], gt["depth"]))
            metrics_dict["median_depth_mean_rel_l2"] = float(self.mean_rel_l2(pred["median_depth"], gt["depth"]))
            metrics_dict["intensity_rmse"] = float(self.rmse(pred["intensity"], gt["intensity"]))
            metrics_dict["ray_drop_accuracy"] = float(
                (((pred["ray_drop"].sigmoid() > 0.5) == gt["ray_drop"]) * gt["valid"]).sum() / gt["valid"].sum()
            )

            if pred["point_cloud"].shape[0] > 0 and gt["point_cloud"].shape[0] > 0:
                metrics_dict["chamfer_distance"] = float(self.chamfer_distance(pred["point_cloud"], gt["point_cloud"]))

            if pred["median_point_cloud"].shape[0] > 0 and gt["point_cloud"].shape[0] > 0:
                metrics_dict["median_chamfer_distance"] = float(
                    self.chamfer_distance(pred["median_point_cloud"], gt["point_cloud"])
                )

        return metrics_dict, images_dict

    def _get_appearance_embedding(self, sensor: Union[Cameras, Lidars], features: torch.Tensor) -> torch.Tensor:
        metadata = sensor.metadata if sensor.metadata is not None else {}
        sensor_idx = metadata.get("sensor_idxs", None)
        if sensor_idx is None:
            assert not self.training, "Sensor sensor_idx must be present in metadata during training"
            sensor_idx = torch.full((1,), self.fallback_sensor_idx.value, device=features.device, dtype=torch.long)

        embed = self.appearance_embedding(sensor_idx).expand(*features.shape[:-1], -1)
        return embed
