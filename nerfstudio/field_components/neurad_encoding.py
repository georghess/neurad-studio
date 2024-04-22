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
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Type, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nerfstudio.cameras.lidars import transform_points_pairwise
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.spatial_distortions import ScaledSceneContraction
from nerfstudio.model_components.dynamic_actors import DynamicActors
from nerfstudio.utils.math import GaussiansStd
from nerfstudio.utils.poses import inverse as pose_inverse

EPS = 1.0e-7


@dataclass
class StaticSettings:
    hashgrid_dim: int = 4
    """Number of dimensions of each hashgrid feature (per level)."""
    num_levels: int = 8  # TODO: reduce this and/or increase res and dim
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 32
    """Resolution of the base grid for the hasgrid."""
    max_res: int = 8192
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 22
    """Maximum size of the static-world hashmap (per level)."""


@dataclass
class ActorSettings:
    flip_prob: float = 0.5
    """The probability of flipping the sign of positions and directions fed to actor networks."""
    actor_scale: float = 10.0
    """Actor scale, in meters. This is applied before scene contraction."""
    # Hashgrid settings
    hashgrid_dim: int = 4
    """Number of dimensions of each hashgrid feature (per level)."""
    num_levels: int = 4
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 64
    """Resolution of the base grid for the hasgrid."""
    max_res: int = 1024
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 17
    """Maximum size of the static-world hashmap (per level)."""
    use_4d_hashgrid: bool = True
    """Whether to use a 4D hashgrid (with actor idx as 4th dimension), or a list of hashgrids."""


@dataclass
class NeuRADHashEncodingConfig(InstantiateConfig):
    """Hashgrid config"""

    _target: Type = field(default_factory=lambda: NeuRADHashEncoding)
    """Target class to instantiate."""
    static: StaticSettings = field(default_factory=StaticSettings)
    """Static settings"""
    actor: ActorSettings = field(default_factory=ActorSettings)
    """Dynamic settings"""
    disable_actors: bool = False
    """Whether to disable actors."""
    require_actor_grad: bool = True
    """Whether to propagate gradients to actor trajectories."""


class NeuRADHashEncoding(nn.Module):
    def __init__(
        self,
        config: NeuRADHashEncodingConfig,
        dynamic_actors: DynamicActors,
        static_scale: float,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        self.config = config
        self.implementation = implementation
        self.actors = dynamic_actors

        self.static_contraction = ScaledSceneContraction(order=float("inf"), scale=static_scale)
        self.actor_contraction = ScaledSceneContraction(order=float("inf"), scale=config.actor.actor_scale)

        self.static_grid = HashEncoding(
            implementation=implementation,
            features_per_level=config.static.hashgrid_dim,
            num_levels=config.static.num_levels,
            min_res=config.static.base_res,
            max_res=config.static.max_res,
            log2_hashmap_size=config.static.log2_hashmap_size,
        )

        if config.actor.use_4d_hashgrid:
            self._get_actor_features = self._get_actor_features_fast
            n_grids, n_input_dims = 1, 4
        else:
            self._get_actor_features = self._get_actor_features_slow
            n_grids, n_input_dims = self.actors.n_actors, 3
        self.actor_grids = nn.ModuleList(
            [
                HashEncoding(
                    implementation=implementation,
                    features_per_level=config.actor.hashgrid_dim,
                    num_levels=config.actor.num_levels,
                    min_res=config.actor.base_res,
                    max_res=config.actor.max_res,
                    log2_hashmap_size=config.actor.log2_hashmap_size,
                    n_input_dims=n_input_dims,
                )
                for _ in range(n_grids)
            ]
        )

        self.scene_repr_dim = self.static_grid.get_out_dim()

    def get_out_dim(self) -> int:
        return self.scene_repr_dim

    def get_param_groups(self, param_groups: Dict):
        """Get parameter groups."""
        param_groups["hashgrids"] += list(self.static_grid.parameters()) + list(self.actor_grids.parameters())

    @overload
    def forward(self, positions: GaussiansStd, times: Tensor, directions: None) -> Tuple[Tensor, None]:
        ...

    @overload
    def forward(self, positions: GaussiansStd, times: Tensor, directions: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    def forward(
        self,
        positions: GaussiansStd,
        times: Tensor,
        directions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Evaluates the hash grids at the provided positions.

        Args:
            positions: Positions to get features for.

        Returns:
            features: Hashgrid features at the provided positions.
            directions: Directions, transformed to actor space if applicable, and normalized.

        """
        # Static
        static_positions = self.static_contraction(positions)
        static_features = self._get_static_features(static_positions)
        static_features = static_features.to(positions.dtype)
        out_shape = (*times[..., 0].shape, self.scene_repr_dim)
        features = static_features.reshape(out_shape)

        # Actors
        with torch.no_grad() if not self.config.require_actor_grad else nullcontext():
            indices, actor_positions, directions = self._split_static_vs_actors(positions, times, directions)
            ray_idx, sample_idx, actor_idx = indices
        if actor_positions is None or actor_idx.shape[0] == 0:
            return features.view(-1, features.shape[-1]), directions
        actor_positions = self.actor_contraction(actor_positions)

        actor_hashgrid_idx = self.actors.actor_to_id[actor_idx]
        actor_features = self._get_actor_features(actor_positions, actor_hashgrid_idx)
        padded_actor_features = F.pad(actor_features, (0, self.scene_repr_dim - actor_features.shape[-1]))
        features[ray_idx, sample_idx] = padded_actor_features

        return features.view(-1, features.shape[-1]), directions

    def _split_static_vs_actors(self, positions: GaussiansStd, times: Tensor, directions: Optional[Tensor]):
        if self.config.disable_actors or self.actors.n_actors == 0:
            # # haxx to disable actors
            return (torch.empty(0), torch.empty(0), torch.empty(0)), None, directions

        boxes2world, valid = self.actors.get_boxes2world(times[:, 0].squeeze(-1), flatten=False)
        world2boxes = pose_inverse(boxes2world)

        ray_idx, sample_idx, actor_idx = self._get_actor_indices(positions.mean, boxes2world, valid, world2boxes)

        w2b = world2boxes[ray_idx, actor_idx]
        pos = positions.mean[ray_idx, sample_idx]
        pos = transform_points_pairwise(pos, w2b.unsqueeze(-3))

        # Transform directions and/or positions to actor space
        if directions is not None:
            directions = directions.clone()
            dirs = directions[ray_idx, sample_idx]
            dirs = transform_points_pairwise(dirs, w2b, with_translation=False).squeeze(1)
            dirs = dirs / (torch.linalg.norm(dirs, dim=-1, keepdim=True) + EPS)
            directions[ray_idx, sample_idx] = dirs

        # Apply random flip if applicable
        if self.training and self.config.actor.flip_prob > EPS:
            # -1 with prob flip_prob, else 1. Per ray.
            ray_flip = torch.bernoulli(torch.full_like(world2boxes[:, 0, 0, 0], self.config.actor.flip_prob)) * -2 + 1
            flip = torch.ones_like(pos[..., 0:1, :])
            flip[..., 0] = ray_flip[ray_idx].unsqueeze(-1)
            pos = pos * flip
            if directions is not None:
                directions[ray_idx, sample_idx, 0] = directions[ray_idx, sample_idx, 0] * flip[..., 0].squeeze(-1)

        dynamic_positions = GaussiansStd(pos, positions.std[ray_idx, sample_idx])

        return (ray_idx, sample_idx, actor_idx), dynamic_positions, directions

    @torch.no_grad()
    def _get_actor_indices(self, pos, boxes2world, valid, world2boxes):
        # Finds all sample-actor pairs that are close enough (can be multiple per sample)
        actor_radii = (actor_bounds := self.actors.actor_bounds()).norm(dim=-1)
        sample_mean_pos = pos.mean(-2)

        # Check if actor is close to ray (defined by first and last sample)
        point_on_line = sample_mean_pos[:, 0, :]  # n_rays x 3
        line_direction = sample_mean_pos[:, -1, :] - point_on_line
        line_direction = line_direction / (torch.linalg.norm(line_direction, dim=-1, keepdim=True) + EPS)
        line_direction = line_direction.unsqueeze(-2)  # n_rays x 1 x 3
        vec_from_line = boxes2world[..., :3, 3] - point_on_line.unsqueeze(-2)  # n_rays x n_actors x 3
        cross_prod = torch.cross(vec_from_line, line_direction, dim=-1)  # n_rays x n_actors x 3
        distance = torch.linalg.norm(cross_prod, dim=-1)  # n_rays x n_actors
        close_to_line_and_valid_mask = (distance < actor_radii) & valid  # n_rays x n_actors
        ray_idx, actor_idx = close_to_line_and_valid_mask.nonzero(as_tuple=False).T  # N x 2 (ray_idx, actor_idx)

        sample_pos = sample_mean_pos[ray_idx]  # N x n_samples x 3
        actor_pos = boxes2world[ray_idx, actor_idx, :3, 3].unsqueeze(-2).repeat(1, sample_pos.shape[-2], 1)  # N x 3
        distance = torch.linalg.norm(sample_pos - actor_pos, dim=-1)  # N x n_samples
        within_range_mask = distance < actor_radii[actor_idx].unsqueeze(-1)  # N x n_samples
        within_range_idx = within_range_mask.nonzero(as_tuple=False)  # N x 2 (ray_idx, sample_idx)
        indices = torch.stack(
            [ray_idx[within_range_idx[:, 0]], within_range_idx[:, 1], actor_idx[within_range_idx[:, 0]]], dim=-1
        )
        # Check each selected pair to see if it is inside the box
        selected_smp = sample_mean_pos[indices[:, 0], indices[:, 1]]
        selected_w2b = world2boxes[indices[:, 0], indices[:, 2]]
        pos_in_box = transform_points_pairwise(selected_smp, selected_w2b)
        inside_box = (pos_in_box.abs() < actor_bounds[indices[:, 2]]).all(dim=-1)
        indices = indices[inside_box]
        # # remove dupliacate ray-sample pairs (can happen if multiple actors are close to the same ray-sample pair)
        # uniques, inverse_indices = torch.unique(indices[:, :2], dim=0, return_inverse=True, sorted=False)
        # if uniques.shape[0] != indices.shape[0]:
        #     first_occurrences = torch.zeros_like(inverse_indices, dtype=torch.bool)
        #     first_occurrences.scatter_(0, inverse_indices, 1)
        #     indices = indices[first_occurrences]
        # NOTE: here we return potential duplicates, and will "randomly" discard them during feature merging
        return indices[:, 0], indices[:, 1], indices[:, 2]

    def _get_static_features(self, positions: GaussiansStd) -> Tensor:
        features = self.static_grid(positions.mean.view(-1, 3))
        features = self._rescale_grid_features(features, positions, self.static_grid)
        return features

    def _get_actor_features_fast(self, positions: GaussiansStd, actor_indices: Tensor) -> Tensor:
        # Create 4D query positions
        spatial_pos = positions.mean.view(-1, 3)
        actor_indices = actor_indices.unsqueeze(-1).repeat(1, positions.mean.shape[-2])
        actoridx_pos = actor_indices / self.actors.n_actors
        pos = torch.cat([spatial_pos, actoridx_pos.view(-1, 1)], dim=-1)

        actor_grid: HashEncoding = self.actor_grids[0]
        features = actor_grid(pos)
        features = self._rescale_grid_features(features, positions, actor_grid)

        return features

    def _get_actor_features_slow(self, positions: GaussiansStd, actor_indices: Tensor) -> Tensor:
        feats = None
        for i_actor in actor_indices.unique():
            grid: HashEncoding = self.actor_grids[i_actor]
            mask = (actor_indices == i_actor).view(-1)
            actor_pos = positions[mask]
            pos = actor_pos.mean.view(-1, 3) if isinstance(actor_pos, GaussiansStd) else actor_pos
            actor_feats = grid(pos)
            actor_feats = self._rescale_grid_features(actor_feats, actor_pos, grid)
            if feats is None:
                feats = torch.zeros((*mask.shape, actor_feats.shape[-1]), device=pos.device, dtype=actor_feats.dtype)
            feats[mask] = actor_feats
        return feats

    def _rescale_grid_features(self, grid_features: Tensor, x: GaussiansStd, hash_encoding: HashEncoding) -> Tensor:
        prefix_shape = list(x.mean.shape[:-1])
        # Reshape into [..., n_samples, n_levels, n_features_per_level]
        grid_feats = grid_features.view(prefix_shape + [hash_encoding.num_levels * hash_encoding.features_per_level])
        grid_feats = grid_feats.unflatten(-1, (hash_encoding.num_levels, hash_encoding.features_per_level))
        weights = 1 / (hash_encoding.scalings.squeeze(-1) * 2 * x.std).clamp_min(1.0)
        grid_feats = (grid_feats * weights[..., None]).mean(dim=-3).flatten(-2, -1)
        return grid_feats
