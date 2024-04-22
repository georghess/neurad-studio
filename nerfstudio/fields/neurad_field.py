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

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Type

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.neurad_encoding import (
    ActorSettings,
    NeuRADHashEncoding,
    NeuRADHashEncodingConfig,
    StaticSettings,
)
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.model_components.dynamic_actors import DynamicActors
from nerfstudio.model_components.utils import SigmoidDensity

EPS = 1.0e-7


@dataclass
class NeuRADFieldConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: NeuRADField)

    grid: NeuRADHashEncodingConfig = field(
        default_factory=lambda: NeuRADHashEncodingConfig(
            require_actor_grad=True,
            actor=ActorSettings(flip_prob=0.25),
        )
    )
    """Hashgrid config"""

    geo_hidden_dim: int = 32
    """Dimensionality of the hidden units in the first mlp, describing geometry."""
    geo_num_layers: int = 2
    """Number of layers in the first, geometry, mlp."""
    nff_hidden_dim: int = 32
    """Dimensionality of the hidden units in the nff mlp."""
    nff_num_layers: int = 3
    """Number of layers in the second mlp."""
    nff_out_dim: int = 32
    """Dimensionality of the neural feature field output."""

    num_multisamples: int = 1
    """Number of multisamples to use for the fast gaussian approximation."""

    use_sdf: bool = True
    """Whether to learn an SDF or a more 'vanilla' density-field."""
    sdf_beta: float = 20.0  # TODO: tune this (maybe 20.0 like in NeuSim)
    """Slope of the sigmoid function used to convert SDF to density."""
    learnable_beta: bool = True
    """Whether to learn the beta (sdf to density) parameter or not."""


class NeuRADField(Field):
    """NeuRAD field."""

    def __init__(
        self,
        config: NeuRADFieldConfig,
        actors: DynamicActors,
        static_scale: float,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.config = config
        self.implementation = implementation

        self.hashgrid: NeuRADHashEncoding = self.config.grid.setup(
            dynamic_actors=actors,
            static_scale=static_scale,
            implementation=implementation,
        )
        self.geo_feat_dim = self.config.nff_out_dim
        self.mlp_geo = MLP(
            in_dim=self.hashgrid.get_out_dim(),
            num_layers=self.config.geo_num_layers,
            layer_width=self.config.geo_hidden_dim,
            out_dim=self.geo_feat_dim + 1,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
        direction_dim = self.direction_encoding.get_out_dim()
        self.mlp_feature = MLP(
            in_dim=direction_dim + self.geo_feat_dim,
            num_layers=self.config.nff_num_layers,
            layer_width=self.config.nff_hidden_dim,
            out_dim=self.config.nff_out_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        if self.config.use_sdf:
            self.sdf_to_density = SigmoidDensity(self.config.sdf_beta, learnable_beta=self.config.learnable_beta)

    def get_param_groups(self, param_groups: Dict):
        """Get camera optimizer parameters"""
        self.hashgrid.get_param_groups(param_groups)
        param_groups["fields"] += list(self.mlp_geo.parameters()) + list(self.mlp_feature.parameters())
        if self.config.use_sdf:
            param_groups["fields"] += list(self.sdf_to_density.parameters())

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        gaussians = ray_samples.frustums.get_fast_isotropic_gaussian(self.config.num_multisamples)
        features, directions = self.hashgrid(gaussians, ray_samples.times, ray_samples.frustums.directions)

        # First output is the geometry, second is the pass-along vector.
        geo_out, geo_embedding = torch.split(self.mlp_geo(features), [1, self.geo_feat_dim], dim=-1)
        geo_out = geo_out.view(*ray_samples.shape, 1)
        direction_embedding = self.direction_encoding(get_normalized_directions(directions).view(-1, 3))
        feature = geo_embedding + self.mlp_feature(torch.cat([geo_embedding, direction_embedding], dim=-1))
        feature = feature.view(*ray_samples.shape, self.config.nff_out_dim)

        outputs = {FieldHeadNames.FEATURE: feature}
        if self.config.use_sdf:
            signed_distance = geo_out
            outputs[FieldHeadNames.SDF] = signed_distance
            outputs[FieldHeadNames.ALPHA] = self.sdf_to_density(signed_distance)
        else:
            outputs[FieldHeadNames.DENSITY] = trunc_exp(geo_out)

        return outputs


@dataclass
class NeuRADProposalFieldConfig(InstantiateConfig):
    """Configuration of the NeuRAD proposal field."""

    _target: Type = field(default_factory=lambda: NeuRADProposalField)
    """Target class for instantiation."""
    grid: NeuRADHashEncodingConfig = field(
        default_factory=lambda: NeuRADHashEncodingConfig(
            static=StaticSettings(
                log2_hashmap_size=20,
                num_levels=6,
                max_res=4096,
                base_res=128,
                hashgrid_dim=1,
            ),
            actor=ActorSettings(
                log2_hashmap_size=15,
                num_levels=4,
                base_res=64,
                max_res=1024,
                hashgrid_dim=1,
            ),
            require_actor_grad=False,  # No need for trajectory gradients in proposal field
        )
    )
    """Hashgrid config"""
    hidden_dim: int = 16
    """Dimensionality of the hidden units in the density mlp."""


class NeuRADProposalField(Field):
    """Simplified NeuRAD field, for use as a proposal field."""

    def __init__(
        self,
        config: NeuRADProposalFieldConfig,
        actors: DynamicActors,
        static_scale: float,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.config = config
        self.implementation = implementation
        self.hashgrid: NeuRADHashEncoding = self.config.grid.setup(
            dynamic_actors=actors, static_scale=static_scale, implementation=implementation
        )
        self.density_decoder = nn.Linear(self.hashgrid.get_out_dim(), 1, bias=False)

    def get_param_groups(self, param_groups: Dict):
        """Get camera optimizer parameters"""
        self.hashgrid.get_param_groups(param_groups)
        param_groups["fields"] += list(self.density_decoder.parameters())

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        pos = ray_samples.frustums.get_fast_isotropic_gaussian(num_multisamples=1)
        density, _ = self.hashgrid(pos, ray_samples.times, None)
        density = self.density_decoder(density)
        density = trunc_exp(density.to(pos.dtype)).view(*ray_samples.shape, 1)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}
