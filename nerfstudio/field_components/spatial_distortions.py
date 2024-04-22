# Copyright 2024 the authors of NeuRAD and contributors.
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

"""Space distortions."""

import abc
from typing import Optional, Union, overload

import torch
from functorch import jacrev, vmap
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.utils.math import Gaussians, GaussiansStd


class SpatialDistortion(nn.Module):
    """Apply spatial distortions"""

    @abc.abstractmethod
    def forward(self, positions: Union[Float[Tensor, "*bs 3"], Gaussians]) -> Union[Float[Tensor, "*bs 3"], Gaussians]:
        """
        Args:
            positions: Sample to distort

        Returns:
            Union: distorted sample
        """


class SceneContraction(SpatialDistortion):
    """Contract unbounded space using the contraction was proposed in MipNeRF-360.
        We use the following contraction equation:

        .. math::

            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}

        If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
        radius 2. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 4.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.

        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.

    """

    def __init__(self, order: Optional[Union[float, int]] = None) -> None:
        super().__init__()
        self.order = order

    @overload
    def forward(self, positions: Gaussians) -> Gaussians:
        ...

    @overload
    def forward(self, positions: GaussiansStd) -> GaussiansStd:
        ...

    @overload
    def forward(self, positions: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs 3"]:
        ...

    def forward(self, positions: Union[Gaussians, GaussiansStd, Tensor]) -> Union[Gaussians, GaussiansStd, Tensor]:
        def contract(x):
            mag = torch.linalg.norm(x, ord=self.order, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        if isinstance(positions, Gaussians):
            means = contract(positions.mean.clone())

            def contract_gauss(x):
                return (2 - 1 / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)) * (
                    x / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)
                )

            jc_means = vmap(jacrev(contract_gauss))(positions.mean.view(-1, positions.mean.shape[-1]))
            jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])

            # Only update covariances on positions outside the unit sphere
            mag = positions.mean.norm(dim=-1)
            mask = mag >= 1
            cov = positions.cov.clone()
            cov[mask] = jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)

            return Gaussians(mean=means, cov=cov)

        elif isinstance(positions, GaussiansStd):
            # ZipNerf-style linearized contraction
            means = positions.mean.clone()
            std = positions.std.clone()
            mag = torch.linalg.norm(means, ord=self.order, dim=-1)[..., None]
            mask = mag < 1
            clamped_mag = mag.clamp_min(1.0)
            means = torch.where(mask, means, (2 - (1 / clamped_mag)) * (means / clamped_mag))
            std_scaling = ((2 * clamped_mag - 1).pow(1 / 3) / clamped_mag) ** 2
            std = torch.where(mask, std, std * std_scaling)
            return GaussiansStd(mean=means, std=std)

        return contract(positions)


class ScaledSceneContraction(SceneContraction):
    """This is a SceneContraction with a scale factor applied before the contraction."""

    def __init__(self, order: Optional[Union[float, int]] = None, scale: float = 1.0, normalize: bool = True) -> None:
        super().__init__(order=order)
        self.scale = scale
        self.normalize = normalize

    def forward(self, positions):
        if isinstance(positions, Gaussians):
            positions = super().forward(Gaussians(mean=positions.mean / self.scale, cov=positions.cov / self.scale**2))
            if self.normalize:
                positions.mean = (positions.mean + 2.0) / 4.0
                positions.cov = positions.cov / 16.0
        elif isinstance(positions, GaussiansStd):
            positions = super().forward(GaussiansStd(mean=positions.mean / self.scale, std=positions.std / self.scale))
            if self.normalize:
                positions.mean = (positions.mean + 2.0) / 4.0
                positions.std = positions.std / 4.0
        else:
            positions = super().forward(positions / self.scale)
            if self.normalize:
                positions = (positions + 2.0) / 4.0
        return positions
