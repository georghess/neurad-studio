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

"""
Some ray datastructures.
"""
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, overload

import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor

from nerfstudio.utils.math import Gaussians, GaussiansStd, conical_frustum_to_gaussian, multisampled_frustum_to_gaussian
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[str, torch.device]


@dataclass
class Frustums(TensorDataclass):
    """Describes region of space as a frustum."""

    origins: Float[Tensor, "*bs 3"]
    """xyz coordinate for ray origin."""
    directions: Float[Tensor, "*bs 3"]
    """Direction of ray."""
    starts: Float[Tensor, "*bs 1"]
    """Where the frustum starts along a ray."""
    ends: Float[Tensor, "*bs 1"]
    """Where the frustum ends along a ray."""
    pixel_area: Float[Tensor, "*bs 1"]
    """Projected area of pixel a distance 1 away from origin."""
    offsets: Optional[Float[Tensor, "*bs 3"]] = None
    """Offsets for each sample position"""

    def get_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "center" position of frustum. Not weighted by mass.

        Returns:
            xyz positions.
        """
        pos = self.origins + self.directions * (self.starts + self.ends) / 2
        if self.offsets is not None:
            pos = pos + self.offsets
        return pos

    def get_start_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "start" position of frustum.

        Returns:
            xyz positions.
        """
        return self.origins + self.directions * self.starts

    def set_offsets(self, offsets):
        """Sets offsets for this frustum for computing positions"""
        self.offsets = offsets

    def get_conical_gaussian_blob(self) -> Gaussians:
        """Calculates guassian approximation of conical frustum.

        Returns:
            Conical frustums approximated by gaussian distribution.
        """
        # Cone radius is set such that the square pixel_area matches the cone area.
        cone_radius = torch.sqrt(self.pixel_area / torch.pi)
        if self.offsets is not None:
            raise NotImplementedError()
        return conical_frustum_to_gaussian(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            radius=cone_radius,
        )

    def get_multisampled_gaussian_blob(self, rand: bool = False) -> GaussiansStd:
        """Calculates Gaussian approximation of conical frustum via multisampling.
        Returns:
            Conical frustums approximated by multisampled gaussian distribution.
        """
        # Cone radius is set such that the square pixel_area matches the cone area.
        cone_radius = torch.sqrt(self.pixel_area / torch.pi)
        if self.offsets is not None:
            raise NotImplementedError()
        return multisampled_frustum_to_gaussian(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            radius=cone_radius,
            rand=rand,
        )

    def get_fast_isotropic_gaussian(self, num_multisamples: int) -> GaussiansStd:
        """This is a sloppy but very fast approximation of the gaussian blob.

        Args:
            num_multisamples: Number of samples to use for the gaussian blob.

        """
        if self.offsets is not None:
            raise NotImplementedError()
        multisample_dist = (self.ends - self.starts) / (num_multisamples + 1)
        ts = torch.arange(1, num_multisamples + 1, device=self.ends.device, dtype=self.ends.dtype)
        t = self.starts + ts.unsqueeze(0) * multisample_dist
        mean = self.origins.unsqueeze(-2) + self.directions.unsqueeze(-2) * t.unsqueeze(-1)
        frust_crossection_area = self.pixel_area.unsqueeze(-2) * t.unsqueeze(-1).pow(2)
        std = (frust_crossection_area * multisample_dist.unsqueeze(-2)).pow(1 / 3)  # TODO: * 0.5?
        return GaussiansStd(mean=mean, std=std)

    @classmethod
    def get_mock_frustum(cls, device: Optional[TORCH_DEVICE] = "cpu") -> "Frustums":
        """Helper function to generate a placeholder frustum.

        Returns:
            A size 1 frustum with meaningless values.
        """
        return Frustums(
            origins=torch.ones((1, 3), device=device),
            directions=torch.ones((1, 3), device=device),
            starts=torch.ones((1, 1), device=device),
            ends=torch.ones((1, 1), device=device),
            pixel_area=torch.ones((1, 1), device=device),
        )


@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray"""

    frustums: Frustums
    """Frustums along ray."""
    camera_indices: Optional[Int[Tensor, "*bs 1"]] = None
    """Camera index."""
    deltas: Optional[Float[Tensor, "*bs 1"]] = None
    """"width" of each sample."""
    spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_to_euclidean_fn: Optional[Callable] = None
    """Function to convert bins to euclidean distance."""
    metadata: Optional[Dict[str, Shaped[Tensor, "*bs latent_dims"]]] = None
    """additional information relevant to generating ray samples"""
    times: Optional[Float[Tensor, "*batch 1"]] = None
    """Times at which rays are sampled"""

    def __init__(
        self,
        frustums: Frustums,
        camera_indices: Optional[Int[Tensor, "*bs 1"]] = None,
        deltas: Optional[Float[Tensor, "*bs 1"]] = None,
        spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_to_euclidean_fn: Optional[Callable] = None,
        metadata: Optional[Dict[str, Shaped[Tensor, "*bs latent_dims"]]] = None,
        times: Optional[Float[Tensor, "*batch 1"]] = None,
    ) -> None:
        # This will notify the tensordataclass that we have a field with more than 1 dimension
        self._field_custom_dimensions = {"world2box": 2}

        self.frustums = frustums
        self.camera_indices = camera_indices
        self.deltas = deltas
        self.spacing_starts = spacing_starts
        self.spacing_ends = spacing_ends
        self.spacing_to_euclidean_fn = spacing_to_euclidean_fn
        self.metadata = metadata
        self.times = times

        self.__post_init__()

    def get_weights(self, densities: Float[Tensor, "*batch num_samples 1"]) -> Float[Tensor, "*batch num_samples 1"]:
        """Return weights based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)

        return weights

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[True]
    ) -> Float[Tensor, "*batch num_samples 1"]:
        ...

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[False] = False
    ) -> Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]]:
        ...

    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: bool = False
    ) -> Union[
        Float[Tensor, "*batch num_samples 1"],
        Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]],
    ]:
        """Return weights based on predicted alphas
        Args:
            alphas: Predicted alphas (maybe from sdf) for samples along ray
            weights_only: If function should return only weights
        Returns:
            Tuple of weights and transmittance for each sample
        """

        transmittance = torch.cumprod(
            torch.cat([torch.ones((*alphas.shape[:1], 1, 1), device=alphas.device), 1.0 - alphas + 1e-7], 1), 1
        )

        weights = alphas * transmittance[:, :-1, :]
        if weights_only:
            return weights
        return weights, transmittance


@dataclass
class RayBundle(TensorDataclass):
    """A bundle of ray parameters."""

    # TODO(ethan): make sure the sizes with ... are correct
    origins: Float[Tensor, "*batch 3"]
    """Ray origins (XYZ)"""
    directions: Float[Tensor, "*batch 3"]
    """Unit ray direction vector"""
    pixel_area: Float[Tensor, "*batch 1"]
    """Projected area of pixel a distance 1 away from origin"""
    camera_indices: Optional[Int[Tensor, "*batch 1"]] = None
    """Camera indices"""
    nears: Optional[Float[Tensor, "*batch 1"]] = None
    """Distance along ray to start sampling"""
    fars: Optional[Float[Tensor, "*batch 1"]] = None
    """Rays Distance along ray to stop sampling"""
    metadata: Dict[str, Shaped[Tensor, "num_rays latent_dims"]] = field(default_factory=dict)
    """Additional metadata or data needed for interpolation, will mimic shape of rays"""
    times: Optional[Float[Tensor, "*batch 1"]] = None
    """Times at which rays are sampled"""
    termination_distances: Optional[Float[Tensor, "*batch 1"]] = None
    """Distance along ray where ray is measured to terminate. -1 if unknown and inf if ray does not terminate."""

    def set_camera_indices(self, camera_index: int) -> None:
        """Sets all the camera indices to a specific camera index.

        Args:
            camera_index: Camera index.
        """
        self.camera_indices = torch.ones_like(self.origins[..., 0:1]).long() * camera_index

    def __len__(self) -> int:
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def sample(self, num_rays: int) -> "RayBundle":
        """Returns a RayBundle as a subset of rays.

        Args:
            num_rays: Number of rays in output RayBundle

        Returns:
            RayBundle with subset of rays.
        """
        assert num_rays <= len(self)
        indices = random.sample(range(len(self)), k=num_rays)
        return self[indices]

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int) -> "RayBundle":
        """Flattens RayBundle and extracts chunk given start and end indices.

        Args:
            start_idx: Start index of RayBundle chunk.
            end_idx: End index of RayBundle chunk.

        Returns:
            Flattened RayBundle with end_idx-start_idx rays.

        """
        return self.flatten()[start_idx:end_idx]

    def get_ray_samples(
        self,
        bin_starts: Float[Tensor, "*bs num_samples 1"],
        bin_ends: Float[Tensor, "*bs num_samples 1"],
        spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_to_euclidean_fn: Optional[Callable] = None,
    ) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction. Currently samples uniformly.

        Args:
            bin_starts: Distance from origin to start of bin.
            bin_ends: Distance from origin to end of bin.

        Returns:
            Samples projected along ray.
        """
        deltas = bin_ends - bin_starts
        if self.camera_indices is not None:
            camera_indices = self.camera_indices[..., None]
        else:
            camera_indices = None

        shaped_raybundle_fields = self[..., None]

        frustums = Frustums(
            origins=shaped_raybundle_fields.origins,  # [..., 1, 3]
            directions=shaped_raybundle_fields.directions,  # [..., 1, 3]
            starts=bin_starts,  # [..., num_samples, 1]
            ends=bin_ends,  # [..., num_samples, 1]
            pixel_area=shaped_raybundle_fields.pixel_area,  # [..., 1, 1]
        )

        ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=camera_indices,  # [..., 1, 1]
            deltas=deltas,  # [..., num_samples, 1]
            spacing_starts=spacing_starts,  # [..., num_samples, 1]
            spacing_ends=spacing_ends,  # [..., num_samples, 1]
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            metadata=shaped_raybundle_fields.metadata,
            times=None if self.times is None else self.times[..., None],  # [..., 1, 1]
        )

        return ray_samples


@overload
def merge_raysamples(ray_samples: List[RaySamples], sort: Literal[True]) -> Tuple[RaySamples, Int[Tensor, "num_rays"]]:
    ...


@overload
def merge_raysamples(ray_samples: List[RaySamples], sort: Literal[False]) -> RaySamples:
    ...


def merge_raysamples(
    ray_samples: List[RaySamples], sort: bool = False
) -> Union[RaySamples, Tuple[RaySamples, Int[Tensor, "num_rays"]]]:
    """Merges two RaySamples objects, and make sure rays are nicely packed."""
    assert len(ray_samples[0].shape) == 1, "This function is overkill for batched ray samples, use .cat() instead."
    merged = ray_samples[0].cat(
        ray_samples[1:], dim=0, ignore_fields={"spacing_to_euclidean_fn", "spacing_starts", "spacing_ends"}
    )
    assert merged.metadata is not None and "ray_indices" in merged.metadata, "metadata must contain ray_indices"

    if sort:
        if merged.frustums.ends.device.type == "mps":
            ends = merged.frustums.ends.cpu().double()
            ray_indices = merged.metadata["ray_indices"].cpu()
        else:
            ends = merged.frustums.ends.double()
            ray_indices = merged.metadata["ray_indices"]
        score = ray_indices + (ends / (ends.max() + 1.0))
        sorting = torch.argsort(score[..., 0], dim=0)
        if merged.frustums.ends.device.type == "mps":
            sorting.to(merged.frustums.ends.device)
        return merged[sorting], sorting
    else:
        return merged
