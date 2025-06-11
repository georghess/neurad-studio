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
Camera Models
"""
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import Parameter

import nerfstudio.utils.math
import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.lidar_elevation_mappings import (
    PANDAR64_ELEVATION_MAPPING,
    VELODYNE_128_ELEVATION_MAPPING,
    VELODYNE_HDL32E_ELEVATION_MAPPING,
    VELODYNE_VLP32C_ELEVATION_MAPPING,
)
from nerfstudio.utils.misc import strtobool, torch_compile
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[torch.device, str]  # pylint: disable=invalid-name

HORIZONTAL_BEAM_DIVERGENCE = 3.0e-3  # radians, or meters at a distance of 1m
VERTICAL_BEAM_DIVERGENCE = 1.5e-3  # radians, or meters at a distance of 1m


class LidarType(Enum):
    """Supported Lidar Types."""

    VELODYNE16 = auto()
    VELODYNE_HDL32E = auto()
    VELODYNE_VLP32C = auto()
    VELODYNE64E = auto()
    VELODYNE128 = auto()
    PANDAR64 = auto()
    WOD64 = auto()


LIDAR_MODEL_TO_TYPE = {
    "VELODYNE16": LidarType.VELODYNE16,
    "VELODYNE_HDL32E": LidarType.VELODYNE_HDL32E,
    "VELODYNE_VLP32C": LidarType.VELODYNE_VLP32C,
    "VELODYNE64E": LidarType.VELODYNE64E,
    "VELODYNE128": LidarType.VELODYNE128,
    "PANDAR64": LidarType.PANDAR64,
    "WOD64": LidarType.WOD64,
}


@dataclass(init=False)
class Lidars(TensorDataclass):
    """Dataparser outputs for the lidar dataset and the ray generator.

    If a single value is provided, it is broadcasted to all lidars.

    Args:
        lidar_to_worlds: Lidar to world matrices. Tensor of per-image l2w matrices, in [R | t] format
        lidar_type: Type of lidar sensor. This will be an int corresponding to the LidarType enum.
        assume_ego_compensated: Whether or not to assume that points are ego-compensated.
        times: Timestamps for each lidar
        n_points: Number of points in each lidar
        metadata: Additional metadata or data needed for interpolation, will mimic shape of the lidars
            and will be broadcasted to the rays generated from any derivative RaySamples we create with this
    """

    lidar_to_worlds: Float[Tensor, "*num_lidars 3 4"]
    lidar_type: Float[Tensor, "*num_lidars 1"]
    times: Optional[Int[Tensor, "*num_lidars 1"]]
    metadata: Optional[Dict]
    horizontal_beam_divergence: Float[Tensor, "*num_lidars 1"]
    vertical_beam_divergence: Float[Tensor, "*num_lidars 1"]
    azimuths: Optional[Float[Tensor, "*num_lidars n_azimuth_channels 1"]]
    elevations: Optional[Float[Tensor, "*num_lidars n_elevation_channels 1"]]

    def __init__(
        self,
        lidar_to_worlds: Float[Tensor, "*batch_l2ws 3 4"],
        lidar_type: Optional[
            Union[
                Int[Tensor, "*batch_lidar_types 1"],
                int,
                List[LidarType],
                LidarType,
            ]
        ] = LidarType.VELODYNE_VLP32C,
        assume_ego_compensated: bool = True,
        times: Optional[Float[Tensor, "*num_lidars"]] = None,
        metadata: Optional[Dict] = None,
        horizontal_beam_divergence: Optional[Union[Float, Float[Tensor, "*num_lidars 1"]]] = None,
        vertical_beam_divergence: Optional[Union[Float, Float[Tensor, "*num_lidars 1"]]] = None,
        valid_lidar_distance_threshold: float = 1e3,
        azimuths: Optional[Float[Tensor, "*num_lidars n_azimuth_channels 1"]] = None,
        elevations: Optional[Float[Tensor, "*num_lidars n_elevation_channels 1"]] = None,
    ) -> None:
        """Initializes the Lidars object.

        Note on Input Tensor Dimensions: All of these tensors have items of dimensions Shaped[Tensor, "3 4"]
        (in the case of the c2w matrices), Shaped[Tensor, "6"] (in the case of distortion params), or
        Shaped[Tensor, "1"] (in the case of the rest of the elements). The dimensions before that are
        considered the batch dimension of that tensor (batch_c2ws, batch_fxs, etc.). We will broadcast
        all the tensors to be the same batch dimension. This means you can use any combination of the
        input types in the function signature and it won't break. Your batch size for all tensors
        must be broadcastable to the same size, and the resulting number of batch dimensions will be
        the batch dimension with the largest number of dimensions.
        """

        # This will notify the tensordataclass that we have a field with more than 1 dimension
        self._field_custom_dimensions = {"lidar_to_worlds": 2}

        self.lidar_to_worlds = lidar_to_worlds

        # @dataclass's post_init will take care of broadcasting
        self.lidar_type = self._init_get_lidar_type(lidar_type)
        self.times = self._init_get_times(times)

        self.metadata = metadata

        self.horizontal_beam_divergence = self._init_get_beam_divergence(
            horizontal_beam_divergence, HORIZONTAL_BEAM_DIVERGENCE
        )
        self.vertical_beam_divergence = self._init_get_beam_divergence(
            vertical_beam_divergence, VERTICAL_BEAM_DIVERGENCE
        )
        self.azimuths = self._init_get_azimuths(azimuths)
        self.elevations = self._init_get_elevations(elevations)

        self.__post_init__()  # This will do the dataclass post_init and broadcast all the tensors

        self._use_nerfacc = strtobool(os.environ.get("INTERSECT_WITH_NERFACC", "TRUE"))
        self.assume_ego_compensated = assume_ego_compensated

        self.valid_lidar_distance_threshold = valid_lidar_distance_threshold

    def _init_get_beam_divergence(
        self,
        beam_divergence: Union[None, Float, Float[Tensor, "*num_lidars 1"]],
        default: float,
    ) -> Float[Tensor, "*num_lidars 1"]:
        if beam_divergence is None:
            beam_divergence = torch.ones_like(self.lidar_type) * default
        elif isinstance(beam_divergence, float):
            beam_divergence = torch.ones_like(self.lidar_type) * beam_divergence
        elif isinstance(beam_divergence, torch.Tensor):
            if beam_divergence.ndim == 0 or beam_divergence.shape[-1] != 1:
                beam_divergence = beam_divergence.unsqueeze(-1)
        else:
            raise ValueError(f"beam_divergence must be None, float, or tensor, got {type(beam_divergence)}")

        return beam_divergence

    def _init_get_lidar_type(
        self,
        lidar_type: Union[
            Int[Tensor, "*batch_lidar_types 1"], Int[Tensor, "*batch_lidar_types"], int, List[LidarType], LidarType
        ],
    ) -> Int[Tensor, "*num_lidars 1"]:
        """
        Parses the __init__() argument lidar_type

        Lidar Type Calculation:
        If LidarType, convert to int and then to tensor, then broadcast to all lidars
        If List of LidarTypes, convert to ints and then to tensor, then broadcast to all lidars
        If int, first go to tensor and then broadcast to all lidars
        If tensor, broadcast to all lidars

        Args:
            lidar_type: lidar_type argument from __init__()
        """
        if isinstance(lidar_type, LidarType):
            lidar_type = torch.tensor([lidar_type.value], device=self.device)
        elif isinstance(lidar_type, List) and isinstance(lidar_type[0], LidarType):
            lidar_type = torch.tensor([[c.value] for c in lidar_type], device=self.device)
        elif isinstance(lidar_type, int):
            lidar_type = torch.tensor([lidar_type], device=self.device)
        elif isinstance(lidar_type, torch.Tensor):
            assert not torch.is_floating_point(
                lidar_type
            ), f"lidar_type tensor must be of type int, not: {lidar_type.dtype}"
            lidar_type = lidar_type.to(self.device)
            if lidar_type.ndim == 0 or lidar_type.shape[-1] != 1:
                lidar_type = lidar_type.unsqueeze(-1)
            # assert torch.all(
            #     lidar_type.view(-1)[0] == lidar_type
            # ), "Batched lidars of different lidar_types will be allowed in the future."
        else:
            raise ValueError(
                'Invalid lidar_type. Must be LidarType, List[LidarType], int, or torch.Tensor["num_lidars"]. \
                    Received: '
                + str(type(lidar_type))
            )
        return lidar_type

    def _init_get_times(self, times: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if times is None:
            times = None
        elif isinstance(times, torch.Tensor):
            if times.ndim == 0 or times.shape[-1] != 1:
                times = times.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"times must be None or a tensor, got {type(times)}")

        return times

    def _init_get_azimuths(self, azimuths: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if azimuths is None:
            azimuths = None
        elif isinstance(azimuths, torch.Tensor):
            if azimuths.ndim == 0 or azimuths.shape[-1] != 1:
                azimuths = azimuths.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"azimuths must be None or a tensor, got {type(azimuths)}")

        return azimuths

    def _init_get_elevations(self, elevations: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if elevations is None:
            elevations = None
        elif isinstance(elevations, torch.Tensor):
            if elevations.ndim == 0 or elevations.shape[-1] != 1:
                elevations = elevations.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"elevations must be None or a tensor, got {type(elevations)}")

        return elevations

    @property
    def device(self) -> TORCH_DEVICE:
        """Returns the device that the camera is on."""
        return self.lidar_to_worlds.device

    @property
    def is_jagged(self) -> bool:
        """
        Returns whether or not the lidars are "jagged" (i.e. number of point vary between lidars)
        TODO: base assumption is that all lidars have different number of points
        """
        return all(self.n_points.view(-1)[0] == self.n_points)

    @property
    def width(self) -> int:
        """Returns the width of the lidar."""
        return self.azimuths.shape[1] if self.azimuths is not None else 0

    @property
    def height(self) -> int:
        """Returns the height of the lidar."""
        return self.elevations.shape[1] if self.elevations is not None else 0

    def generate_rays(  # pylint: disable=too-many-statements
        self,
        lidar_indices: Union[Int[Tensor, "*num_rays num_lidars_batch_dims"], int],
        coords: Optional[Float[Tensor, "*num_rays 2"]] = None,
        lidar_opt_to_lidar: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
        keep_shape: Optional[bool] = None,
        aabb_box: Optional[SceneBox] = None,
        points: Optional[Float[Tensor, "*num_rays point_dim"]] = None,
    ) -> RayBundle:
        """Generates rays for the given camera indices.

        This function will standardize the input arguments and then call the _generate_rays_from_coords function
        to generate the rays. Our goal is to parse the arguments and then get them into the right shape:
            - lidar_indices: (num_rays:..., num_lidars_batch_dims)
            - coords: (num_rays:..., 2)
            - lidar_opt_to_lidar: (num_rays:..., 3, 4) or None

        Read the docstring for _generate_rays_from_coords for more information on how we generate the rays
        after we have standardized the arguments.

        We are only concerned about different combinations of lidar_indices and coords matrices, and the following
        are the 4 cases we have to deal with:
            1. isinstance(lidar_indices, int) and coords == None
                - In this case we broadcast our lidar_indices / coords shape (h, w, 1 / 2 respectively)
            2. isinstance(lidar_indices, int) and coords != None
                - In this case, we broadcast lidar_indices to the same batch dim as coords
            3. not isinstance(lidar_indices, int) and coords == None
                - In this case, we will need to set coords so that it is of shape (h, w, num_rays, 2), and broadcast
                    all our other args to match the new definition of num_rays := (h, w) + num_rays
            4. not isinstance(lidar_indices, int) and coords != None
                - In this case, we have nothing to do, only check that the arguments are of the correct shape

        There is one more edge case we need to be careful with: when we have "jagged lidars" (ie: different heights
        and widths for each camera). This isn't problematic when we specify coords, since coords is already a tensor.
        When coords == None (ie: when we render out the whole image associated with this camera), we run into problems
        since there's no way to stack each coordinate map as all coordinate maps are all different shapes. In this case,
        we will need to flatten each individual coordinate map and concatenate them, giving us only one batch dimension,
        regardless of the number of prepended extra batch dimensions in the lidar_indices tensor.


        Args:
            lidar_indices: Camera indices of the flattened lidars object to generate rays for.
            coords: Coordinates of the pixels to generate rays for. If None, the full image will be rendered.
            lidar_opt_to_lidar: Optional transform for the camera to world matrices.
            distortion_params_delta: Optional delta for the distortion parameters.
            keep_shape: If None, then we default to the regular behavior of flattening if lidars is jagged, otherwise
                keeping dimensions. If False, we flatten at the end. If True, then we keep the shape of the
                lidar_indices and coords tensors (if we can).
            disable_distortion: If True, disables distortion.
            aabb_box: if not None will calculate nears and fars of the ray according to aabb box intesection

        Returns:
            Rays for the given camera indices and coords.
        """
        # Check the argument types to make sure they're valid and all shaped correctly
        assert isinstance(lidar_indices, (torch.Tensor, int)), "lidar_indices must be a tensor or int"
        assert (
            coords is None or isinstance(coords, torch.Tensor) or points is not None
        ), "coords must be a tensor or None, or points must be specified"
        assert lidar_opt_to_lidar is None or isinstance(lidar_opt_to_lidar, torch.Tensor)
        if isinstance(lidar_indices, torch.Tensor) and isinstance(coords, torch.Tensor):
            num_rays_shape = lidar_indices.shape[:-1]
            errormsg = "Batch dims of inputs must match when inputs are all tensors"
            if coords is not None:
                assert coords.shape == num_rays_shape, errormsg
            assert lidar_opt_to_lidar is None or lidar_opt_to_lidar.shape[:-2] == num_rays_shape, errormsg

        # If zero dimensional, we need to unsqueeze to get a batch dimension and then squeeze later
        if not self.shape:
            lidars = self.reshape((1,))
            assert torch.all(
                torch.tensor(lidar_indices == 0) if isinstance(lidar_indices, int) else lidar_indices == 0
            ), "Can only index into single camera with no batch dimensions if index is zero"
        else:
            lidars = self

        # If the camera indices are an int, then we need to make sure that the camera batch is 1D
        if isinstance(lidar_indices, int):
            assert (
                len(lidars.shape) == 1
            ), "lidar_indices must be a tensor if lidars are batched with more than 1 batch dimension"
            lidar_indices = torch.tensor([lidar_indices], device=lidars.device)

        assert lidar_indices.shape[-1] == len(
            lidars.shape
        ), "lidar_indices must have shape (num_rays:..., num_lidars_batch_dims)"

        assert (coords is None) or (points is None), "Cannot specify both coords and points"
        if points is not None:
            raybundle = lidars._generate_rays_from_points(lidar_indices, points, lidar_opt_to_lidar)
        elif coords is not None:
            raise NotImplementedError("No lidar modeling yet")
        else:
            raise NotImplementedError("No lidar modeling yet")

        # If we have mandated that we don't keep the shape, then we flatten
        if keep_shape is False:
            raybundle = raybundle.flatten()

        if aabb_box:
            with torch.no_grad():
                tensor_aabb = Parameter(aabb_box.aabb.flatten(), requires_grad=False)

                rays_o = raybundle.origins.contiguous()
                rays_d = raybundle.directions.contiguous()

                tensor_aabb = tensor_aabb.to(rays_o.device)
                shape = rays_o.shape

                rays_o = rays_o.reshape((-1, 3))
                rays_d = rays_d.reshape((-1, 3))

                t_min, t_max = nerfstudio.utils.math.intersect_aabb(rays_o, rays_d, tensor_aabb)

                t_min = t_min.reshape([shape[0], shape[1], 1])
                t_max = t_max.reshape([shape[0], shape[1], 1])

                raybundle.nears = t_min
                raybundle.fars = t_max

        # TODO: We should have to squeeze the last dimension here if we started with zero batch dims, but never have to,
        # so there might be a rogue squeeze happening somewhere, and this may cause some unintended behaviour
        # that we haven't caught yet with tests
        return raybundle

    # pylint: disable=too-many-statements

    def _generate_rays_from_points(
        self,
        lidar_indices: Int[Tensor, "*num_rays num_lidars_batch_dims"],
        points: Float[Tensor, "*num_points point_dim"],
        lidar_opt_to_lidar: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
    ) -> RayBundle:
        # Make sure we're on the right devices
        lidar_indices = lidar_indices.to(self.device)
        points = points.to(self.device)

        # Checking to make sure everything is of the right shape and type
        num_rays_shape = lidar_indices.shape[:-1]
        assert lidar_indices.shape == num_rays_shape + (self.ndim,)
        assert lidar_opt_to_lidar is None or lidar_opt_to_lidar.shape == num_rays_shape + (3, 4)
        l2w = self.lidar_to_worlds[lidar_indices.squeeze(-1)]  # (..., 3, 4)
        assert l2w.shape == num_rays_shape + (3, 4)
        if lidar_opt_to_lidar is not None:
            l2w = pose_utils.multiply(l2w, lidar_opt_to_lidar)

        points_world = transform_points_pairwise(points[..., :3], l2w)
        origins = l2w[..., :3, 3]  # (..., 3)
        assert origins.shape == num_rays_shape + (3,)
        if points.shape[-1] >= 5 and self.metadata and "velocities" in self.metadata:
            # Offset the point origins according to timediff and velocity
            origins = origins + points[..., 4:5] * self.metadata["velocities"][lidar_indices.squeeze(-1)]
            if not self.assume_ego_compensated:
                # offset the world points according to velocity too
                points_world = points_world + points[..., 4:5] * self.metadata["velocities"][lidar_indices.squeeze(-1)]
        directions = points_world - origins
        directions, distance = camera_utils.normalize_with_norm(directions, -1)
        assert directions.shape == num_rays_shape + (3,)

        # norms of the vector going between adjacent coords, giving us dx and dy per output ray
        dx = self.horizontal_beam_divergence[lidar_indices.squeeze(-1)]  # ("num_rays":...,)
        dy = self.vertical_beam_divergence[lidar_indices.squeeze(-1)]  # ("num_rays":...,)
        pixel_area = dx * dy  # ("num_rays":..., 1)
        assert pixel_area.shape == num_rays_shape + (1,)

        metadata = (
            self._apply_fn_to_dict(self.metadata, lambda x: x[lidar_indices.squeeze(-1)])
            if self.metadata is not None
            else None
        )
        if metadata is not None:
            metadata["directions_norm"] = distance.detach()
        else:
            metadata = {"directions_norm": distance.detach()}
        metadata["is_lidar"] = torch.ones_like(distance, dtype=torch.bool)
        metadata["did_return"] = distance.detach() < self.valid_lidar_distance_threshold

        times = self.times[lidar_indices, 0] if self.times is not None else None
        times = times + points[..., 4:5]

        return RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=lidar_indices,
            times=times,
            metadata=metadata,
            fars=torch.ones_like(pixel_area) * 1_000_000,  # TODO: is this cheating?
        )

    def _generate_rays_from_coords(
        self,
        lidar_indices: Int[Tensor, "*num_rays num_lidars_batch_dims"],
        coords: Float[Tensor, "num_rays 2"],
        lidar_opt_to_lidar: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
    ) -> RayBundle:
        """Generates rays for the given camera indices and coords where self isn't jagged

        This is a fairly complex function, so let's break this down slowly.

        Shapes involved:
            - num_rays: This is your output raybundle shape. It dictates the number and shape of the rays generated
            - num_lidars_batch_dims: This is the number of dimensions of our camera

        Args:
            lidar_indices: Camera indices of the flattened lidars object to generate rays for.
                The shape of this is such that indexing into lidar_indices["num_rays":...] will return the
                index into each batch dimension of the camera in order to get the correct camera specified by
                "num_rays".

                Example:
                    >>> lidars = lidars(...)
                    >>> lidars.shape
                        (2, 3, 4)

                    >>> lidar_indices = torch.tensor([0, 0, 0]) # We need an axis of length 3 since lidars.ndim == 3
                    >>> lidar_indices.shape
                        (3,)
                    >>> coords = torch.tensor([1,1])
                    >>> coords.shape
                        (2,)
                    >>> out_rays = lidars.generate_rays(lidar_indices=lidar_indices, coords = coords)
                        # This will generate a RayBundle with a single ray for the
                        # camera at lidars[0,0,0] at image coordinates (1,1), so out_rays.shape == ()
                    >>> out_rays.shape
                        ()

                    >>> lidar_indices = torch.tensor([[0,0,0]])
                    >>> lidar_indices.shape
                        (1, 3)
                    >>> coords = torch.tensor([[1,1]])
                    >>> coords.shape
                        (1, 2)
                    >>> out_rays = lidars.generate_rays(lidar_indices=lidar_indices, coords = coords)
                        # This will generate a RayBundle with a single ray for the
                        # camera at lidars[0,0,0] at point (1,1), so out_rays.shape == (1,)
                        # since we added an extra dimension in front of lidar_indices
                    >>> out_rays.shape
                        (1,)

                If you want more examples, check tests/lidars/test_lidars and the function check_generate_rays_shape

                The bottom line is that for lidar_indices: (num_rays:..., num_lidars_batch_dims), num_rays is the
                output shape and if you index into the output RayBundle with some indices [i:...], if you index into
                lidar_indices with lidar_indices[i:...] as well, you will get a 1D tensor containing the batch
                indices into the original lidars object corresponding to that ray (ie: you will get the camera
                from our batched lidars corresponding to the ray at RayBundle[i:...]).

            coords: Coordinates of the pixels to generate rays for. If None, the full image will be rendered, meaning
                height and width get prepended to the num_rays dimensions. Indexing into coords with [i:...] will
                get you the image coordinates [x, y] of that specific ray located at output RayBundle[i:...].

            lidar_opt_to_lidar: Optional transform for the camera to world matrices.
                In terms of shape, it follows the same rules as coords, but indexing into it with [i:...] gets you
                the 2D camera to world transform matrix for the camera optimization at RayBundle[i:...].

            distortion_params_delta: Optional delta for the distortion parameters.
                In terms of shape, it follows the same rules as coords, but indexing into it with [i:...] gets you
                the 1D tensor with the 6 distortion parameters for the camera optimization at RayBundle[i:...].

            disable_distortion: If True, disables distortion.

        Returns:
            Rays for the given camera indices and coords. RayBundle.shape == num_rays
        """
        raise NotImplementedError("No lidar modeling yet")


def transform_points(points, transform):
    """Transforms points by a transform."""
    points = points.clone()
    rotations = transform[..., :3, :3]
    translations = transform[..., :3, 3]
    points[:, :3] = points[:, :3] @ rotations.swapaxes(-2, -1) + translations
    return points


@torch_compile(dynamic=True, mode="reduce-overhead", backend="eager")
def transform_points_pairwise(points, transforms, with_translation=True):
    """Transform points by a pairwise transform."""
    # points: (*, 3)
    # transforms: (*, 4, 4)
    # return: (*, 3)
    rotations = transforms[..., :3, :3]
    translations = transforms[..., :3, 3]

    # Perform batch matrix multiplication
    rotated_points = torch.bmm(rotations.reshape(-1, 3, 3), points.reshape(-1, 3, 1)).reshape(*points.shape[:-1], 3)

    if with_translation:
        return rotated_points + translations
    else:
        return rotated_points


@torch_compile(dynamic=True, mode="reduce-overhead", backend="eager")
def transform_points_by_batch(points, transforms, with_translation=True):
    """Transform points by a batch of transforms."""
    # points: (N, 3)
    # transforms: (B, 4, 4)
    # return: (B, N, 3)
    batch_size = transforms.shape[0]
    points = points.clone().unsqueeze(0).repeat(batch_size, 1, 1)
    rotations = transforms[:, :3, :3]
    translations = transforms[:, :3, 3]

    if with_translation:
        return points @ rotations.permute(0, 2, 1) + translations.unsqueeze(1)
    else:
        return points @ rotations.permute(0, 2, 1)


def intensity_to_rgb(intensities: np.ndarray) -> np.ndarray:  # N -> N x 3
    """Converts intensities to RGB values."""
    # use log-scale for better visualization
    log_intensities = np.log(1 + intensities * 255) / np.log(256)
    return plt.cm.inferno(log_intensities)[:, :3]


def get_lidar_elevation_mapping(lidar_type: LidarType) -> dict:
    if lidar_type == LidarType.VELODYNE16:
        raise NotImplementedError("No elevation mapping for Velodyne 16")
    elif lidar_type == LidarType.VELODYNE_HDL32E:
        return VELODYNE_HDL32E_ELEVATION_MAPPING
    elif lidar_type == LidarType.VELODYNE_VLP32C:
        return VELODYNE_VLP32C_ELEVATION_MAPPING
    elif lidar_type == LidarType.VELODYNE64E:
        raise NotImplementedError("No elevation mapping for Velodyne 64E")
    elif lidar_type == LidarType.VELODYNE128:
        return VELODYNE_128_ELEVATION_MAPPING
    elif lidar_type == LidarType.PANDAR64:
        return PANDAR64_ELEVATION_MAPPING
    else:
        raise ValueError(f"Invalid lidar type: {lidar_type}")


def get_lidar_azimuth_resolution(lidar_type: LidarType) -> float:
    if lidar_type == LidarType.VELODYNE16:
        return 0.2
    elif lidar_type == LidarType.VELODYNE_HDL32E:
        return 1 / 3.0
    elif lidar_type == LidarType.VELODYNE_VLP32C:
        return 0.2
    elif lidar_type == LidarType.VELODYNE64E:
        return 0.09
    elif lidar_type == LidarType.VELODYNE128:
        return 0.2
    elif lidar_type == LidarType.PANDAR64:
        return 0.2
    else:
        raise ValueError(f"Invalid lidar type: {lidar_type}")


def get_lidar_relovution_time(lidar_type: LidarType) -> float:
    if lidar_type == LidarType.VELODYNE16:
        return 0.1
    elif lidar_type == LidarType.VELODYNE_HDL32E:
        return 0.1
    elif lidar_type == LidarType.VELODYNE_VLP32C:
        return 0.1
    elif lidar_type == LidarType.VELODYNE64E:
        return 0.1
    elif lidar_type == LidarType.VELODYNE128:
        return 0.1
    elif lidar_type == LidarType.PANDAR64:
        return 0.1
    else:
        raise ValueError(f"Invalid lidar type: {lidar_type}")
