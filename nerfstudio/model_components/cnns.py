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

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Abstract Residual Block class."""

    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        if in_dim != dim:
            self.res_branch = nn.Conv2d(in_dim, dim, kernel_size=1)
        else:
            self.res_branch = nn.Identity()
        self.main_branch = nn.Identity()
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_activation(self.res_branch(x) + self.main_branch(x))


class BasicBlock(ResidualBlock):
    """Basic residual block."""

    def __init__(self, in_dim: int, dim: int, kernel_size: int, padding: int, use_bn: bool = False):
        super().__init__(in_dim, dim)
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
        )


class BottleneckBlock(ResidualBlock):
    """Simple residual bottleneck block."""

    def __init__(
        self, in_dim: int, dim: int, kernel_size: int, padding: int, channel_multiplier: int = 1, use_bn: bool = False
    ):
        super().__init__(in_dim, dim)
        mid_dim = channel_multiplier * dim
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1),
            nn.BatchNorm2d(mid_dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim, kernel_size=1),
            nn.BatchNorm2d(mid_dim) if use_bn else nn.Identity(),
        )
