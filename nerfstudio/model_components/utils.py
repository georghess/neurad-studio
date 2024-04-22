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

from typing import Union

import torch
from torch import Tensor, nn


class SigmoidDensity(nn.Module):
    """Learnable sigmoid density"""

    def __init__(self, init_val, beta_min=0.0001, learnable_beta=False):
        super().__init__()
        self.register_buffer("beta_min", torch.tensor(beta_min))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=learnable_beta))

    def forward(self, sdf: Tensor, beta: Union[Tensor, None] = None) -> Tensor:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        # negtive sdf will have large density
        return torch.sigmoid(-sdf * beta)

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta
