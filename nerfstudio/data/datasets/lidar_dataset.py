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
Lidar Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch

from nerfstudio.cameras.lidars import Lidars
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import Dataset


class LidarDataset(Dataset):
    """Dataset that returns lidar data.

    Args:
        dataparser_outputs: description of where and how to read data.
        downsample_factor: The downsample factor for the dataparser outputs (lidar only)
    """

    exclude_batch_keys_from_device: List[str] = []
    lidars: Lidars

    def __init__(self, dataparser_outputs: DataparserOutputs, downsample_factor: float = 1.0) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor
        self._dataparser_outputs = dataparser_outputs
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.lidars: Lidars = deepcopy(self.metadata.pop("lidars"))
        self.point_clouds = deepcopy(self.metadata.pop("point_clouds"))
        self.has_masks = dataparser_outputs.mask_filenames is not None
        self.scene_box = deepcopy(dataparser_outputs.scene_box)

    def __len__(self):
        return len(self.lidars)

    # pylint: disable=no-self-use
    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        return {}

    def get_data(self, lidar_idx: int) -> Dict:
        """Returns the LidarDataset data as a dictionary.

        Args:
            lidar_idx: The lidar index in the dataset.
        """
        data = {"lidar_idx": lidar_idx}
        point_cloud = self.point_clouds[lidar_idx]
        if point_cloud.shape[-1] == 4:  # no time offset
            # add column of zeros
            point_cloud = torch.cat([point_cloud, torch.zeros(point_cloud.shape[0], 1)], dim=1)
        data["lidar"] = point_cloud

        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def __getitem__(self, lidar_idx: int) -> Dict:
        data = self.get_data(lidar_idx)
        return data
