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
Aggregate all the dataparser configs in one location.
"""

from typing import TYPE_CHECKING

import tyro

from nerfstudio.data.dataparsers.argoverse2_dataparser import Argoverse2DataParserConfig
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig
from nerfstudio.data.dataparsers.kittimot_dataparser import KittiMotDataParserConfig
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from nerfstudio.data.dataparsers.pandaset_dataparser import PandaSetDataParserConfig
from nerfstudio.data.dataparsers.zod_dataparser import ZodDataParserConfig
from nerfstudio.plugins.registry_dataparser import discover_dataparsers
from nerfstudio.utils.rich_utils import CONSOLE

dataparsers = {
    "kittimot-data": KittiMotDataParserConfig(),
    "nuscenes-data": NuScenesDataParserConfig(),
    "argoverse2-data": Argoverse2DataParserConfig(),
    "zod-data": ZodDataParserConfig(),
    "pandaset-data": PandaSetDataParserConfig(),
}

try:
    from nerfstudio.data.dataparsers.waymo_dataparser import WoDParserConfig

    dataparsers["wod-data"] = WoDParserConfig()
except ImportError:
    CONSOLE.print("Waymo dataparser has missing dependencies, please following installation instructions in README.md")

external_dataparsers, _ = discover_dataparsers()
all_dataparsers = {**dataparsers, **external_dataparsers}

if TYPE_CHECKING:
    # For static analysis (tab completion, type checking, etc), just use the base
    # dataparser config.
    DataParserUnion = DataParserConfig
else:
    # At runtime, populate a Union type dynamically. This is used by `tyro` to generate
    # subcommands in the CLI.
    DataParserUnion = tyro.extras.subcommand_type_from_defaults(
        all_dataparsers,
        prefix_names=False,  # Omit prefixes in subcommands themselves.
    )

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[DataParserUnion]  # Omit prefixes of flags in subcommands.
"""Union over possible dataparser types, annotated with metadata for tyro. This is
the same as the vanilla union, but results in shorter subcommand names."""
